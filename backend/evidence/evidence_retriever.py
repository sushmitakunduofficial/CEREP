"""
Evidence Retriever — orchestrator that queries all biological evidence
sources in parallel and returns a unified, deduplicated ``EvidenceReport``.

This is the primary entry-point for the CEREP evidence layer.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from backend.core.logging import get_logger
from backend.evidence.pubmed_client import PubMedClient, PubMedArticle
from backend.evidence.clinvar_client import ClinVarClient, ClinVarVariant
from backend.evidence.civic_client import CIViCClient, CIViCEvidenceItem
from backend.evidence.cosmic_client import COSMICClient, COSMICMutation
from backend.evidence.opentargets_client import (
    OpenTargetsClient,
    DiseaseAssociation,
    DrugTarget,
)

logger = get_logger("evidence.retriever")


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class EvidenceEntry:
    """A single piece of evidence from one source for one gene."""

    source: str                    # pubmed | clinvar | civic | cosmic | opentargets
    gene: str
    pmids: List[str] = field(default_factory=list)
    clinical_significance: str = ""
    evidence_level: str = ""       # A–E for CIViC, review stars for ClinVar, etc.
    summary: str = ""
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "gene": self.gene,
            "pmids": self.pmids,
            "clinical_significance": self.clinical_significance,
            "evidence_level": self.evidence_level,
            "summary": self.summary,
        }


@dataclass
class GeneEvidence:
    """Aggregated evidence across all sources for a single gene."""

    gene: str
    entries: List[EvidenceEntry] = field(default_factory=list)
    all_pmids: List[str] = field(default_factory=list)  # deduplicated union
    articles: List[PubMedArticle] = field(default_factory=list)
    variants: List[ClinVarVariant] = field(default_factory=list)
    civic_items: List[CIViCEvidenceItem] = field(default_factory=list)
    cosmic_mutations: List[COSMICMutation] = field(default_factory=list)
    disease_associations: List[DiseaseAssociation] = field(default_factory=list)
    drug_targets: List[DrugTarget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene": self.gene,
            "entries": [e.to_dict() for e in self.entries],
            "all_pmids": self.all_pmids,
            "articles": [a.to_dict() for a in self.articles],
            "variants": [v.to_dict() for v in self.variants],
            "civic_items": [c.to_dict() for c in self.civic_items],
            "cosmic_mutations": [m.to_dict() for m in self.cosmic_mutations],
            "disease_associations": [d.to_dict() for d in self.disease_associations],
            "drug_targets": [d.to_dict() for d in self.drug_targets],
        }


@dataclass
class EvidenceReport:
    """Complete evidence report across all queried genes."""

    genes: Dict[str, GeneEvidence] = field(default_factory=dict)
    total_pmids: int = 0
    total_entries: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genes": {k: v.to_dict() for k, v in self.genes.items()},
            "total_pmids": self.total_pmids,
            "total_entries": self.total_entries,
            "errors": self.errors,
        }


@dataclass
class PathStepEvidence:
    """Evidence for a single step (edge) in a knowledge-graph path."""

    source_node: str
    target_node: str
    relationship: str
    source_evidence: Optional[GeneEvidence] = None
    target_evidence: Optional[GeneEvidence] = None
    shared_pmids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relationship": self.relationship,
            "source_evidence": self.source_evidence.to_dict() if self.source_evidence else None,
            "target_evidence": self.target_evidence.to_dict() if self.target_evidence else None,
            "shared_pmids": self.shared_pmids,
        }


# ── Orchestrator ─────────────────────────────────────────────────────


class EvidenceRetriever:
    """Orchestrates parallel evidence retrieval across all sources.

    Parameters
    ----------
    ncbi_api_key:
        Optional NCBI API key (shared by PubMed and ClinVar).
    timeout:
        Per-source HTTP timeout in seconds.
    fetch_abstracts:
        Whether to eagerly fetch PubMed abstracts for discovered PMIDs.
    """

    def __init__(
        self,
        ncbi_api_key: Optional[str] = None,
        timeout: float = 10.0,
        fetch_abstracts: bool = False,
    ) -> None:
        self._pubmed = PubMedClient(api_key=ncbi_api_key, timeout=timeout)
        self._clinvar = ClinVarClient(api_key=ncbi_api_key, timeout=timeout)
        self._civic = CIViCClient(timeout=timeout)
        self._cosmic = COSMICClient()
        self._opentargets = OpenTargetsClient(timeout=timeout)
        self._fetch_abstracts = fetch_abstracts

    # ------------------------------------------------------------------
    # Primary entry-points
    # ------------------------------------------------------------------

    async def get_evidence(
        self,
        gene_names: Optional[List[str]] = None,
        variant_ids: Optional[List[str]] = None,
    ) -> EvidenceReport:
        """Retrieve evidence for a list of genes and/or variant IDs.

        All five sources are queried in parallel per gene.  Results are
        aggregated into a single ``EvidenceReport`` with deduplicated
        PMIDs.

        Parameters
        ----------
        gene_names:
            Gene symbols to look up (e.g. ``["BRCA1", "TP53"]``).
        variant_ids:
            Optional ClinVar variant IDs (not yet used; reserved for
            future variant-level queries).

        Returns
        -------
        EvidenceReport
            Unified report with per-gene evidence.
        """
        genes = gene_names or []
        report = EvidenceReport()

        if not genes:
            logger.warning("get_evidence called with no gene names")
            return report

        logger.info(
            "Starting parallel evidence retrieval",
            extra={"extra": {"genes": genes}},
        )

        # Query all genes concurrently
        tasks = [self._get_gene_evidence(gene) for gene in genes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        global_pmids: Set[str] = set()
        for gene, result in zip(genes, results):
            if isinstance(result, Exception):
                err = f"Evidence retrieval failed for {gene}: {result}"
                logger.error(err)
                report.errors.append(err)
                continue

            gene_ev: GeneEvidence = result
            report.genes[gene] = gene_ev
            global_pmids.update(gene_ev.all_pmids)
            report.total_entries += len(gene_ev.entries)

        report.total_pmids = len(global_pmids)

        logger.info(
            "Evidence retrieval complete",
            extra={"extra": {
                "genes_queried": len(genes),
                "total_entries": report.total_entries,
                "total_pmids": report.total_pmids,
                "errors": len(report.errors),
            }},
        )
        return report

    async def get_evidence_for_path(
        self,
        path_nodes: List[Dict[str, str]],
    ) -> List[PathStepEvidence]:
        """Retrieve evidence for each step in a knowledge-graph path.

        Parameters
        ----------
        path_nodes:
            Ordered list of path nodes, each a dict with at least
            ``{"id": "...", "node_type": "..."}`` and optionally
            ``{"relationship": "..."}``.  The path is walked pairwise.

        Returns
        -------
        list[PathStepEvidence]
            One entry per edge in the path.
        """
        if len(path_nodes) < 2:
            return []

        # Collect unique gene names from the path
        gene_names: List[str] = []
        for node in path_nodes:
            ntype = node.get("node_type", "").lower()
            if ntype == "gene":
                gene_names.append(node["id"])

        # Retrieve evidence for all genes in one batch
        report = await self.get_evidence(gene_names=gene_names) if gene_names else EvidenceReport()

        # Build per-step evidence
        steps: List[PathStepEvidence] = []
        for i in range(len(path_nodes) - 1):
            src = path_nodes[i]
            tgt = path_nodes[i + 1]
            rel = tgt.get("relationship", src.get("relationship", "ASSOCIATED_WITH"))

            src_ev = report.genes.get(src["id"])
            tgt_ev = report.genes.get(tgt["id"])

            # Shared PMIDs between source and target genes
            shared: List[str] = []
            if src_ev and tgt_ev:
                shared = sorted(set(src_ev.all_pmids) & set(tgt_ev.all_pmids))

            steps.append(PathStepEvidence(
                source_node=src["id"],
                target_node=tgt["id"],
                relationship=rel,
                source_evidence=src_ev,
                target_evidence=tgt_ev,
                shared_pmids=shared,
            ))

        return steps

    # ------------------------------------------------------------------
    # Per-gene parallel retrieval
    # ------------------------------------------------------------------

    async def _get_gene_evidence(self, gene_name: str) -> GeneEvidence:
        """Query all five sources for a single gene concurrently."""
        gene_ev = GeneEvidence(gene=gene_name)

        # Fire all sources in parallel
        (
            pubmed_pmids,
            clinvar_variants,
            civic_items,
            cosmic_mutations,
            ot_associations,
            ot_drugs,
        ) = await asyncio.gather(
            self._pubmed.search(gene_name),
            self._clinvar.get_variants(gene_name),
            self._civic.get_evidence(gene_name),
            self._cosmic.get_mutations(gene_name),
            self._opentargets.get_associations(gene_name),
            self._opentargets.get_drug_targets(gene_name),
        )

        all_pmids: Set[str] = set()

        # ── PubMed ────────────────────────────────────────────────────
        if pubmed_pmids:
            all_pmids.update(pubmed_pmids)
            gene_ev.entries.append(EvidenceEntry(
                source="pubmed",
                gene=gene_name,
                pmids=list(pubmed_pmids),
                summary=f"Found {len(pubmed_pmids)} recent publications for {gene_name}",
            ))
            if self._fetch_abstracts:
                gene_ev.articles = await self._pubmed.fetch_abstracts(pubmed_pmids)

        # ── ClinVar ───────────────────────────────────────────────────
        gene_ev.variants = clinvar_variants
        for var in clinvar_variants:
            gene_ev.entries.append(EvidenceEntry(
                source="clinvar",
                gene=gene_name,
                clinical_significance=var.clinical_significance,
                evidence_level=var.review_status,
                summary=f"{var.name}: {var.clinical_significance}",
                raw_data=var.to_dict(),
            ))

        # ── CIViC ─────────────────────────────────────────────────────
        gene_ev.civic_items = civic_items
        for item in civic_items:
            all_pmids.update(item.pmids)
            drugs_str = ", ".join(item.drugs) if item.drugs else "none"
            gene_ev.entries.append(EvidenceEntry(
                source="civic",
                gene=gene_name,
                pmids=item.pmids,
                clinical_significance=item.significance,
                evidence_level=item.evidence_level,
                summary=(
                    f"{item.variant} — {item.evidence_type}: "
                    f"{item.significance} ({item.disease}); drugs: {drugs_str}"
                ),
                raw_data=item.to_dict(),
            ))

        # ── COSMIC ────────────────────────────────────────────────────
        gene_ev.cosmic_mutations = cosmic_mutations
        for mut in cosmic_mutations:
            gene_ev.entries.append(EvidenceEntry(
                source="cosmic",
                gene=gene_name,
                clinical_significance=mut.significance,
                summary=(
                    f"{mut.mutation} ({mut.cosmic_id}): "
                    f"{mut.significance} in {mut.tissue_type} "
                    f"({mut.sample_count} samples)"
                ),
                raw_data=mut.to_dict(),
            ))

        # ── Open Targets ─────────────────────────────────────────────
        gene_ev.disease_associations = ot_associations
        gene_ev.drug_targets = ot_drugs

        if ot_associations:
            top = ot_associations[0]
            gene_ev.entries.append(EvidenceEntry(
                source="opentargets",
                gene=gene_name,
                evidence_level=f"score={top.association_score:.3f}",
                summary=(
                    f"Top association: {top.disease_name} "
                    f"(score {top.association_score:.3f}); "
                    f"{len(ot_associations)} disease associations total"
                ),
            ))

        if ot_drugs:
            drug_names = [d.drug_name for d in ot_drugs[:5]]
            gene_ev.entries.append(EvidenceEntry(
                source="opentargets",
                gene=gene_name,
                summary=(
                    f"{len(ot_drugs)} known drugs: "
                    f"{', '.join(drug_names)}"
                    f"{'…' if len(ot_drugs) > 5 else ''}"
                ),
            ))

        # ── Deduplicate PMIDs ─────────────────────────────────────────
        gene_ev.all_pmids = sorted(all_pmids)

        logger.info(
            "Gene evidence aggregated",
            extra={"extra": {
                "gene": gene_name,
                "entries": len(gene_ev.entries),
                "unique_pmids": len(gene_ev.all_pmids),
                "clinvar_variants": len(clinvar_variants),
                "civic_items": len(civic_items),
                "cosmic_mutations": len(cosmic_mutations),
                "ot_associations": len(ot_associations),
                "ot_drugs": len(ot_drugs),
            }},
        )
        return gene_ev
