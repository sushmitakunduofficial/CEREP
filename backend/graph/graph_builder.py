"""
Graph Builder — constructs the CEREP biomedical Knowledge Graph.

Supports two construction modes:
    1. Adapter-driven: ingests nodes/edges from BioCypher-style adapters (Reactome, STRING, GO, OpenTargets)
    2. Seed fallback: uses hardcoded BRCA seed data for instant testing (retained for backward compat)

The builder produces a GraphStore (NetworkXStore by default) that can be queried by the reasoning engine.
"""
import networkx as nx
from typing import Dict, Any, Optional, List

from backend.graph.schema import (
    NodeSchema, EdgeSchema, BiolinkCategory, BiolinkPredicate,
    Provenance, EdgeQualifiers, EvidenceLevel,
)
from backend.graph.graph_store import GraphStore, NetworkXStore, create_graph_store
from backend.graph.biolink_adapter import (
    BioCypherAdapter, get_all_adapters,
    ReactomeAdapter, STRINGAdapter, GOAdapter, OpenTargetsAdapter,
)
from backend.core.logging import get_logger

logger = get_logger("graph.builder")


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Seed Data (retained for fast testing / backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def _build_legacy_seed_nodes() -> list[NodeSchema]:
    """Original 24-node BRCA seed graph — now Biolink-compliant."""
    return [
        # Tumour suppressor genes
        NodeSchema("TP53", BiolinkCategory.GENE, "TP53", aliases=["p53", "TRP53"],
                   description="Tumour protein p53, master regulator of apoptosis"),
        NodeSchema("BRCA1", BiolinkCategory.GENE, "BRCA1", aliases=["BRCA-1", "BRCA_1", "RNF53"],
                   description="Breast cancer gene 1, DNA repair"),
        NodeSchema("BRCA2", BiolinkCategory.GENE, "BRCA2", aliases=["BRCA-2", "BRCA_2", "FANCD1"],
                   description="Breast cancer gene 2, homologous recombination"),
        NodeSchema("RB1", BiolinkCategory.GENE, "RB1", aliases=["ppRB", "RB"],
                   description="Retinoblastoma protein, cell cycle checkpoint"),
        NodeSchema("PTEN", BiolinkCategory.GENE, "PTEN", aliases=["MMAC1", "TEP1"],
                   description="Phosphatase and tensin homologue"),
        # Oncogenes
        NodeSchema("PIK3CA", BiolinkCategory.GENE, "PIK3CA", aliases=["PI3K", "p110alpha"],
                   description="PI3K catalytic subunit alpha"),
        NodeSchema("KRAS", BiolinkCategory.GENE, "KRAS", aliases=["Ki-RAS", "KRAS2"],
                   description="KRAS proto-oncogene"),
        NodeSchema("MYC", BiolinkCategory.GENE, "MYC", aliases=["c-MYC", "bHLHe39"],
                   description="MYC proto-oncogene"),
        NodeSchema("ERBB2", BiolinkCategory.GENE, "ERBB2", aliases=["HER2", "NEU", "CD340"],
                   description="Human epidermal growth factor receptor 2"),
        NodeSchema("CDH1", BiolinkCategory.GENE, "CDH1", aliases=["E-Cadherin", "ECAD"],
                   description="E-cadherin, invasion suppressor"),
        # Proteins
        NodeSchema("MDM2", BiolinkCategory.PROTEIN, "MDM2", description="p53-binding ubiquitin ligase"),
        NodeSchema("AKT1", BiolinkCategory.PROTEIN, "AKT1", description="AKT serine/threonine kinase 1"),
        NodeSchema("MTOR", BiolinkCategory.PROTEIN, "MTOR", description="Mechanistic target of rapamycin"),
        # Pathways
        NodeSchema("PI3K_AKT_MTOR", BiolinkCategory.PATHWAY, "PI3K/AKT/mTOR Pathway",
                   description="Central oncogenic signalling pathway"),
        NodeSchema("DNA_REPAIR", BiolinkCategory.PATHWAY, "Homologous Recombination",
                   description="DNA double-strand break repair"),
        NodeSchema("APOPTOSIS", BiolinkCategory.PATHWAY, "Apoptosis Pathway",
                   description="Programmed cell death cascade"),
        NodeSchema("CELL_CYCLE", BiolinkCategory.PATHWAY, "Cell Cycle Regulation",
                   description="G1/S and G2/M checkpoint control"),
        # Diseases
        NodeSchema("BRCA", BiolinkCategory.DISEASE, "Breast Cancer",
                   description="Invasive breast carcinoma"),
        NodeSchema("LUAD", BiolinkCategory.DISEASE, "Lung Adenocarcinoma"),
        NodeSchema("COAD", BiolinkCategory.DISEASE, "Colorectal Adenocarcinoma"),
        # Drugs
        NodeSchema("OLAPARIB", BiolinkCategory.DRUG, "Olaparib",
                   description="PARP inhibitor; approved for BRCA-mutant breast cancer"),
        NodeSchema("TRASTUZUMAB", BiolinkCategory.DRUG, "Trastuzumab",
                   description="HER2-targeted monoclonal antibody"),
        NodeSchema("TAMOXIFEN", BiolinkCategory.DRUG, "Tamoxifen",
                   description="Selective oestrogen receptor modulator"),
        NodeSchema("ALPELISIB", BiolinkCategory.DRUG, "Alpelisib",
                   description="PI3K alpha inhibitor for PIK3CA-mutant BC"),
        NodeSchema("EVEROLIMUS", BiolinkCategory.DRUG, "Everolimus",
                   description="mTOR inhibitor"),
    ]


def _build_legacy_seed_edges() -> list[EdgeSchema]:
    """Original 27 seed edges — now using Biolink predicates."""
    P = BiolinkPredicate
    prov = Provenance(source_database="seed_brca", evidence_level=EvidenceLevel.CURATED)
    return [
        # TP53 network
        EdgeSchema("TP53", "MDM2", P.POSITIVELY_REGULATES, 0.9, provenance=prov),
        EdgeSchema("MDM2", "TP53", P.NEGATIVELY_REGULATES, 0.9,
                   provenance=Provenance(source_database="seed_brca", pmids=["10499594"], evidence_level=EvidenceLevel.CURATED)),
        EdgeSchema("TP53", "APOPTOSIS", P.POSITIVELY_REGULATES, 1.0, provenance=prov),
        EdgeSchema("TP53", "CELL_CYCLE", P.REGULATES, 0.8, provenance=prov),
        EdgeSchema("TP53", "BRCA", P.GENE_ASSOCIATED_WITH_CONDITION, 1.0, provenance=prov),
        # BRCA1/2 network
        EdgeSchema("BRCA1", "DNA_REPAIR", P.POSITIVELY_REGULATES, 1.0, provenance=prov),
        EdgeSchema("BRCA2", "DNA_REPAIR", P.POSITIVELY_REGULATES, 1.0, provenance=prov),
        EdgeSchema("BRCA1", "BRCA", P.GENE_ASSOCIATED_WITH_CONDITION, 1.0, provenance=prov),
        EdgeSchema("BRCA2", "BRCA", P.GENE_ASSOCIATED_WITH_CONDITION, 1.0, provenance=prov),
        EdgeSchema("BRCA1", "TP53", P.PHYSICALLY_INTERACTS_WITH, 0.7, provenance=prov),
        EdgeSchema("OLAPARIB", "BRCA1", P.TARGETS, 0.9, provenance=prov),
        EdgeSchema("OLAPARIB", "BRCA2", P.TARGETS, 0.9, provenance=prov),
        # PIK3CA / AKT / mTOR axis
        EdgeSchema("PIK3CA", "AKT1", P.POSITIVELY_REGULATES, 0.95, provenance=prov),
        EdgeSchema("AKT1", "MTOR", P.POSITIVELY_REGULATES, 0.9, provenance=prov),
        EdgeSchema("PTEN", "PIK3CA", P.NEGATIVELY_REGULATES, 0.85, provenance=prov),
        EdgeSchema("AKT1", "PI3K_AKT_MTOR", P.ASSOCIATED_WITH, 1.0, provenance=prov),
        EdgeSchema("MTOR", "PI3K_AKT_MTOR", P.ASSOCIATED_WITH, 1.0, provenance=prov),
        EdgeSchema("PIK3CA", "BRCA", P.GENE_ASSOCIATED_WITH_CONDITION, 0.9, provenance=prov),
        EdgeSchema("ALPELISIB", "PIK3CA", P.TARGETS, 0.95, provenance=prov),
        EdgeSchema("EVEROLIMUS", "MTOR", P.TARGETS, 0.9, provenance=prov),
        # ERBB2 / HER2
        EdgeSchema("ERBB2", "AKT1", P.POSITIVELY_REGULATES, 0.8, provenance=prov),
        EdgeSchema("ERBB2", "PI3K_AKT_MTOR", P.ASSOCIATED_WITH, 0.85, provenance=prov),
        EdgeSchema("ERBB2", "BRCA", P.EXPRESSED_IN, 0.9, provenance=prov),
        EdgeSchema("TRASTUZUMAB", "ERBB2", P.TARGETS, 0.95, provenance=prov),
        # KRAS / MYC
        EdgeSchema("KRAS", "PI3K_AKT_MTOR", P.POSITIVELY_REGULATES, 0.75, provenance=prov),
        EdgeSchema("MYC", "CELL_CYCLE", P.POSITIVELY_REGULATES, 0.8, provenance=prov),
        # RB1 cell cycle
        EdgeSchema("RB1", "CELL_CYCLE", P.REGULATES, 0.85, provenance=prov),
        # CDH1 / invasion
        EdgeSchema("CDH1", "BRCA", P.ASSOCIATED_WITH, 0.8, provenance=prov),
        # BRCA drug treatment
        EdgeSchema("TAMOXIFEN", "BRCA", P.TARGETS, 0.8, provenance=prov),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main Builder
# ══════════════════════════════════════════════════════════════════════════════

class CERAPGraphBuilder:
    """Builds and maintains the CEREP biomedical knowledge graph.

    Construction modes:
        build_seed_graph()      — fast 24-node legacy graph (for testing)
        build_from_adapters()   — full research-grade KG via BioCypher adapters
    """

    def __init__(self, store: Optional[GraphStore] = None) -> None:
        self._store: GraphStore = store or NetworkXStore()
        # Backward compat: expose graph directly for legacy code
        self._graph_built = False

    # ── Properties (backward compat) ─────────────────────────────────────────

    @property
    def graph(self) -> nx.DiGraph:
        """Legacy accessor — returns the underlying NetworkX graph."""
        return self._store.to_networkx()

    @property
    def store(self) -> GraphStore:
        """Access the underlying GraphStore directly."""
        return self._store

    # ── Construction: Legacy Seed ─────────────────────────────────────────────

    def build_seed_graph(self) -> nx.DiGraph:
        """Populate graph from curated BRCA seed data (legacy 24-node graph)."""
        for node in _build_legacy_seed_nodes():
            self._store.add_node(node)
        for edge in _build_legacy_seed_edges():
            self._store.add_edge(edge)
        self._graph_built = True
        logger.info(
            "Seed graph built",
            extra={"extra": {"nodes": self._store.node_count(),
                             "edges": self._store.edge_count()}}
        )
        return self._store.to_networkx()

    # ── Construction: Adapter-driven ─────────────────────────────────────────

    def build_from_adapters(
        self,
        adapters: Optional[List[BioCypherAdapter]] = None,
        include_seed: bool = False,
    ) -> nx.DiGraph:
        """Build a research-grade KG by ingesting from BioCypher adapters.

        Args:
            adapters: List of adapters to use. Defaults to all registered adapters.
            include_seed: If True, also load legacy seed data first.
        """
        if include_seed:
            self.build_seed_graph()

        adapters = adapters or get_all_adapters()

        total_nodes = 0
        total_edges = 0
        adapter_stats: Dict[str, Dict[str, int]] = {}

        for adapter in adapters:
            adapter_name = adapter.name
            node_count = 0
            edge_count = 0

            logger.info(f"Ingesting from adapter: {adapter_name} v{adapter.version}")

            # Ingest nodes
            for node in adapter.get_nodes():
                if not self._store.has_node(node.id):
                    self._store.add_node(node)
                    node_count += 1
                else:
                    # Merge: update aliases and xrefs from this adapter
                    existing = self._store.get_node(node.id)
                    if existing and isinstance(self._store, NetworkXStore):
                        existing_aliases = set(existing.get("aliases", []))
                        new_aliases = existing_aliases | set(node.aliases)
                        self._store.graph.nodes[node.id]["aliases"] = list(new_aliases)
                        if node.xrefs:
                            existing_xrefs = existing.get("xrefs", {})
                            existing_xrefs.update(node.xrefs)
                            self._store.graph.nodes[node.id]["xrefs"] = existing_xrefs

            # Ingest edges
            for edge in adapter.get_edges():
                # Auto-create missing nodes as gene stubs
                if not self._store.has_node(edge.source):
                    stub = NodeSchema(
                        id=edge.source,
                        category=BiolinkCategory.GENE,
                        label=edge.source,
                        source=adapter_name,
                    )
                    self._store.add_node(stub)
                    node_count += 1
                if not self._store.has_node(edge.target):
                    stub = NodeSchema(
                        id=edge.target,
                        category=BiolinkCategory.GENE,
                        label=edge.target,
                        source=adapter_name,
                    )
                    self._store.add_node(stub)
                    node_count += 1

                self._store.add_edge(edge)
                edge_count += 1

            adapter_stats[adapter_name] = {"nodes": node_count, "edges": edge_count}
            total_nodes += node_count
            total_edges += edge_count
            logger.info(
                f"Adapter {adapter_name} complete",
                extra={"extra": {"nodes_added": node_count, "edges_added": edge_count}}
            )

        self._graph_built = True
        logger.info(
            "Full KG build complete",
            extra={"extra": {
                "total_nodes": self._store.node_count(),
                "total_edges": self._store.edge_count(),
                "new_nodes": total_nodes,
                "new_edges": total_edges,
                "adapter_stats": adapter_stats,
            }}
        )
        return self._store.to_networkx()

    # ── Dynamic insertion (from uploaded data) ────────────────────────────────

    def add_node(self, node: NodeSchema) -> None:
        self._store.add_node(node)

    def add_edge(self, edge: EdgeSchema) -> None:
        self._store.add_edge(edge)

    def add_gene_node(self, gene_id: str, aliases: Optional[list] = None,
                      category: BiolinkCategory = BiolinkCategory.GENE,
                      description: str = "Uploaded gene") -> None:
        """Add a gene node if it doesn't already exist."""
        if not self._store.has_node(gene_id):
            node = NodeSchema(
                id=gene_id, category=category, label=gene_id,
                aliases=aliases or [], description=description,
                source="uploaded",
            )
            self._store.add_node(node)

    def add_relationship(self, src: str, tgt: str,
                         predicate: BiolinkPredicate = BiolinkPredicate.ASSOCIATED_WITH,
                         weight: float = 0.5, source: str = "uploaded") -> None:
        """Add a relationship, auto-creating gene nodes if needed."""
        self.add_gene_node(src)
        self.add_gene_node(tgt)
        edge = EdgeSchema(
            src, tgt, predicate, weight=weight,
            provenance=Provenance(source_database=source),
        )
        self._store.add_edge(edge)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_entity_index(self) -> Dict[str, str]:
        """Return alias → canonical ID mapping."""
        if isinstance(self._store, NetworkXStore):
            return self._store.get_entity_index()
        return {}

    def resolve_alias(self, name: str) -> Optional[str]:
        """Return canonical ID for a name/alias, or None if not in graph."""
        if isinstance(self._store, NetworkXStore):
            return self._store.resolve_alias(name)
        return name if self._store.has_node(name) else None

    def to_cytoscape_json(self) -> dict:
        """Export graph as Cytoscape.js-compatible JSON."""
        return self._store.to_cytoscape_json()

    def to_serializable(self) -> dict:
        """Export the full graph as a JSON-serializable dict (for caching)."""
        g = self._store.to_networkx()
        return {
            "nodes": [g.nodes[n] for n in g.nodes],
            "edges": [
                {**g.edges[u, v], "source": u, "target": v}
                for u, v in g.edges
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return detailed graph statistics."""
        return self._store.get_statistics()
