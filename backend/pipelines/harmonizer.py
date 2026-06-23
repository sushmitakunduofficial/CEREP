"""
Multi-Omics Harmonizer — aligns genomic, transcriptomic, and proteomic
data modalities into a unified molecular profile matrix.

Core capabilities:
  • Alignment by HUGO gene symbols across modalities
  • Independent z-score normalization per modality
  • NMF-based / correlation-based BRCA subtype classification
  • Output as a unified molecular profile matrix

Pure Python implementation (stdlib only; pandas is optional but
supported for interoperability).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.core.logging import get_logger

logger = get_logger("pipelines.harmonizer")

# ── BRCA subtype reference signatures ─────────────────────────────────────────
# Canonical gene markers for each BRCA intrinsic subtype.
# Positive weights = upregulated; negative weights = downregulated.

SUBTYPE_SIGNATURES: Dict[str, Dict[str, float]] = {
    "Luminal A": {
        "ESR1": 2.5, "PGR": 2.0, "GATA3": 1.8, "FOXA1": 1.6,
        "XBP1": 1.3, "TFF1": 1.2, "AGR2": 1.0, "BCL2": 1.1,
        "ERBB2": -1.5, "MKI67": -1.8, "CCNB1": -1.2,
        "EGFR": -1.0, "KRT5": -1.5, "KRT17": -1.3,
    },
    "Luminal B": {
        "ESR1": 1.8, "PGR": 0.8, "GATA3": 1.2, "FOXA1": 1.0,
        "MKI67": 2.0, "CCNB1": 1.8, "AURKA": 1.5, "BIRC5": 1.3,
        "MYBL2": 1.4, "UBE2C": 1.2,
        "ERBB2": 0.5, "GRB7": 0.4,
        "KRT5": -1.0, "KRT17": -0.8,
    },
    "HER2+": {
        "ERBB2": 3.0, "GRB7": 2.5, "STARD3": 1.8, "PGAP3": 1.5,
        "ORMDL3": 1.2, "PSMD3": 1.0,
        "ESR1": -1.5, "PGR": -2.0, "GATA3": -0.8,
        "KRT5": -0.5, "KRT17": -0.5,
        "MKI67": 1.0, "CCNB1": 0.8,
    },
    "Basal": {
        "KRT5": 2.5, "KRT17": 2.2, "KRT14": 2.0, "EGFR": 2.0,
        "FOXC1": 1.8, "CDH3": 1.5, "MIA": 1.2, "TRIM29": 1.0,
        "ESR1": -2.5, "PGR": -2.5, "GATA3": -2.0, "FOXA1": -1.8,
        "ERBB2": -1.0,
        "MKI67": 1.5, "CCNB1": 1.3,
    },
    "TNBC": {
        "KRT5": 2.0, "KRT17": 1.8, "EGFR": 2.2, "FOXC1": 1.5,
        "CDH3": 1.3, "CD274": 1.0,  # PD-L1
        "ESR1": -3.0, "PGR": -3.0, "ERBB2": -2.5,
        "GATA3": -2.0, "FOXA1": -1.8,
        "MKI67": 1.8, "CCNE1": 1.5, "MYC": 1.2,
    },
}

# Complete set of signature genes across all subtypes
ALL_SIGNATURE_GENES: Set[str] = set()
for _sig in SUBTYPE_SIGNATURES.values():
    ALL_SIGNATURE_GENES.update(_sig.keys())


class OmicsHarmonizer:
    """Harmonises multi-omics data layers into a unified profile matrix.

    Workflow:
    1. Align all modalities by HUGO gene symbol
    2. Independently z-score normalise each modality
    3. Merge into a single profile matrix
    4. (Optional) Classify BRCA intrinsic subtype

    The harmonizer works with plain dicts — no pandas required.
    """

    def __init__(self, signature_genes: Optional[Set[str]] = None) -> None:
        """
        Parameters
        ----------
        signature_genes : set[str] | None
            If provided, restrict alignment to these genes only.
            Defaults to the full set of BRCA subtype signature genes.
        """
        self._signature_genes = signature_genes or ALL_SIGNATURE_GENES

    # ── Public API ────────────────────────────────────────────────────────

    def harmonize(
        self,
        genomics: Dict[str, Any],
        transcriptomics: Dict[str, float],
        proteomics: Dict[str, float],
        *,
        restrict_to_signatures: bool = False,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Harmonise three omics modalities into a unified profile matrix.

        Parameters
        ----------
        genomics : dict
            Mutation data — either a list of mutation records
            (``[{"gene": ..., "variant_classification": ...}, ...]``) or
            a dict mapping ``{gene: mutation_count}``.
        transcriptomics : dict
            Gene expression — ``{HUGO_SYMBOL: expression_value, ...}``.
        proteomics : dict
            Protein abundance — ``{HUGO_SYMBOL: abundance_value, ...}``.
        restrict_to_signatures : bool
            If ``True``, include only genes present in the subtype
            signature set.

        Returns
        -------
        dict[str, dict[str, float | None]]
            Unified matrix: ``{gene: {"genomics_z": ...,
            "transcriptomics_z": ..., "proteomics_z": ...,
            "mutation_flag": 0|1}, ...}``.
        """
        logger.info(
            "Starting harmonization",
            extra={
                "genomics_type": type(genomics).__name__,
                "transcriptomics_genes": len(transcriptomics),
                "proteomics_genes": len(proteomics),
            },
        )

        # Step 1: normalise genomics input to {gene: mutation_count}
        mutation_map = self._normalise_genomics(genomics)

        # Step 2: collect all genes present across modalities
        all_genes: Set[str] = set()
        all_genes.update(mutation_map.keys())
        all_genes.update(transcriptomics.keys())
        all_genes.update(proteomics.keys())

        if restrict_to_signatures:
            all_genes &= self._signature_genes

        if not all_genes:
            logger.warning("No overlapping genes found across modalities")
            return {}

        # Step 3: z-score normalise each modality independently
        expr_z = self._zscore(transcriptomics, all_genes)
        prot_z = self._zscore(proteomics, all_genes)

        # Genomics: binary mutation flag + z-scored count
        mut_values = {g: float(mutation_map.get(g, 0)) for g in all_genes}
        mut_z = self._zscore(mut_values, all_genes)

        # Step 4: assemble unified matrix
        matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for gene in sorted(all_genes):
            matrix[gene] = {
                "genomics_z": mut_z.get(gene),
                "transcriptomics_z": expr_z.get(gene),
                "proteomics_z": prot_z.get(gene),
                "mutation_flag": 1 if gene in mutation_map else 0,
                "expression_raw": transcriptomics.get(gene),
                "proteomics_raw": proteomics.get(gene),
            }

        logger.info(
            "Harmonization complete",
            extra={"total_genes": len(matrix)},
        )
        return matrix

    def classify_subtype(
        self,
        harmonized: Dict[str, Dict[str, Optional[float]]],
        method: str = "correlation",
    ) -> Dict[str, Any]:
        """Classify BRCA intrinsic subtype from harmonised profile.

        Parameters
        ----------
        harmonized : dict
            Output from :meth:`harmonize`.
        method : str
            Classification method: ``"correlation"`` (Pearson-based) or
            ``"nmf"`` (non-negative matrix factorisation heuristic).

        Returns
        -------
        dict
            ``{"predicted_subtype": ..., "confidence": ...,
            "scores": {subtype: score, ...}, "method": ...}``.
        """
        if method == "nmf":
            return self._classify_nmf(harmonized)
        return self._classify_correlation(harmonized)

    # ── Correlation-based classification ──────────────────────────────────

    def _classify_correlation(
        self,
        harmonized: Dict[str, Dict[str, Optional[float]]],
    ) -> Dict[str, Any]:
        """Classify subtype via Pearson correlation with reference signatures.

        For each subtype, builds a profile vector from the signature gene
        weights and correlates it against the observed expression
        z-scores.
        """
        scores: Dict[str, float] = {}

        for subtype, signature in SUBTYPE_SIGNATURES.items():
            obs_values: List[float] = []
            ref_values: List[float] = []

            for gene, weight in signature.items():
                entry = harmonized.get(gene)
                if entry is None:
                    continue

                # Use transcriptomics z-score as the primary signal;
                # fall back to proteomics z-score
                observed = entry.get("transcriptomics_z")
                if observed is None:
                    observed = entry.get("proteomics_z")
                if observed is None:
                    continue

                obs_values.append(observed)
                ref_values.append(weight)

            if len(obs_values) < 3:
                scores[subtype] = -999.0
                continue

            corr = self._pearson(obs_values, ref_values)
            scores[subtype] = corr

        # Pick the best
        best_subtype = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_subtype]

        # Confidence: sigmoid-scaled distance from second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = 1.0 / (1.0 + math.exp(-5.0 * gap))  # sigmoid

        result = {
            "predicted_subtype": best_subtype,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "method": "correlation",
            "genes_evaluated": sum(
                1 for g in ALL_SIGNATURE_GENES if g in harmonized
            ),
        }

        logger.info(
            "Subtype classification complete",
            extra={
                "subtype": best_subtype,
                "confidence": result["confidence"],
                "method": "correlation",
            },
        )
        return result

    # ── NMF-based classification ──────────────────────────────────────────

    def _classify_nmf(
        self,
        harmonized: Dict[str, Dict[str, Optional[float]]],
        n_components: int = 5,
        max_iter: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Classify subtype via simplified NMF decomposition.

        Decomposes the expression profile into *n_components* factors
        and correlates each factor's gene loadings with reference
        subtype signatures.

        This is a lightweight pure-Python NMF implementation using
        multiplicative update rules (Lee & Seung, 1999).
        """
        # Build the observation matrix (genes × 1 sample)
        genes_ordered = sorted(
            g for g in ALL_SIGNATURE_GENES if g in harmonized
        )
        if len(genes_ordered) < 3:
            logger.warning("Too few signature genes for NMF classification")
            return {
                "predicted_subtype": "Unknown",
                "confidence": 0.0,
                "scores": {},
                "method": "nmf",
                "genes_evaluated": len(genes_ordered),
            }

        # Expression values (shift to non-negative for NMF)
        values: List[float] = []
        for gene in genes_ordered:
            entry = harmonized[gene]
            v = entry.get("transcriptomics_z") or entry.get("proteomics_z") or 0.0
            values.append(v)

        min_val = min(values)
        if min_val < 0:
            values = [v - min_val + 0.01 for v in values]

        # Run simplified NMF: V ≈ W × H
        n_genes = len(values)
        rng = random.Random(seed)

        # W: n_genes × n_components
        W = [[rng.uniform(0.01, 1.0) for _ in range(n_components)] for _ in range(n_genes)]
        # H: n_components × 1
        H = [[rng.uniform(0.01, 1.0)] for _ in range(n_components)]

        for _iteration in range(max_iter):
            # Update H: H *= (W^T V) / (W^T W H)
            for k in range(n_components):
                numerator = sum(W[i][k] * values[i] for i in range(n_genes))
                denominator = sum(
                    W[i][k] * sum(W[i][j] * H[j][0] for j in range(n_components))
                    for i in range(n_genes)
                )
                if denominator > 1e-12:
                    H[k][0] *= numerator / denominator

            # Update W: W *= (V H^T) / (W H H^T)
            for i in range(n_genes):
                for k in range(n_components):
                    numerator = values[i] * H[k][0]
                    denominator = sum(W[i][j] * H[j][0] for j in range(n_components)) * H[k][0]
                    if denominator > 1e-12:
                        W[i][k] *= numerator / denominator

        # Extract gene loadings per component
        component_profiles: List[Dict[str, float]] = []
        for k in range(n_components):
            profile = {genes_ordered[i]: W[i][k] for i in range(n_genes)}
            component_profiles.append(profile)

        # Correlate each component with subtype signatures
        scores: Dict[str, float] = {}
        for subtype, signature in SUBTYPE_SIGNATURES.items():
            best_corr = -999.0
            for profile in component_profiles:
                obs: List[float] = []
                ref: List[float] = []
                for gene, weight in signature.items():
                    if gene in profile:
                        obs.append(profile[gene])
                        ref.append(weight)
                if len(obs) >= 3:
                    corr = self._pearson(obs, ref)
                    best_corr = max(best_corr, corr)
            scores[subtype] = best_corr

        best_subtype = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_subtype]

        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = 1.0 / (1.0 + math.exp(-5.0 * gap))

        result = {
            "predicted_subtype": best_subtype,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "method": "nmf",
            "n_components": n_components,
            "genes_evaluated": len(genes_ordered),
        }

        logger.info(
            "NMF subtype classification complete",
            extra={
                "subtype": best_subtype,
                "confidence": result["confidence"],
                "method": "nmf",
            },
        )
        return result

    # ── Utility methods ───────────────────────────────────────────────────

    @staticmethod
    def _normalise_genomics(
        genomics: Any,
    ) -> Dict[str, int]:
        """Normalise various genomics input formats to {gene: count}.

        Accepts:
          - list of mutation dicts: ``[{"gene": "TP53", ...}, ...]``
          - dict: ``{"TP53": 3, "PIK3CA": 1}``
          - dict with "mutations" key (GDC client output)
        """
        if isinstance(genomics, dict):
            # GDC client output format
            if "mutations" in genomics:
                genomics = genomics["mutations"]
            else:
                # Already gene→count
                return {k: int(v) for k, v in genomics.items()}

        if isinstance(genomics, list):
            counts: Dict[str, int] = defaultdict(int)
            for entry in genomics:
                gene = entry.get("gene", "").strip()
                if gene:
                    counts[gene] += 1
            return dict(counts)

        logger.warning(
            "Unrecognised genomics format; returning empty",
            extra={"type": type(genomics).__name__},
        )
        return {}

    @staticmethod
    def _zscore(
        values: Dict[str, float],
        gene_set: Set[str],
    ) -> Dict[str, Optional[float]]:
        """Compute z-scores for values within *gene_set*.

        Genes in *gene_set* without a value get ``None``.
        """
        present = {g: values[g] for g in gene_set if g in values}
        n = len(present)
        if n < 2:
            return {g: 0.0 if g in present else None for g in gene_set}

        vals = list(present.values())
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9

        result: Dict[str, Optional[float]] = {}
        for gene in gene_set:
            if gene in present:
                result[gene] = round((present[gene] - mean) / std, 6)
            else:
                result[gene] = None
        return result

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient between two vectors."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denom = math.sqrt(var_x * var_y)
        if denom < 1e-12:
            return 0.0
        return cov / denom

    def get_signature_overlap(
        self,
        gene_list: List[str],
    ) -> Dict[str, Any]:
        """Evaluate how well a gene list covers the subtype signatures.

        Useful for quality control before running classification.

        Returns
        -------
        dict
            ``{"total_signature_genes": ..., "covered": ...,
            "missing": [...], "coverage_pct": ...}``.
        """
        gene_set = set(gene_list)
        covered = gene_set & self._signature_genes
        missing = self._signature_genes - gene_set

        return {
            "total_signature_genes": len(self._signature_genes),
            "covered": len(covered),
            "missing": sorted(missing),
            "coverage_pct": round(
                100.0 * len(covered) / max(len(self._signature_genes), 1), 1
            ),
        }

    @staticmethod
    def matrix_to_table(
        matrix: Dict[str, Dict[str, Optional[float]]],
    ) -> List[Dict[str, Any]]:
        """Convert the harmonised matrix to a flat list of row dicts.

        Convenient for serialisation to JSON or loading into pandas::

            import pandas as pd
            df = pd.DataFrame(harmonizer.matrix_to_table(matrix))
        """
        rows: List[Dict[str, Any]] = []
        for gene, values in matrix.items():
            row: Dict[str, Any] = {"gene": gene}
            row.update(values)
            rows.append(row)
        return rows
