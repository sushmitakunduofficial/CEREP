"""
Gene Normalizer — resolves gene/protein name variants to canonical KG identifiers.
Handles common naming inconsistencies in real-world omics datasets:
  BRCA-1, BRCA_1, brca1 → BRCA1
  p53, TRP53 → TP53
"""
import re
from typing import Optional, Dict, List
from backend.core.logging import get_logger

logger = get_logger("pipelines.gene_normalizer")

# Static normalization rules (applied before alias lookup)
_NORMALIZATION_PATTERNS: list[tuple] = [
    (r"[-_\s]+", ""),           # Remove hyphens, underscores, spaces
    (r"^p(\d+)$", r"TP\1"),     # p53 → TP53, p21 → TP21 etc
    (r"(?i)^her(\d+)$", r"ERBB\1"),   # HER2 → ERBB2
    (r"(?i)^c-(.+)$", r"\1"),   # c-MYC → MYC
    (r"(?i)^ki-ras$", "KRAS"),
    (r"(?i)^ki-ras(\d)$", r"KRAS\1"),
]


class GeneNormalizer:
    """
    Two-stage normalization:
    1. Apply regex pattern rules (handle punctuation, common aliases)
    2. Lookup in KG entity index (alias → canonical id)
    """

    def __init__(self, entity_index: Optional[Dict[str, str]] = None) -> None:
        """
        Parameters
        ----------
        entity_index : dict
            Mapping of UPPER-CASE alias → canonical KG node id.
            Obtained from CERAPGraphBuilder.get_entity_index().
        """
        self._entity_index: Dict[str, str] = entity_index or {}

    def update_index(self, entity_index: Dict[str, str]) -> None:
        self._entity_index = entity_index

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(self, raw_name: str) -> str:
        """
        Normalize a raw gene name string.
        Returns the canonical KG id if found, otherwise the cleaned symbol.
        """
        cleaned = self._clean(raw_name)
        canonical = self._entity_index.get(cleaned.upper())
        if canonical:
            return canonical
        # Try without trailing digits (e.g., BRCA vs BRCA1)
        stripped = re.sub(r"\d+$", "", cleaned)
        canonical = self._entity_index.get(stripped.upper())
        return canonical if canonical else cleaned

    def normalize_list(self, names: List[str]) -> List[str]:
        """Normalize a list of gene names, deduplicating while preserving order."""
        seen: set = set()
        result: List[str] = []
        for name in names:
            norm = self.normalize(name)
            if norm not in seen:
                seen.add(norm)
                result.append(norm)
        return result

    def normalize_with_report(self, names: List[str]) -> Dict[str, str]:
        """
        Normalize a list and return a mapping of raw → canonical.
        Useful for audit logging.
        """
        return {name: self.normalize(name) for name in names}

    def is_in_graph(self, raw_name: str) -> bool:
        return self.normalize(raw_name) in self._entity_index.values()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _clean(self, raw: str) -> str:
        """Apply sequential normalization patterns to raw gene name."""
        cleaned = raw.strip()
        for pattern, replacement in _NORMALIZATION_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.upper()
