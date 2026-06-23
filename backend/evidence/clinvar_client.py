"""
ClinVar Client — async wrapper around NCBI E-utilities for ClinVar variant data.

Retrieves variant clinical-significance annotations (pathogenic / benign / VUS)
for genes of interest and maps them to structured ``ClinVarVariant`` objects.
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from backend.core.logging import get_logger

logger = get_logger("evidence.clinvar")

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DEFAULT_TIMEOUT = 10.0
_MAX_RPS_NO_KEY = 3


@dataclass
class ClinVarVariant:
    """Single ClinVar variant record."""

    variant_id: str
    name: str = ""
    clinical_significance: str = ""  # Pathogenic | Benign | Uncertain significance …
    review_status: str = ""
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "clinical_significance": self.clinical_significance,
            "review_status": self.review_status,
            "conditions": self.conditions,
        }


class ClinVarClient:
    """Async client for ClinVar via NCBI E-utilities.

    Parameters
    ----------
    api_key:
        Optional NCBI API key (raises rate limit from 3 → 10 req/s).
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        max_concurrent = 10 if api_key else _MAX_RPS_NO_KEY
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_variants(
        self,
        gene_name: str,
        *,
        max_results: int = 20,
    ) -> List[ClinVarVariant]:
        """Return ClinVar variants for *gene_name*.

        Uses the E-utilities esearch → esummary pipeline.
        Returns an empty list on any failure.
        """
        try:
            ids = await self._search_ids(gene_name, max_results=max_results)
            if not ids:
                return []
            return await self._fetch_summaries(ids)
        except Exception as exc:
            logger.error(
                "ClinVar query failed",
                extra={"extra": {"gene": gene_name, "error": str(exc)}},
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search_ids(self, gene_name: str, *, max_results: int) -> List[str]:
        """Search ClinVar for variant UIDs related to a gene."""
        params: Dict[str, Any] = {
            "db": "clinvar",
            "term": f"{gene_name}[gene] AND clinical_significance_pathogenic_or_likely_pathogenic[filter]",
            "retmax": max_results,
            "retmode": "json",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        data = await self._get_json(f"{_EUTILS_BASE}/esearch.fcgi", params)
        result = data.get("esearchresult", {})
        ids: List[str] = result.get("idlist", [])
        logger.info(
            "ClinVar search completed",
            extra={"extra": {
                "gene": gene_name,
                "total_hits": int(result.get("count", 0)),
                "returned": len(ids),
            }},
        )
        return ids

    async def _fetch_summaries(self, uids: List[str]) -> List[ClinVarVariant]:
        """Fetch document summaries (esummary) for a list of ClinVar UIDs."""
        params: Dict[str, Any] = {
            "db": "clinvar",
            "id": ",".join(uids),
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        raw = await self._get_raw(f"{_EUTILS_BASE}/esummary.fcgi", params)
        return self._parse_esummary_xml(raw)

    # -- HTTP ----------------------------------------------------------

    async def _get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

    async def _get_raw(self, url: str, params: Dict[str, Any]) -> str:
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.text

    # -- XML parsing ---------------------------------------------------

    @staticmethod
    def _parse_esummary_xml(xml_text: str) -> List[ClinVarVariant]:
        """Parse the ClinVar esummary XML into ``ClinVarVariant`` objects."""
        variants: List[ClinVarVariant] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.warning("ClinVar esummary XML parse error")
            return variants

        for doc_sum in root.findall(".//DocumentSummary"):
            uid = doc_sum.get("uid", "")
            variant = ClinVarVariant(variant_id=uid)

            # Variant title / name
            title_el = doc_sum.find("title")
            if title_el is not None and title_el.text:
                variant.name = title_el.text

            # Clinical significance
            clin_sig_el = doc_sum.find("clinical_significance/description")
            if clin_sig_el is not None and clin_sig_el.text:
                variant.clinical_significance = clin_sig_el.text

            # Review status
            review_el = doc_sum.find("clinical_significance/review_status")
            if review_el is not None and review_el.text:
                variant.review_status = review_el.text

            # Conditions / traits
            for trait_el in doc_sum.findall(".//trait_set/trait/trait_name"):
                if trait_el.text:
                    variant.conditions.append(trait_el.text)

            # Fallback: germline_classification (newer ClinVar XML schema)
            if not variant.clinical_significance:
                gc_el = doc_sum.find("germline_classification/description")
                if gc_el is not None and gc_el.text:
                    variant.clinical_significance = gc_el.text

                gc_review_el = doc_sum.find("germline_classification/review_status")
                if gc_review_el is not None and gc_review_el.text:
                    variant.review_status = gc_review_el.text

            variants.append(variant)

        logger.info("ClinVar parsed %d variants", len(variants))
        return variants
