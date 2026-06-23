"""
PubMed Client — async wrapper around NCBI E-utilities for literature search.

Supports keyword-based gene searches and abstract fetching with built-in
rate limiting (3 req/s without API key) and graceful degradation.
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from backend.core.logging import get_logger

logger = get_logger("evidence.pubmed")

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DEFAULT_TIMEOUT = 10.0
_MAX_RPS_NO_KEY = 3  # NCBI rate limit without an API key


@dataclass
class PubMedArticle:
    """Parsed article metadata returned by ``fetch_abstract``."""

    pmid: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    journal: str = ""
    year: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal": self.journal,
            "year": self.year,
        }


class PubMedClient:
    """Async client for NCBI E-utilities (PubMed).

    Parameters
    ----------
    api_key:
        Optional NCBI API key.  When provided the rate limit increases
        from 3 to 10 requests per second.
    timeout:
        HTTP timeout in seconds (default 10).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        # Semaphore enforces NCBI rate limit per second.
        max_concurrent = 10 if api_key else _MAX_RPS_NO_KEY
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        gene_name: str,
        *,
        max_results: int = 5,
        additional_terms: Optional[str] = None,
    ) -> List[str]:
        """Search PubMed for articles related to *gene_name*.

        Returns a list of up to *max_results* PMIDs (most recent first).
        On any failure an empty list is returned so that downstream
        callers never crash.
        """
        query = f"{gene_name}[Gene] AND cancer"
        if additional_terms:
            query += f" AND {additional_terms}"

        params: Dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "date",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            data = await self._get(f"{_EUTILS_BASE}/esearch.fcgi", params=params)
            result = data.get("esearchresult", {})
            pmids: List[str] = result.get("idlist", [])
            logger.info(
                "PubMed search completed",
                extra={"extra": {
                    "gene": gene_name,
                    "hits": int(result.get("count", 0)),
                    "returned": len(pmids),
                }},
            )
            return pmids
        except Exception as exc:
            logger.error(
                "PubMed search failed",
                extra={"extra": {"gene": gene_name, "error": str(exc)}},
            )
            return []

    async def fetch_abstract(self, pmid: str) -> PubMedArticle:
        """Fetch article metadata and abstract for a single PMID.

        Returns a ``PubMedArticle`` populated with whatever fields the
        E-utilities response contains.  On failure a stub with just the
        PMID is returned.
        """
        params: Dict[str, Any] = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            raw = await self._get_raw(f"{_EUTILS_BASE}/efetch.fcgi", params=params)
            return self._parse_article_xml(pmid, raw)
        except Exception as exc:
            logger.error(
                "PubMed fetch failed",
                extra={"extra": {"pmid": pmid, "error": str(exc)}},
            )
            return PubMedArticle(pmid=pmid)

    async def fetch_abstracts(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch abstracts for multiple PMIDs concurrently."""
        if not pmids:
            return []
        tasks = [self.fetch_abstract(p) for p in pmids]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, url: str, *, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rate-limited async GET returning parsed JSON."""
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

    async def _get_raw(self, url: str, *, params: Dict[str, Any]) -> str:
        """Rate-limited async GET returning raw text (for XML)."""
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.text

    @staticmethod
    def _parse_article_xml(pmid: str, xml_text: str) -> PubMedArticle:
        """Parse PubMed XML efetch response into a ``PubMedArticle``."""
        article = PubMedArticle(pmid=pmid)
        try:
            root = ET.fromstring(xml_text)
            art_el = root.find(".//MedlineCitation/Article")
            if art_el is None:
                return article

            # Title
            title_el = art_el.find("ArticleTitle")
            if title_el is not None and title_el.text:
                article.title = title_el.text

            # Abstract
            abs_el = art_el.find("Abstract/AbstractText")
            if abs_el is not None and abs_el.text:
                article.abstract = abs_el.text

            # Journal
            journal_el = art_el.find("Journal/Title")
            if journal_el is not None and journal_el.text:
                article.journal = journal_el.text

            # Year
            year_el = art_el.find("Journal/JournalIssue/PubDate/Year")
            if year_el is not None and year_el.text:
                article.year = year_el.text

            # Authors
            for author_el in art_el.findall("AuthorList/Author"):
                last = author_el.findtext("LastName", "")
                initials = author_el.findtext("Initials", "")
                if last:
                    article.authors.append(f"{last} {initials}".strip())

        except ET.ParseError:
            logger.warning("XML parse error for PMID %s", pmid)

        return article
