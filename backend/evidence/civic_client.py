"""
CIViC Client — async wrapper around the CIViC GraphQL API.

Retrieves clinical-evidence items (drugs, diseases, evidence levels)
for a given gene and returns them as structured ``CIViCEvidenceItem`` objects.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from backend.core.logging import get_logger

logger = get_logger("evidence.civic")

_CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
_DEFAULT_TIMEOUT = 10.0

# ── GraphQL query ────────────────────────────────────────────────────
_GENE_EVIDENCE_QUERY = """
query GeneEvidence($geneName: String!) {
  genes(name: $geneName) {
    nodes {
      id
      name
      variants(first: 50) {
        nodes {
          id
          name
          evidenceItems(first: 50) {
            nodes {
              id
              status
              evidenceType
              evidenceLevel
              evidenceDirection
              significance
              disease {
                name
              }
              therapies {
                name
              }
              source {
                citationId
                sourceType
              }
            }
          }
        }
      }
    }
  }
}
"""


@dataclass
class CIViCEvidenceItem:
    """Single evidence item from CIViC."""

    id: int
    gene: str
    variant: str = ""
    disease: str = ""
    drugs: List[str] = field(default_factory=list)
    evidence_type: str = ""       # Predictive | Diagnostic | Prognostic | …
    evidence_level: str = ""      # A (validated) → E (case study)
    evidence_direction: str = ""  # Supports | Does Not Support
    significance: str = ""        # Sensitivity/Response | Resistance | …
    pmids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "gene": self.gene,
            "variant": self.variant,
            "disease": self.disease,
            "drugs": self.drugs,
            "evidence_type": self.evidence_type,
            "evidence_level": self.evidence_level,
            "evidence_direction": self.evidence_direction,
            "significance": self.significance,
            "pmids": self.pmids,
        }


class CIViCClient:
    """Async client for the CIViC GraphQL API.

    Parameters
    ----------
    timeout:
        HTTP timeout in seconds (default 10).
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_evidence(self, gene_name: str) -> List[CIViCEvidenceItem]:
        """Retrieve all accepted evidence items for *gene_name*.

        Returns an empty list on any failure so callers never crash.
        """
        try:
            raw = await self._query(gene_name)
            items = self._parse_response(raw, gene_name)
            logger.info(
                "CIViC evidence retrieved",
                extra={"extra": {"gene": gene_name, "items": len(items)}},
            )
            return items
        except Exception as exc:
            logger.error(
                "CIViC query failed",
                extra={"extra": {"gene": gene_name, "error": str(exc)}},
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _query(self, gene_name: str) -> Dict[str, Any]:
        """Execute the GraphQL query against CIViC."""
        payload = {
            "query": _GENE_EVIDENCE_QUERY,
            "variables": {"geneName": gene_name},
        }
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    _CIVIC_GRAPHQL_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

    @staticmethod
    def _parse_response(
        data: Dict[str, Any],
        gene_name: str,
    ) -> List[CIViCEvidenceItem]:
        """Flatten nested GraphQL response into a list of evidence items."""
        items: List[CIViCEvidenceItem] = []

        genes = (
            data.get("data", {})
            .get("genes", {})
            .get("nodes", [])
        )
        for gene_node in genes:
            g_name = gene_node.get("name", gene_name)
            variants = gene_node.get("variants", {}).get("nodes", [])

            for var_node in variants:
                var_name = var_node.get("name", "")
                evidence_nodes = (
                    var_node.get("evidenceItems", {}).get("nodes", [])
                )

                for ev in evidence_nodes:
                    # Only include accepted evidence
                    if ev.get("status", "").lower() not in ("accepted", "submitted"):
                        continue

                    # Extract PMIDs from source
                    pmids: List[str] = []
                    source = ev.get("source")
                    if source:
                        cit_id = source.get("citationId")
                        source_type = source.get("sourceType", "")
                        if cit_id and source_type.upper() == "PUBMED":
                            pmids.append(str(cit_id))

                    # Extract drugs / therapies
                    drugs: List[str] = []
                    therapies = ev.get("therapies") or []
                    if isinstance(therapies, list):
                        drugs = [t["name"] for t in therapies if t.get("name")]
                    elif isinstance(therapies, dict):
                        # Handle potential nested structure
                        for t in therapies.get("nodes", therapies.get("edges", [])):
                            name = t.get("name") or (t.get("node", {}) or {}).get("name")
                            if name:
                                drugs.append(name)

                    disease_name = ""
                    disease_obj = ev.get("disease")
                    if disease_obj and disease_obj.get("name"):
                        disease_name = disease_obj["name"]

                    items.append(CIViCEvidenceItem(
                        id=int(ev.get("id", 0)),
                        gene=g_name,
                        variant=var_name,
                        disease=disease_name,
                        drugs=drugs,
                        evidence_type=ev.get("evidenceType", ""),
                        evidence_level=ev.get("evidenceLevel", ""),
                        evidence_direction=ev.get("evidenceDirection", ""),
                        significance=ev.get("significance", ""),
                        pmids=pmids,
                    ))

        return items
