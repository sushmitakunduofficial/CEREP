"""
Open Targets Client — async wrapper around the Open Targets Platform GraphQL API.

Provides disease-association scores and drug-target information for
genes of interest.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from backend.core.logging import get_logger

logger = get_logger("evidence.opentargets")

_OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_DEFAULT_TIMEOUT = 10.0

# ── GraphQL queries ──────────────────────────────────────────────────

_SEARCH_TARGET_QUERY = """
query SearchTarget($geneName: String!) {
  search(queryString: $geneName, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits {
      id
      entity
      name
    }
  }
}
"""

_ASSOCIATIONS_QUERY = """
query Associations($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 25}) {
      rows {
        disease {
          id
          name
        }
        score
        datasourceScores {
          id
          score
        }
      }
    }
  }
}
"""

_DRUGS_QUERY = """
query DrugTargets($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    knownDrugs(size: 25) {
      rows {
        drug {
          id
          name
          drugType
          maximumClinicalTrialPhase
          hasBeenWithdrawn
        }
        disease {
          name
        }
        phase
        status
        urls {
          url
          name
        }
      }
    }
  }
}
"""


@dataclass
class DiseaseAssociation:
    """A gene–disease association from Open Targets."""

    disease_id: str
    disease_name: str
    association_score: float
    evidence_count: int = 0
    datasource_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "association_score": self.association_score,
            "evidence_count": self.evidence_count,
            "datasource_scores": self.datasource_scores,
        }


@dataclass
class DrugTarget:
    """An approved or clinical-phase drug targeting a gene."""

    drug_id: str
    drug_name: str
    drug_type: str = ""
    disease_name: str = ""
    clinical_phase: int = 0
    status: str = ""
    is_withdrawn: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "drug_type": self.drug_type,
            "disease_name": self.disease_name,
            "clinical_phase": self.clinical_phase,
            "status": self.status,
            "is_withdrawn": self.is_withdrawn,
        }


class OpenTargetsClient:
    """Async client for the Open Targets Platform GraphQL API.

    Parameters
    ----------
    timeout:
        HTTP timeout in seconds (default 10).
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(5)
        # Cache resolved gene symbol → Ensembl ID to avoid repeated lookups.
        self._ensembl_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_associations(
        self,
        gene_name: str,
    ) -> List[DiseaseAssociation]:
        """Return disease associations for *gene_name*, ordered by score.

        Returns an empty list on any failure.
        """
        try:
            ensembl_id = await self._resolve_ensembl_id(gene_name)
            if not ensembl_id:
                logger.warning(
                    "Could not resolve Ensembl ID",
                    extra={"extra": {"gene": gene_name}},
                )
                return []

            data = await self._graphql(_ASSOCIATIONS_QUERY, {"ensemblId": ensembl_id})
            associations = self._parse_associations(data)
            logger.info(
                "Open Targets associations retrieved",
                extra={"extra": {"gene": gene_name, "count": len(associations)}},
            )
            return associations
        except Exception as exc:
            logger.error(
                "Open Targets associations query failed",
                extra={"extra": {"gene": gene_name, "error": str(exc)}},
            )
            return []

    async def get_drug_targets(
        self,
        gene_name: str,
    ) -> List[DrugTarget]:
        """Return approved/clinical drugs targeting *gene_name*.

        Returns an empty list on any failure.
        """
        try:
            ensembl_id = await self._resolve_ensembl_id(gene_name)
            if not ensembl_id:
                logger.warning(
                    "Could not resolve Ensembl ID for drug lookup",
                    extra={"extra": {"gene": gene_name}},
                )
                return []

            data = await self._graphql(_DRUGS_QUERY, {"ensemblId": ensembl_id})
            drugs = self._parse_drugs(data)
            logger.info(
                "Open Targets drugs retrieved",
                extra={"extra": {"gene": gene_name, "count": len(drugs)}},
            )
            return drugs
        except Exception as exc:
            logger.error(
                "Open Targets drug query failed",
                extra={"extra": {"gene": gene_name, "error": str(exc)}},
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_ensembl_id(self, gene_name: str) -> Optional[str]:
        """Map a gene symbol to its Ensembl gene ID via OT search."""
        key = gene_name.strip().upper()
        if key in self._ensembl_cache:
            return self._ensembl_cache[key]

        try:
            data = await self._graphql(_SEARCH_TARGET_QUERY, {"geneName": gene_name})
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            for hit in hits:
                if hit.get("entity") == "target":
                    ensembl_id: str = hit["id"]
                    self._ensembl_cache[key] = ensembl_id
                    return ensembl_id
            self._ensembl_cache[key] = None
            return None
        except Exception:
            self._ensembl_cache[key] = None
            return None

    async def _graphql(
        self,
        query: str,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a GraphQL request against the Open Targets API."""
        payload = {"query": query, "variables": variables}
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    _OT_GRAPHQL_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]

    # -- Parsers -------------------------------------------------------

    @staticmethod
    def _parse_associations(data: Dict[str, Any]) -> List[DiseaseAssociation]:
        """Parse the associations query response."""
        results: List[DiseaseAssociation] = []
        target = data.get("data", {}).get("target")
        if not target:
            return results

        rows = target.get("associatedDiseases", {}).get("rows", [])
        for row in rows:
            disease = row.get("disease", {})
            ds_scores: Dict[str, float] = {}
            for ds in row.get("datasourceScores", []):
                ds_id = ds.get("id", "")
                ds_score = ds.get("score", 0.0)
                if ds_id:
                    ds_scores[ds_id] = ds_score

            results.append(DiseaseAssociation(
                disease_id=disease.get("id", ""),
                disease_name=disease.get("name", ""),
                association_score=float(row.get("score", 0.0)),
                evidence_count=len(ds_scores),
                datasource_scores=ds_scores,
            ))

        # Sort descending by score
        results.sort(key=lambda a: a.association_score, reverse=True)
        return results

    @staticmethod
    def _parse_drugs(data: Dict[str, Any]) -> List[DrugTarget]:
        """Parse the drugs query response."""
        results: List[DrugTarget] = []
        target = data.get("data", {}).get("target")
        if not target:
            return results

        rows = target.get("knownDrugs", {}).get("rows", [])
        seen_drug_ids: set[str] = set()

        for row in rows:
            drug = row.get("drug", {})
            drug_id = drug.get("id", "")

            # Deduplicate by drug ID
            if drug_id in seen_drug_ids:
                continue
            seen_drug_ids.add(drug_id)

            disease_name = ""
            disease_obj = row.get("disease")
            if disease_obj and disease_obj.get("name"):
                disease_name = disease_obj["name"]

            results.append(DrugTarget(
                drug_id=drug_id,
                drug_name=drug.get("name", ""),
                drug_type=drug.get("drugType", ""),
                disease_name=disease_name,
                clinical_phase=int(drug.get("maximumClinicalTrialPhase") or row.get("phase") or 0),
                status=row.get("status", ""),
                is_withdrawn=bool(drug.get("hasBeenWithdrawn", False)),
            ))

        # Sort by clinical phase descending (approved first)
        results.sort(key=lambda d: d.clinical_phase, reverse=True)
        return results
