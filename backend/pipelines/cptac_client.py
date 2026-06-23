"""
CPTAC Data Access Client — retrieves proteomics and phosphoproteomics
data from the Clinical Proteomic Tumor Analysis Consortium.

Provides sample querying, proteomics/phosphoproteomics download,
and CPTAC-to-TCGA case ID mapping.  Handles both MaxQuant
intensity-based and TMT reporter-ion quantification formats.

Pure async implementation using httpx, with structured JSON logging.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.core.logging import get_logger

logger = get_logger("pipelines.cptac_client")

# ── CPTAC API constants ──────────────────────────────────────────────────────

# CPTAC data portal base URL (PDC API)
PDC_BASE_URL = "https://pdc.cancer.gov/graphql"

# Supported CPTAC studies and their PDC study identifiers
CPTAC_STUDIES: Dict[str, Dict[str, str]] = {
    "brca": {
        "pdc_study_id": "PDC000120",
        "study_name": "CPTAC BRCA Discovery Study",
        "cancer_type": "Breast Invasive Carcinoma",
    },
    "ccrcc": {
        "pdc_study_id": "PDC000127",
        "study_name": "CPTAC CCRCC Discovery Study",
        "cancer_type": "Clear Cell Renal Cell Carcinoma",
    },
    "luad": {
        "pdc_study_id": "PDC000153",
        "study_name": "CPTAC LUAD Discovery Study",
        "cancer_type": "Lung Adenocarcinoma",
    },
    "ucec": {
        "pdc_study_id": "PDC000125",
        "study_name": "CPTAC UCEC Discovery Study",
        "cancer_type": "Uterine Corpus Endometrial Carcinoma",
    },
    "gbm": {
        "pdc_study_id": "PDC000204",
        "study_name": "CPTAC GBM Discovery Study",
        "cancer_type": "Glioblastoma Multiforme",
    },
}

# Default cache directory
_DEFAULT_CACHE_DIR = Path("data/raw/cptac_cache")

# Request timeout
_REQUEST_TIMEOUT = 90.0


class CPTACClient:
    """Async client for CPTAC proteomics data via the PDC GraphQL API.

    Provides methods to query samples, download global proteomics and
    phosphoproteomics quantification, and map CPTAC identifiers to
    TCGA case IDs.

    Parameters
    ----------
    cache_dir : Path | str | None
        Directory for caching downloaded data.  Defaults to
        ``data/raw/cptac_cache``.
    timeout : float
        HTTP timeout in seconds for all PDC requests.
    """

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout

        # CPTAC sample-ID → TCGA case-ID mapping (lazily populated)
        self._id_map: Dict[str, str] = {}

        logger.info(
            "CPTACClient initialised",
            extra={"cache_dir": str(self._cache_dir), "timeout": timeout},
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def query_samples(
        self,
        study: str = "brca",
        max_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query available samples for a CPTAC study.

        Parameters
        ----------
        study : str
            Short study name (e.g. ``brca``, ``luad``).  Must be a key
            in ``CPTAC_STUDIES``.
        max_samples : int
            Maximum number of sample records to return.

        Returns
        -------
        list[dict]
            Each dict has ``sample_id``, ``case_id`` (TCGA-mapped if
            available), ``sample_type``, ``study``, and optional
            demographic metadata.
        """
        study_lower = study.lower()
        if study_lower not in CPTAC_STUDIES:
            logger.error(
                "Unknown CPTAC study",
                extra={"study": study, "available": list(CPTAC_STUDIES.keys())},
            )
            raise ValueError(
                f"Unknown CPTAC study '{study}'.  "
                f"Available: {list(CPTAC_STUDIES.keys())}"
            )

        study_info = CPTAC_STUDIES[study_lower]
        pdc_id = study_info["pdc_study_id"]

        cache_key = self._cache_key("samples", {"pdc_id": pdc_id, "max": max_samples})
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info(
                "Samples loaded from cache",
                extra={"study": study_lower, "count": len(cached)},
            )
            return cached

        logger.info(
            "Querying CPTAC/PDC samples",
            extra={"study": study_lower, "pdc_study_id": pdc_id},
        )

        query = """
        query SamplesPerStudy($pdc_study_id: String!, $offset: Int!, $limit: Int!) {
            paginatedCasesPerStudy(
                pdc_study_id: $pdc_study_id
                offset: $offset
                limit: $limit
            ) {
                total
                casesPerStudy {
                    case_id
                    case_submitter_id
                    disease_type
                    primary_site
                    demographics {
                        gender
                        race
                        ethnicity
                        vital_status
                    }
                    samples {
                        sample_id
                        sample_submitter_id
                        sample_type
                    }
                }
            }
        }
        """

        variables = {
            "pdc_study_id": pdc_id,
            "offset": 0,
            "limit": max_samples,
        }

        data = await self._graphql(query, variables)
        paginated = (
            data.get("data", {})
            .get("paginatedCasesPerStudy", {})
        )
        cases_raw = paginated.get("casesPerStudy", []) or []

        results: List[Dict[str, Any]] = []
        for case in cases_raw:
            case_submitter_id = case.get("case_submitter_id", "")
            demo = (case.get("demographics") or [{}])
            demo_first = demo[0] if demo else {}

            for sample in case.get("samples", []) or []:
                sample_record: Dict[str, Any] = {
                    "sample_id": sample.get("sample_id", ""),
                    "sample_submitter_id": sample.get("sample_submitter_id", ""),
                    "sample_type": sample.get("sample_type", ""),
                    "case_submitter_id": case_submitter_id,
                    "tcga_case_id": self._map_to_tcga(case_submitter_id),
                    "disease_type": case.get("disease_type", ""),
                    "primary_site": case.get("primary_site", ""),
                    "study": study_lower,
                    "pdc_study_id": pdc_id,
                    "gender": demo_first.get("gender"),
                    "race": demo_first.get("race"),
                    "vital_status": demo_first.get("vital_status"),
                }
                results.append(sample_record)

                # Populate ID mapping
                if case_submitter_id:
                    self._id_map[sample.get("sample_id", "")] = case_submitter_id

                if len(results) >= max_samples:
                    break
            if len(results) >= max_samples:
                break

        self._write_cache(cache_key, results)
        logger.info(
            "CPTAC samples retrieved",
            extra={"study": study_lower, "count": len(results)},
        )
        return results

    async def download_proteomics(
        self,
        sample_id: str,
        study: str = "brca",
    ) -> Dict[str, Any]:
        """Download global proteomics quantification for a sample.

        Retrieves protein-level abundance data, handling both
        MaxQuant (LFQ intensity) and TMT (reporter-ion ratio) formats.

        Parameters
        ----------
        sample_id : str
            CPTAC/PDC sample UUID.
        study : str
            Short study name for context.

        Returns
        -------
        dict
            ``{"sample_id": ..., "proteins": {GENE_SYMBOL: abundance, ...},
            "quantification_method": ..., "source": "cptac"}``.
        """
        cache_key = self._cache_key("proteomics", {"sample_id": sample_id})
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info(
                "Proteomics loaded from cache",
                extra={"sample_id": sample_id},
            )
            return cached

        logger.info(
            "Downloading proteomics",
            extra={"sample_id": sample_id, "study": study},
        )

        query = """
        query ProteomicsData($sample_id: String!) {
            quantDataMatrix(
                data_type: "proteome"
                sample_id: $sample_id
            ) {
                gene_symbol
                log2_ratio
                unshared_log2_ratio
                precursor_area
            }
        }
        """

        data = await self._graphql(query, {"sample_id": sample_id})
        quant_rows = data.get("data", {}).get("quantDataMatrix", []) or []

        proteins: Dict[str, float] = {}
        quant_method = "unknown"

        for row in quant_rows:
            gene = (row.get("gene_symbol") or "").strip()
            if not gene:
                continue

            # Prefer log2 ratio (TMT), fall back to precursor area (MaxQuant)
            value, method = self._extract_abundance(row)
            if value is not None:
                proteins[gene] = value
                quant_method = method

        result: Dict[str, Any] = {
            "sample_id": sample_id,
            "tcga_case_id": self._id_map.get(sample_id),
            "study": study,
            "source": "cptac",
            "quantification_method": quant_method,
            "protein_count": len(proteins),
            "proteins": proteins,
        }
        self._write_cache(cache_key, result)
        logger.info(
            "Proteomics downloaded",
            extra={
                "sample_id": sample_id,
                "protein_count": len(proteins),
                "method": quant_method,
            },
        )
        return result

    async def download_phosphoproteomics(
        self,
        sample_id: str,
        study: str = "brca",
    ) -> Dict[str, Any]:
        """Download phosphoproteomics (site-level) data for a sample.

        Retrieves phosphorylation-site quantification, typically from
        TMT-labelled enrichment experiments.

        Parameters
        ----------
        sample_id : str
            CPTAC/PDC sample UUID.
        study : str
            Short study name for context.

        Returns
        -------
        dict
            ``{"sample_id": ..., "phosphosites": {GENE_SITE: abundance, ...},
            "quantification_method": ..., "source": "cptac"}``.
        """
        cache_key = self._cache_key("phospho", {"sample_id": sample_id})
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info(
                "Phosphoproteomics loaded from cache",
                extra={"sample_id": sample_id},
            )
            return cached

        logger.info(
            "Downloading phosphoproteomics",
            extra={"sample_id": sample_id, "study": study},
        )

        query = """
        query PhosphoData($sample_id: String!) {
            quantDataMatrix(
                data_type: "phosphoproteome"
                sample_id: $sample_id
            ) {
                gene_symbol
                phosphosite
                log2_ratio
                unshared_log2_ratio
                precursor_area
            }
        }
        """

        data = await self._graphql(query, {"sample_id": sample_id})
        quant_rows = data.get("data", {}).get("quantDataMatrix", []) or []

        phosphosites: Dict[str, float] = {}
        quant_method = "unknown"

        for row in quant_rows:
            gene = (row.get("gene_symbol") or "").strip()
            site = (row.get("phosphosite") or "").strip()
            if not gene:
                continue

            # Composite key: GENE_pSITE (e.g., AKT1_pS473)
            site_key = f"{gene}_{site}" if site else gene
            value, method = self._extract_abundance(row)
            if value is not None:
                phosphosites[site_key] = value
                quant_method = method

        result: Dict[str, Any] = {
            "sample_id": sample_id,
            "tcga_case_id": self._id_map.get(sample_id),
            "study": study,
            "source": "cptac",
            "quantification_method": quant_method,
            "phosphosite_count": len(phosphosites),
            "phosphosites": phosphosites,
        }
        self._write_cache(cache_key, result)
        logger.info(
            "Phosphoproteomics downloaded",
            extra={
                "sample_id": sample_id,
                "phosphosite_count": len(phosphosites),
                "method": quant_method,
            },
        )
        return result

    def get_tcga_case_id(self, sample_id: str) -> Optional[str]:
        """Look up the TCGA case ID for a CPTAC sample.

        Parameters
        ----------
        sample_id : str
            CPTAC sample identifier.

        Returns
        -------
        str | None
            Corresponding TCGA submitter ID (e.g. ``TCGA-A2-A0T2``),
            or ``None`` if mapping is unavailable.
        """
        return self._id_map.get(sample_id)

    def load_id_mapping(self, mapping: Dict[str, str]) -> None:
        """Manually load a CPTAC→TCGA ID mapping.

        Parameters
        ----------
        mapping : dict
            ``{cptac_sample_id: tcga_case_id, ...}``
        """
        self._id_map.update(mapping)
        logger.info(
            "ID mapping loaded",
            extra={"entries": len(mapping), "total": len(self._id_map)},
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _map_to_tcga(self, submitter_id: str) -> Optional[str]:
        """Attempt to extract a TCGA-format case ID from a submitter ID.

        CPTAC BRCA samples often use TCGA submitter IDs directly
        (e.g. ``TCGA-A2-A0T2``).  This method validates the format
        and returns the ID if it matches.
        """
        if not submitter_id:
            return None
        # Standard TCGA format: TCGA-XX-XXXX (project-TSS-participant)
        if re.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", submitter_id):
            return submitter_id
        # Check if it is stored in the mapping
        return self._id_map.get(submitter_id)

    @staticmethod
    def _extract_abundance(
        row: Dict[str, Any],
    ) -> Tuple[Optional[float], str]:
        """Extract quantitative abundance from a PDC data row.

        Prefers ``log2_ratio`` (TMT-based) over ``unshared_log2_ratio``
        over ``precursor_area`` (MaxQuant LFQ).

        Returns
        -------
        tuple[float | None, str]
            ``(abundance_value, quantification_method)``
        """
        # TMT log2 ratio (most common in CPTAC)
        for field, method in [
            ("log2_ratio", "TMT_log2_ratio"),
            ("unshared_log2_ratio", "TMT_unshared_log2_ratio"),
            ("precursor_area", "MaxQuant_LFQ"),
        ]:
            raw = row.get(field)
            if raw is not None and raw != "":
                try:
                    return float(raw), method
                except (ValueError, TypeError):
                    continue
        return None, "unknown"

    async def _graphql(
        self,
        query: str,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a GraphQL query against the PDC API.

        Parameters
        ----------
        query : str
            GraphQL query string.
        variables : dict
            Query variables.

        Returns
        -------
        dict
            Parsed JSON response.
        """
        payload = {"query": query, "variables": variables}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    PDC_BASE_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()

                # Check for GraphQL-level errors
                if "errors" in result:
                    logger.warning(
                        "PDC GraphQL errors",
                        extra={"errors": result["errors"][:3]},
                    )

                return result

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "PDC API HTTP error",
                    extra={
                        "status": exc.response.status_code,
                        "body": exc.response.text[:500],
                    },
                )
                raise
            except httpx.RequestError as exc:
                logger.error(
                    "PDC API request error",
                    extra={"error": str(exc)},
                )
                raise

    # ── Caching helpers ───────────────────────────────────────────────────

    def _cache_key(self, namespace: str, params: Any) -> str:
        """Generate a deterministic cache key."""
        raw = json.dumps({"ns": namespace, "p": params}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> Optional[Any]:
        """Read cached JSON data, or ``None`` if unavailable."""
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Cache read failed",
                extra={"key": key, "error": str(exc)},
            )
            return None

    def _write_cache(self, key: str, data: Any) -> None:
        """Persist data as JSON to the cache directory."""
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps(data, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Cache write failed",
                extra={"key": key, "error": str(exc)},
            )

    def clear_cache(self) -> int:
        """Remove all cached files.  Returns count of files removed."""
        removed = 0
        for path in self._cache_dir.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        logger.info("CPTAC cache cleared", extra={"files_removed": removed})
        return removed
