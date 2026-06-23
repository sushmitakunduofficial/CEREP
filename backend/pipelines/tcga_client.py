"""
TCGA / GDC API Client — async interface to the Genomic Data Commons.

Queries the GDC REST API (https://api.gdc.cancer.gov) for TCGA-BRCA
cases and downloads MAF mutation files and STAR-Counts gene expression
data.  Results are cached locally to avoid redundant network calls.

Pure async implementation using httpx, with structured JSON logging.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from backend.core.logging import get_logger

logger = get_logger("pipelines.tcga_client")

# ── GDC API constants ────────────────────────────────────────────────────────

GDC_BASE_URL = "https://api.gdc.cancer.gov"
GDC_CASES_ENDPOINT = f"{GDC_BASE_URL}/cases"
GDC_FILES_ENDPOINT = f"{GDC_BASE_URL}/files"
GDC_DATA_ENDPOINT = f"{GDC_BASE_URL}/data"

# Default cache directory (relative to project root)
_DEFAULT_CACHE_DIR = Path("data/raw/gdc_cache")

# GRCh38 is the current reference genome used by GDC
_REFERENCE_GENOME = "GRCh38"

# Timeout for GDC requests (seconds)
_REQUEST_TIMEOUT = 60.0


class GDCClient:
    """Async client for the NCI Genomic Data Commons REST API.

    Provides methods to query TCGA-BRCA cases, download MAF mutation
    files, and retrieve STAR-Counts gene expression quantification.

    Parameters
    ----------
    cache_dir : Path | str | None
        Directory for caching downloaded files.  Defaults to
        ``data/raw/gdc_cache`` relative to the working directory.
    timeout : float
        HTTP timeout in seconds for all GDC requests.
    """

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        logger.info(
            "GDCClient initialised",
            extra={"cache_dir": str(self._cache_dir), "timeout": timeout},
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def query_cases(
        self,
        project: str = "TCGA-BRCA",
        max_cases: int = 50,
        *,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query GDC for cases belonging to *project*.

        Parameters
        ----------
        project : str
            GDC project id, e.g. ``TCGA-BRCA``.
        max_cases : int
            Maximum number of case records to return.
        fields : list[str] | None
            Specific GDC case fields to retrieve.  ``None`` uses a
            sensible default that includes demographic and diagnosis info.

        Returns
        -------
        list[dict]
            Each dict contains ``case_id``, ``submitter_id``, and any
            requested metadata fields.
        """
        if fields is None:
            fields = [
                "case_id",
                "submitter_id",
                "demographic.gender",
                "demographic.race",
                "demographic.ethnicity",
                "demographic.vital_status",
                "diagnoses.primary_diagnosis",
                "diagnoses.tumor_stage",
                "diagnoses.age_at_diagnosis",
                "project.project_id",
            ]

        payload: Dict[str, Any] = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "=",
                        "content": {
                            "field": "project.project_id",
                            "value": [project],
                        },
                    },
                ],
            },
            "fields": ",".join(fields),
            "size": max_cases,
            "format": "JSON",
        }

        cache_key = self._cache_key("cases", payload)
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info(
                "Cases loaded from cache",
                extra={"project": project, "count": len(cached)},
            )
            return cached

        logger.info(
            "Querying GDC cases",
            extra={"project": project, "max_cases": max_cases},
        )
        data = await self._post_json(GDC_CASES_ENDPOINT, payload)
        hits: List[Dict[str, Any]] = data.get("data", {}).get("hits", [])

        results: List[Dict[str, Any]] = []
        for hit in hits:
            case_record: Dict[str, Any] = {
                "case_id": hit.get("case_id", ""),
                "submitter_id": hit.get("submitter_id", ""),
                "project": project,
            }
            # Flatten demographic
            demo = hit.get("demographic", {}) or {}
            case_record["gender"] = demo.get("gender")
            case_record["race"] = demo.get("race")
            case_record["ethnicity"] = demo.get("ethnicity")
            case_record["vital_status"] = demo.get("vital_status")

            # Flatten first diagnosis
            diagnoses = hit.get("diagnoses", []) or []
            if diagnoses:
                dx = diagnoses[0]
                case_record["primary_diagnosis"] = dx.get("primary_diagnosis")
                case_record["tumor_stage"] = dx.get("tumor_stage")
                case_record["age_at_diagnosis"] = dx.get("age_at_diagnosis")

            results.append(case_record)

        self._write_cache(cache_key, results)
        logger.info(
            "GDC cases retrieved",
            extra={"project": project, "count": len(results)},
        )
        return results

    async def download_maf(self, case_id: str) -> Dict[str, Any]:
        """Download somatic mutation data (MAF) for a single case.

        Queries GDC for open-access MAF files associated with *case_id*,
        filtered to GRCh38-aligned Masked Somatic Mutation files.

        Parameters
        ----------
        case_id : str
            GDC UUID for the case.

        Returns
        -------
        dict
            ``{"case_id": ..., "mutations": [...], "file_id": ..., "source": ...}``
            where each mutation is ``{"gene": ..., "variant_classification": ...,
            "hgvsp_short": ..., "chromosome": ..., "start_position": ...,
            "end_position": ..., "reference_allele": ..., "tumor_allele": ...}``.
        """
        cache_key = self._cache_key("maf", {"case_id": case_id})
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info("MAF loaded from cache", extra={"case_id": case_id})
            return cached

        # Step 1: find the MAF file id for this case
        file_id = await self._find_file(
            case_id,
            data_category="Simple Nucleotide Variation",
            data_type="Masked Somatic Mutation",
            data_format="MAF",
        )
        if not file_id:
            logger.warning(
                "No MAF file found for case", extra={"case_id": case_id}
            )
            return {"case_id": case_id, "mutations": [], "file_id": None, "source": "gdc"}

        # Step 2: download & parse the MAF
        raw_bytes = await self._download_file(file_id)
        mutations = self._parse_maf(raw_bytes)

        result: Dict[str, Any] = {
            "case_id": case_id,
            "file_id": file_id,
            "source": "gdc",
            "reference_genome": _REFERENCE_GENOME,
            "mutation_count": len(mutations),
            "mutations": mutations,
        }
        self._write_cache(cache_key, result)
        logger.info(
            "MAF downloaded and parsed",
            extra={"case_id": case_id, "mutations": len(mutations)},
        )
        return result

    async def download_expression(self, case_id: str) -> Dict[str, Any]:
        """Download gene expression quantification for a single case.

        Retrieves STAR-Counts HTSeq gene expression data aligned to
        GRCh38.

        Parameters
        ----------
        case_id : str
            GDC UUID for the case.

        Returns
        -------
        dict
            ``{"case_id": ..., "genes": {HUGO_SYMBOL: fpkm_value, ...},
            "file_id": ..., "source": ...}``.
        """
        cache_key = self._cache_key("expression", {"case_id": case_id})
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info(
                "Expression loaded from cache", extra={"case_id": case_id}
            )
            return cached

        file_id = await self._find_file(
            case_id,
            data_category="Transcriptome Profiling",
            data_type="Gene Expression Quantification",
            data_format="TSV",
            workflow_type="STAR - Counts",
        )
        if not file_id:
            logger.warning(
                "No expression file found for case",
                extra={"case_id": case_id},
            )
            return {"case_id": case_id, "genes": {}, "file_id": None, "source": "gdc"}

        raw_bytes = await self._download_file(file_id)
        genes = self._parse_expression(raw_bytes)

        result: Dict[str, Any] = {
            "case_id": case_id,
            "file_id": file_id,
            "source": "gdc",
            "reference_genome": _REFERENCE_GENOME,
            "gene_count": len(genes),
            "genes": genes,
        }
        self._write_cache(cache_key, result)
        logger.info(
            "Expression downloaded and parsed",
            extra={"case_id": case_id, "genes": len(genes)},
        )
        return result

    # ── GDC file discovery ────────────────────────────────────────────────

    async def _find_file(
        self,
        case_id: str,
        *,
        data_category: str,
        data_type: str,
        data_format: str,
        workflow_type: Optional[str] = None,
    ) -> Optional[str]:
        """Locate a specific file on GDC for a given case.

        Applies GRCh38 and access-level filters automatically.

        Returns
        -------
        str | None
            GDC file UUID, or ``None`` if no matching file exists.
        """
        filters_content: List[Dict[str, Any]] = [
            {
                "op": "=",
                "content": {"field": "cases.case_id", "value": [case_id]},
            },
            {
                "op": "=",
                "content": {"field": "data_category", "value": [data_category]},
            },
            {
                "op": "=",
                "content": {"field": "data_type", "value": [data_type]},
            },
            {
                "op": "=",
                "content": {"field": "data_format", "value": [data_format]},
            },
            {
                "op": "=",
                "content": {"field": "access", "value": ["open"]},
            },
            {
                "op": "=",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": [workflow_type] if workflow_type else ["*"],
                },
            },
        ]

        # Filter out workflow_type wildcard if not specified
        if not workflow_type:
            filters_content = filters_content[:-1]

        payload: Dict[str, Any] = {
            "filters": {"op": "and", "content": filters_content},
            "fields": "file_id,file_name,file_size,data_format",
            "size": 1,
            "format": "JSON",
        }

        data = await self._post_json(GDC_FILES_ENDPOINT, payload)
        hits = data.get("data", {}).get("hits", [])
        if not hits:
            return None
        return hits[0].get("file_id")

    async def _download_file(self, file_id: str) -> bytes:
        """Download raw file bytes from GDC data endpoint.

        Uses local cache to avoid re-downloading previously fetched files.

        Parameters
        ----------
        file_id : str
            GDC file UUID.

        Returns
        -------
        bytes
            Raw file content.
        """
        cache_path = self._cache_dir / f"file_{file_id}.dat"
        if cache_path.exists():
            logger.info("File loaded from disk cache", extra={"file_id": file_id})
            return cache_path.read_bytes()

        url = f"{GDC_DATA_ENDPOINT}/{file_id}"
        logger.info("Downloading file from GDC", extra={"file_id": file_id, "url": url})

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content

        cache_path.write_bytes(content)
        logger.info(
            "File cached to disk",
            extra={"file_id": file_id, "size_bytes": len(content)},
        )
        return content

    # ── Parsers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_maf(raw: bytes) -> List[Dict[str, Any]]:
        """Parse a MAF file into a list of mutation dicts.

        Handles GDC MAF format: tab-delimited, comment lines start with
        ``#``.  Extracts HUGO gene symbol, variant classification,
        protein change, and genomic coordinates.
        """
        import csv
        import io

        text = raw.decode("utf-8", errors="ignore")
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        if not lines:
            return []

        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
        mutations: List[Dict[str, Any]] = []

        for row in reader:
            gene = (row.get("Hugo_Symbol") or "").strip()
            if not gene or gene == "Unknown":
                continue

            variant_class = (row.get("Variant_Classification") or "").strip()
            hgvsp = (row.get("HGVSp_Short") or row.get("HGVSp") or "").strip()
            chromosome = (row.get("Chromosome") or "").strip()

            # Parse positions safely
            start_pos: Optional[int] = None
            end_pos: Optional[int] = None
            try:
                start_pos = int(row.get("Start_Position", ""))
            except (ValueError, TypeError):
                pass
            try:
                end_pos = int(row.get("End_Position", ""))
            except (ValueError, TypeError):
                pass

            mutations.append({
                "gene": gene,
                "variant_classification": variant_class,
                "hgvsp_short": hgvsp,
                "chromosome": chromosome,
                "start_position": start_pos,
                "end_position": end_pos,
                "reference_allele": (row.get("Reference_Allele") or "").strip(),
                "tumor_allele": (
                    row.get("Tumor_Seq_Allele2")
                    or row.get("Tumor_Seq_Allele1")
                    or ""
                ).strip(),
                "variant_type": (row.get("Variant_Type") or "").strip(),
                "consequence": (row.get("Consequence") or "").strip(),
            })

        return mutations

    @staticmethod
    def _parse_expression(raw: bytes) -> Dict[str, float]:
        """Parse STAR-Counts gene expression TSV into gene→FPKM map.

        The GDC STAR-Counts format has columns:
        ``gene_id | gene_name | gene_type | unstranded | stranded_first |
        stranded_second | tpm_unstranded | fpkm_unstranded | fpkm_uq_unstranded``

        We extract ``gene_name`` → ``fpkm_unstranded`` for protein-coding
        genes, filtering out non-coding / mitochondrial pseudogenes.
        """
        import csv
        import io

        text = raw.decode("utf-8", errors="ignore")
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        if not lines:
            return {}

        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
        genes: Dict[str, float] = {}

        for row in reader:
            gene_name = (row.get("gene_name") or "").strip()
            gene_type = (row.get("gene_type") or "").strip()

            if not gene_name:
                continue

            # Only include protein-coding genes
            if gene_type and gene_type != "protein_coding":
                continue

            # Prefer FPKM; fall back to TPM or raw counts
            fpkm_str = (
                row.get("fpkm_unstranded")
                or row.get("tpm_unstranded")
                or row.get("unstranded")
                or "0"
            ).strip()

            try:
                value = float(fpkm_str)
            except (ValueError, TypeError):
                continue

            # Keep highest value if gene appears multiple times
            if gene_name not in genes or value > genes[gene_name]:
                genes[gene_name] = value

        return genes

    # ── HTTP helpers ──────────────────────────────────────────────────────

    async def _post_json(
        self, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send a POST request with JSON body and return parsed response."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "GDC API HTTP error",
                    extra={
                        "url": url,
                        "status": exc.response.status_code,
                        "body": exc.response.text[:500],
                    },
                )
                raise
            except httpx.RequestError as exc:
                logger.error(
                    "GDC API request error",
                    extra={"url": url, "error": str(exc)},
                )
                raise

    # ── Caching helpers ───────────────────────────────────────────────────

    def _cache_key(self, namespace: str, params: Any) -> str:
        """Generate a deterministic cache key from namespace + params."""
        raw = json.dumps({"ns": namespace, "p": params}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> Optional[Any]:
        """Read a cached JSON result, or return ``None``."""
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Cache read failed; will re-fetch",
                extra={"key": key, "error": str(exc)},
            )
            return None

    def _write_cache(self, key: str, data: Any) -> None:
        """Persist *data* as JSON to the local cache directory."""
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
        for path in self._cache_dir.glob("*.dat"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        logger.info("Cache cleared", extra={"files_removed": removed})
        return removed
