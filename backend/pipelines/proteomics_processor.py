"""
Proteomics Processor — maps protein identifiers to gene symbols
for knowledge-graph integration.
Pure Python stdlib implementation (no pandas dependency).
"""
import io
import csv
import re
from typing import List, Dict, Any

from backend.core.logging import get_logger

logger = get_logger("pipelines.proteomics")

# Common protein→gene mapping for BRCA-relevant proteins
_PROTEIN_TO_GENE: Dict[str, str] = {
    "P04637": "TP53",  "P38398": "BRCA1", "P51587": "BRCA2",
    "P42336": "PIK3CA","P60484": "PTEN",  "P31749": "AKT1",
    "P42345": "MTOR",  "P04626": "ERBB2", "P01106": "MYC",
    "Q15746": "MDM2",  "P16220": "CREB1", "P16671": "CD36",
}

_GENE_COLS = ["Gene names", "gene", "Gene", "gene_symbol", "Gene Symbol", "Majority protein IDs"]


class ProteomicsProcessor:
    """Maps proteomic data to gene-level identifiers for KG integration."""

    def process(self, file_bytes: bytes, top_n: int = 50) -> Dict[str, Any]:
        rows, delimiter = self._load(file_bytes)
        genes = self._extract_genes(rows, top_n=top_n)
        return {
            "total_proteins": len(rows),
            "mapped_genes": genes,
        }

    def _load(self, file_bytes: bytes):
        text = file_bytes.decode("utf-8", errors="ignore")
        # Detect delimiter: tab (MaxQuant) vs comma
        first_line = text.split("\n")[0] if text else ""
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows = []
        for row in reader:
            # Drop contamination rows
            if any(
                row.get(col, "") == "+"
                for col in row
                if "contaminant" in col.lower() or "reverse" in col.lower()
            ):
                continue
            rows.append(row)
        return rows, delimiter

    def _extract_genes(self, rows: List[Dict], top_n: int) -> List[Dict[str, str]]:
        if not rows:
            return []

        # Find the gene column
        fieldnames = list(rows[0].keys()) if rows else []
        gene_col = ""
        for candidate in _GENE_COLS:
            if candidate in fieldnames:
                gene_col = candidate
                break
        if not gene_col:
            for col in fieldnames:
                for cand in _GENE_COLS:
                    if cand.lower() in col.lower():
                        gene_col = col
                        break
                if gene_col:
                    break

        if not gene_col:
            logger.warning("Could not detect gene column; returning empty list")
            return []

        genes = []
        seen: set = set()
        for row in rows[:top_n * 2]:
            raw = row.get(gene_col, "").strip()
            if not raw:
                continue
            for token in re.split(r"[;,|]", raw):
                sym = token.strip().upper()
                if not sym or sym in seen:
                    continue
                if sym in _PROTEIN_TO_GENE:
                    sym = _PROTEIN_TO_GENE[sym]
                seen.add(sym)
                genes.append({"protein": raw, "gene": sym})
                if len(genes) >= top_n:
                    return genes
        return genes

    def get_gene_list(self, result: Dict) -> List[str]:
        return [r["gene"] for r in result["mapped_genes"]]
