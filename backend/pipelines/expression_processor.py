"""
Expression Processor — normalizes RNA expression matrices and identifies
differentially expressed genes for downstream KG mapping.
Pure Python stdlib implementation (no pandas/numpy/scipy dependency).
"""
import io
import csv
import math
from typing import List, Dict, Any

from backend.core.logging import get_logger

logger = get_logger("pipelines.expression")


class ExpressionProcessor:
    """Processes gene expression data → normalized DE gene list."""

    def process(self, file_bytes: bytes, top_n: int = 50) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Load CSV (genes as rows, samples as columns)
        2. Log2 normalize each value
        3. Compute mean absolute z-score per gene
        4. Return top-N most differentially expressed genes
        """
        rows, gene_col = self._load(file_bytes)
        if not rows:
            return {"total_genes": 0, "samples": 0, "top_de_genes": []}

        scored = []
        sample_count = 0
        for row in rows:
            gene = row.get(gene_col, "").strip()
            if not gene:
                continue
            values = []
            for k, v in row.items():
                if k == gene_col:
                    continue
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
            if not values:
                continue
            sample_count = max(sample_count, len(values))
            log_vals = [math.log2(v + 1) for v in values]
            z_score = self._mean_abs_zscore(log_vals)
            scored.append((gene, z_score))

        scored.sort(key=lambda x: -x[1])
        top_genes = [
            {"gene": g, "z_score": round(z, 4)}
            for g, z in scored[:top_n]
        ]
        logger.info(f"Expression processed: {len(scored)} genes, top {len(top_genes)} returned")
        return {
            "total_genes": len(scored),
            "samples": sample_count,
            "top_de_genes": top_genes,
        }

    def _load(self, file_bytes: bytes):
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return [], ""
        # Detect gene column: first column or one named gene/symbol
        fieldnames = reader.fieldnames or []
        gene_col = fieldnames[0] if fieldnames else ""
        for candidate in ["gene", "Gene", "symbol", "gene_symbol", "Hugo_Symbol"]:
            if candidate in fieldnames:
                gene_col = candidate
                break
        return rows, gene_col

    @staticmethod
    def _mean_abs_zscore(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        z_scores = [abs((v - mean) / std) for v in values]
        return sum(z_scores) / n

    def get_gene_list(self, result: Dict) -> List[str]:
        return [r["gene"] for r in result["top_de_genes"]]
