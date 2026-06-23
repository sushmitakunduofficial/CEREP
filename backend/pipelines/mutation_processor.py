"""
Mutation Processor — parses MAF/CSV mutation files and normalizes gene list.
Supports TCGA MAF format and simple CSV gene + mutation type files.
Pure Python stdlib implementation (no pandas dependency).
"""
import io
import csv
from collections import defaultdict
from typing import List, Dict, Any

from backend.core.logging import get_logger

logger = get_logger("pipelines.mutation")

MAF_GENE_COL = "Hugo_Symbol"
MAF_VARIANT_COL = "Variant_Classification"
PATHOGENIC_VARIANTS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "Translation_Start_Site",
    "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
}


class MutationProcessor:
    """Ingests mutation data and returns structured gene mutation summaries."""

    def process_maf(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Parse a TCGA MAF file (tab-delimited) and extract pathogenic mutations.
        Returns a list of {gene, mutation_type, count} records.
        """
        text = file_bytes.decode("utf-8", errors="ignore")
        # Skip comment lines starting with #
        lines = [l for l in text.splitlines() if not l.startswith("#")]
        if not lines:
            return []

        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
        counts: Dict[tuple, int] = defaultdict(int)
        has_maf_cols = False

        for row in reader:
            gene = row.get(MAF_GENE_COL, "").strip()
            variant = row.get(MAF_VARIANT_COL, "").strip()
            if not gene:
                continue
            has_maf_cols = True
            if variant in PATHOGENIC_VARIANTS:
                counts[(gene, variant)] += 1

        if not has_maf_cols:
            return self.process_simple_csv(file_bytes)

        result = [
            {"gene": gene, "mutation_type": variant, "count": count}
            for (gene, variant), count in sorted(counts.items(), key=lambda x: -x[1])
        ]
        logger.info(f"MAF processed: {len(result)} mutation records")
        return result

    def process_simple_csv(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Parse a simple CSV with at minimum a 'gene' column.
        Optional columns: mutation_type, count.
        """
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        records = []
        for row in reader:
            gene = row.get("gene") or row.get("Gene") or row.get("symbol")
            if not gene:
                continue
            records.append({
                "gene": gene.strip(),
                "mutation_type": row.get("mutation_type", "Unknown"),
                "count": int(row.get("count", 1)),
            })
        return records

    def get_gene_list(self, records: List[Dict]) -> List[str]:
        """Extract unique gene list from processed mutation records."""
        seen: set = set()
        genes = []
        for r in records:
            g = r["gene"]
            if g not in seen:
                seen.add(g)
                genes.append(g)
        return genes

    def process_text_input(self, gene_text: str) -> List[str]:
        """Parse a comma / newline / space-separated string of gene names."""
        import re
        cleaned = re.split(r"[,\n\r\s;]+", gene_text.strip())
        return [g.strip() for g in cleaned if g.strip()]
