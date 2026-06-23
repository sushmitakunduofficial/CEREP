"""
Ablation Study — runs 3 conditions and returns a comparative metric table.

Conditions:
  1. No KG (raw LLM / baseline)
  2. KG paths only (paths injected, no constrained decoding)
  3. Full CEREP (constrained KG + LLM)
"""
from typing import List, Dict, Any, Set, Optional
import csv
import io

import networkx as nx

from backend.reasoning.path_extractor import PathExtractor
from backend.reasoning.template_builder import build_prompt, build_comparison_prompt
from backend.reasoning.constraint_engine import ConstraintEngine
from backend.reasoning.llm_wrapper import OllamaClient
from backend.reasoning.hallucination_checker import HallucinationChecker
from backend.evaluation.baseline_llm import run_baseline
from backend.evaluation.metrics import _analyse_text
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.core.logging import get_logger

logger = get_logger("evaluation.ablation")


def parse_genes_from_upload(content: bytes, *, filename: str = "") -> list[str]:
    """Parse CSV/TSV bytes -> deduplicated gene symbol list.
    Looks for a column named 'gene', 'gene_symbol', or 'symbol';
    falls back to first column."""
    text = content.decode("utf-8-sig").strip()
    if not text:
        return []
    is_tsv = filename.lower().endswith(".tsv") or "\t" in text.splitlines()[0]
    dialect = "excel-tab" if is_tsv else "excel"
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    fieldnames = list(reader.fieldnames or [])
    lower_fields = [f.lower() for f in fieldnames]
    target_col: Optional[str] = next(
        (fieldnames[lower_fields.index(c)] for c in ("gene", "gene_symbol", "symbol")
         if c in lower_fields),
        None,
    )
    seen, genes = set(), []
    if target_col:
        for row in reader:
            v = row.get(target_col, "").strip()
            if v and v not in seen:
                seen.add(v)
                genes.append(v)
    else:
        plain = csv.reader(io.StringIO(text), dialect=dialect)
        # Skip header if it exists (very naive)
        next(plain, None)
        for row in plain:
            v = row[0].strip() if row and row[0] else ""
            if v and v not in seen:
                seen.add(v)
                genes.append(v)
    return genes


async def run_ablation(
    genes: List[str],
    kg_builder: CERAPGraphBuilder,
    max_hops: int = 4,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Execute 3-condition ablation and return full comparison table.
    """
    graph = kg_builder.graph
    kg_entity_set: Set[str] = {n.upper() for n in graph.nodes}

    # ── Extract KG paths ──────────────────────────────────────────────────────
    extractor = PathExtractor(kg_builder)
    path_result = extractor.extract(genes, max_hops=max_hops, top_k=top_k)
    paths = path_result.get("paths", [])
    path_nodes = list({n for p in paths for n in p.get("nodes", [])})

    # ── Constraint engine ─────────────────────────────────────────────────────
    constraint_engine = ConstraintEngine(graph)
    constraint_ctx = constraint_engine.build_prompt_context(paths)

    # ── Hallucination checker ─────────────────────────────────────────────────
    hchecker = HallucinationChecker(graph)

    client = OllamaClient()
    llm_available = await client.is_available()

    # ── Condition 1: No KG (baseline raw LLM) ────────────────────────────────
    logger.info("Ablation — Condition 1: No KG baseline")
    if llm_available:
        cond1_result = await run_baseline(genes)
        cond1_text = cond1_result.get("explanation", "")
    else:
        cond1_text = "[LLM unavailable — Ollama not running]"

    cond1_metrics = _analyse_text(cond1_text, path_nodes, kg_entity_set)

    # ── Condition 2: KG paths only (no constraint decoding) ──────────────────
    logger.info("Ablation — Condition 2: KG paths injected, no constraint")
    paths_only_prompt = (
        f"You are a precision oncology expert. The following knowledge graph paths "
        f"describe the biology of genes {', '.join(genes)}. Summarise the mechanism.\n\n"
        + "\n".join(f"  • {p.get('readable', '')}" for p in paths[:8])
        + "\n\nExplanation:"
    )
    if llm_available:
        cond2_resp = await client.generate(paths_only_prompt)
        cond2_text = cond2_resp.text
    else:
        cond2_text = "[LLM unavailable]"

    cond2_metrics = _analyse_text(cond2_text, path_nodes, kg_entity_set)

    # ── Condition 3: Full CEREP (constrained KG + LLM) ───────────────────────
    logger.info("Ablation — Condition 3: Full CEREP")
    full_prompt = build_prompt(paths, constraint_ctx)
    if llm_available:
        cond3_resp = await client.generate(full_prompt)
        cond3_text = cond3_resp.text
        hall_check = hchecker.check(cond3_text, path_nodes)
    else:
        cond3_text = "[LLM unavailable]"
        hall_check = {"hallucination_rate": None, "confidence_score": None}

    cond3_metrics = _analyse_text(cond3_text, path_nodes, kg_entity_set)

    return {
        "genes": genes,
        "path_count": len(paths),
        "path_nodes": path_nodes,
        "kg_available": True,
        "llm_available": llm_available,
        "conditions": {
            "no_kg": {
                "label": "No KG (baseline LLM)",
                "explanation": cond1_text,
                "metrics": cond1_metrics,
            },
            "kg_paths_only": {
                "label": "KG Paths Only",
                "explanation": cond2_text,
                "metrics": cond2_metrics,
            },
            "full_cerep": {
                "label": "Full CEREP",
                "explanation": cond3_text,
                "metrics": {**cond3_metrics, **hall_check},
            },
        },
        "graph": path_result.get("graph"),
    }
