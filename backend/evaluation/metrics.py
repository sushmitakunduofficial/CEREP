"""
Evaluation Metrics — computes hallucination rate, entity coverage,
path alignment score, and KG grounding ratio.
"""
from typing import List, Dict, Any, Set
import re


def compute_metrics(
    cerep_output: Dict[str, Any],
    baseline_output: Dict[str, Any],
    path_nodes: List[str],
    kg_entity_set: Set[str],
) -> Dict[str, Any]:
    """
    Compute a full metrics comparison between CEREP and baseline outputs.

    Args:
        cerep_output: Full reasoning result from the constrained pipeline.
        baseline_output: Output from unconstrained baseline_llm.
        path_nodes: Ground-truth KG nodes used in the constrained prompt.
        kg_entity_set: Full set of KG entity IDs (uppercase).

    Returns:
        Dict with per-condition metrics and comparison table.
    """
    cerep_text = cerep_output.get("explanation", "")
    baseline_text = baseline_output.get("explanation", "")

    cerep_metrics = _analyse_text(cerep_text, path_nodes, kg_entity_set)
    baseline_metrics = _analyse_text(baseline_text, path_nodes, kg_entity_set)

    return {
        "cerep": {**cerep_metrics, "mode": "cerep"},
        "baseline": {**baseline_metrics, "mode": "baseline"},
        "delta": {
            "hallucination_rate": round(
                baseline_metrics["hallucination_rate"] - cerep_metrics["hallucination_rate"], 3
            ),
            "entity_coverage": round(
                cerep_metrics["entity_coverage"] - baseline_metrics["entity_coverage"], 3
            ),
            "grounding_ratio": round(
                cerep_metrics["grounding_ratio"] - baseline_metrics["grounding_ratio"], 3
            ),
        },
    }


def _analyse_text(
    text: str,
    path_nodes: List[str],
    kg_entity_set: Set[str],
) -> Dict[str, Any]:
    """Internal: compute all metrics for a single text output."""
    # Extract uppercase bio-tokens from text
    tokens = set(re.findall(r"\b([A-Z][A-Z0-9\-_]{1,})\b", text))

    valid = {t for t in tokens if t.upper() in kg_entity_set}
    invalid = tokens - valid
    total = len(tokens)

    hallucination_rate = len(invalid) / total if total > 0 else 0.0

    # Entity coverage — fraction of path nodes mentioned
    path_upper = {n.upper() for n in path_nodes}
    covered = {t.upper() for t in valid} & path_upper
    entity_coverage = len(covered) / len(path_upper) if path_upper else 1.0

    # Path alignment — fraction of valid tokens that are in path nodes
    path_aligned = {t for t in valid if t.upper() in path_upper}
    path_alignment_score = len(path_aligned) / len(valid) if valid else 0.0

    # Grounding ratio — valid KG entities / total tokens
    grounding_ratio = len(valid) / total if total > 0 else 0.0

    return {
        "hallucination_rate": round(hallucination_rate, 3),
        "entity_coverage": round(entity_coverage, 3),
        "path_alignment_score": round(path_alignment_score, 3),
        "grounding_ratio": round(grounding_ratio, 3),
        "total_tokens": total,
        "valid_entities": list(valid),
        "invalid_entities": list(invalid),
        "covered_path_nodes": list(covered),
    }
