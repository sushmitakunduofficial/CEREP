"""
Worker Tasks — async task functions wrapping pipeline orchestration.

Implements the CEREP two-stage reasoning pipeline:
    Stage 1: KG path extraction → KG-Trie compilation → constrained decoding
    Stage 2: Fusion decoder synthesizes constrained paths into clinical narrative

Each task corresponds to a JobType and accepts a job context.
"""
import json
from typing import List, Optional, Dict, Any

from backend.graph.graph_builder import CERAPGraphBuilder
from backend.reasoning.path_extractor import PathExtractor
from backend.reasoning.constraint_engine import ConstraintEngine
from backend.reasoning.constrained_decoder import ConstrainedDecoder
from backend.reasoning.fusion_decoder import FusionDecoder
from backend.reasoning.template_builder import build_prompt
from backend.reasoning.llm_wrapper import OllamaClient
from backend.reasoning.hallucination_checker import HallucinationChecker
from backend.evaluation.ablation import run_ablation
from backend.core.logging import get_logger

logger = get_logger("worker.tasks")


async def analysis_task(
    kg_builder: CERAPGraphBuilder,
    genes: List[str],
    max_hops: int = 4,
    top_k: int = 10,
) -> dict:
    """
    Analysis task: pure KG path extraction without LLM.
    Returns paths and Cytoscape graph for visualisation.
    """
    extractor = PathExtractor(kg_builder)
    result = extractor.extract(genes, max_hops=max_hops, top_k=top_k)
    logger.info(
        "Analysis task complete",
        extra={"extra": {"genes": genes, "paths_found": result.get("total_paths_found", 0)}}
    )
    return result


async def reasoning_task(
    kg_builder: CERAPGraphBuilder,
    genes: List[str],
    max_hops: int = 4,
    top_k: int = 10,
    temperature: Optional[float] = None,
) -> dict:
    """
    Reasoning task: full CEREP two-stage constrained pipeline.

    Stage 1: Extract KG paths → compile KG-Trie → constrained decoding
    Stage 2: Fuse constrained outputs → synthesize clinical narrative
    """
    graph = kg_builder.graph

    # ── Stage 0: Path extraction ─────────────────────────────────────────────
    extractor = PathExtractor(kg_builder)
    path_result = extractor.extract(genes, max_hops=max_hops, top_k=top_k)
    paths = path_result.get("paths", [])
    path_nodes = list({n for p in paths for n in p.get("nodes", [])})

    if not paths:
        return {
            "type": "reasoning",
            "genes": genes,
            "paths": [],
            "graph": path_result.get("graph"),
            "explanation": "No reasoning paths found in the knowledge graph for the given genes.",
            "hallucination_check": {"hallucination_rate": 0.0, "confidence_score": 0.0},
            "confidence_score": 0.0,
            "stage1_result": None,
            "stage2_result": None,
            "path_count": 0,
        }

    # ── Stage 1: Constrained decoding ────────────────────────────────────────
    constraint_engine = ConstraintEngine(graph)
    decoder = ConstrainedDecoder(constraint_engine=constraint_engine)

    # Build the reasoning prompt
    constraint_ctx = constraint_engine.build_prompt_context(paths)
    prompt = build_prompt(paths, constraint_ctx)

    stage1_result = await decoder.generate_constrained(
        prompt=prompt,
        kg_paths=paths,
        beam_width=min(5, len(paths)),  # don't use more beams than paths
        max_new_tokens=256,
        temperature=temperature or 0.1,
    )

    # Extract best constrained output
    best_output = stage1_result.best()
    stage1_text = best_output.text if best_output else ""

    logger.info(
        "Stage 1 complete",
        extra={"extra": {
            "backend": stage1_result.backend_used,
            "beams": len(stage1_result.outputs),
            "best_valid": best_output.is_valid if best_output else False,
        }}
    )

    # ── Stage 2: Fusion decoder ──────────────────────────────────────────────
    fusion = FusionDecoder()
    constrained_outputs = [o.__dict__ if hasattr(o, '__dict__') else o for o in stage1_result.outputs]

    # Collect provenance from paths
    provenance_chain = []
    for p in paths:
        for edge in p.get("edges", []):
            prov = edge.get("provenance", {})
            if isinstance(prov, dict):
                provenance_chain.append({
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "predicate": edge.get("predicate", edge.get("edge_type", "")),
                    "pmids": prov.get("pmids", []),
                    "source_database": prov.get("source_database", ""),
                })

    # Try evidence retrieval (optional — don't block on failure)
    evidence_data = []
    try:
        from backend.evidence.evidence_retriever import EvidenceRetriever
        retriever = EvidenceRetriever()
        report = await retriever.get_evidence_for_genes(genes[:5])
        evidence_data = [e.to_dict() for e in report.entries[:10]]
    except Exception as exc:
        logger.info(f"Evidence retrieval skipped: {exc}")

    stage2_result = await fusion.synthesize(
        constrained_outputs=[{"text": o.text, "score": o.score} for o in stage1_result.outputs if o.is_valid],
        query=", ".join(genes),
        evidence=evidence_data,
        provenance=provenance_chain,
    )

    explanation = stage2_result.narrative or stage1_text

    logger.info(
        "Stage 2 complete",
        extra={"extra": {
            "provider": stage2_result.provider,
            "model": stage2_result.model_used,
            "tokens": stage2_result.tokens_used,
        }}
    )

    # ── Hallucination check ──────────────────────────────────────────────────
    hchecker = HallucinationChecker(graph)
    hall_result = hchecker.check(explanation, path_nodes) if explanation else {
        "hallucination_rate": None,
        "confidence_score": None,
        "flagged_entities": [],
    }

    logger.info(
        "Reasoning task complete (two-stage)",
        extra={"extra": {
            "genes": genes,
            "hallucination_rate": hall_result.get("hallucination_rate"),
            "stage1_backend": stage1_result.backend_used,
            "stage2_provider": stage2_result.provider,
        }}
    )

    return {
        "type": "reasoning",
        "genes": genes,
        "paths": paths,
        "graph": path_result.get("graph"),
        "explanation": explanation,
        "hallucination_check": hall_result,
        "confidence_score": hall_result.get("confidence_score"),
        "path_count": len(paths),
        # Two-stage metadata
        "stage1_result": stage1_result.to_dict(),
        "stage2_result": stage2_result.to_dict(),
        "evidence": evidence_data,
        "provenance_chain": provenance_chain,
    }


async def evaluation_task(
    kg_builder: CERAPGraphBuilder,
    genes: List[str],
    max_hops: int = 4,
    top_k: int = 10,
) -> dict:
    """Evaluation task: runs full 3-condition ablation study."""
    result = await run_ablation(genes, kg_builder, max_hops=max_hops, top_k=top_k)
    logger.info(
        "Evaluation task complete",
        extra={"extra": {"genes": genes, "conditions": list(result.get("conditions", {}).keys())}}
    )
    return result
