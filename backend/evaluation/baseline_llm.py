"""
Baseline LLM — runs unconstrained LLM inference (no KG context).
Used as comparison baseline in ablation studies.
"""
from typing import List, Dict, Any

from backend.reasoning.llm_wrapper import OllamaClient, LLMResponse
from backend.reasoning.template_builder import build_comparison_prompt
from backend.core.logging import get_logger

logger = get_logger("evaluation.baseline_llm")


async def run_baseline(
    genes: List[str],
    model: str | None = None,
    temperature: float | None = None,
) -> Dict[str, Any]:
    """
    Run an unconstrained LLM query for the given gene list.
    Returns a dict compatible with evaluation metrics.
    """
    client = OllamaClient(model=model, temperature=temperature)
    gene_str = ", ".join(genes)

    prompt = build_comparison_prompt(gene_str)
    response: LLMResponse = await client.generate(prompt)

    logger.info(
        "Baseline LLM complete",
        extra={"extra": {"genes": genes, "text_length": len(response.text)}}
    )
    return {
        "mode": "baseline_llm",
        "genes": genes,
        "prompt": prompt,
        "explanation": response.text,
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "error": response.error,
        # No KG grounding — hallucination check done externally in ablation
    }
