"""
Template Builder — converts ranked KG paths into structured LLM prompt templates.
"""
from typing import List, Dict, Any

from backend.core.logging import get_logger

logger = get_logger("reasoning.template_builder")

SYSTEM_PROMPT = """You are a precision oncology expert. You will be given a set of validated knowledge graph paths from a curated biomedical database. Your ONLY job is to translate these paths into a concise mechanistic explanation.

STRICT RULES:
1. You may ONLY mention biological entities listed in the KNOWLEDGE GRAPH CONSTRAINTS section.
2. Every causal claim must map to a provided path edge.
3. Do NOT invent genes, proteins, drugs, or mechanisms not present in the paths.
4. Format your response as a structured explanation with: (a) mechanism summary, (b) key entities, (c) therapeutic implications."""


def build_prompt(paths: List[Dict[str, Any]], constraint_context: str) -> str:
    """
    Build the full LLM prompt for a reasoning job.

    Args:
        paths: list of path dicts from PathExtractor (each has 'readable', 'score', 'nodes', 'edges')
        constraint_context: string from ConstraintEngine.build_prompt_context()

    Returns:
        Full prompt string ready to send to the LLM.
    """
    if not paths:
        return _no_paths_prompt()

    # Build path section with provenance
    path_lines = []
    for i, p in enumerate(paths[:8], 1):
        readable = p.get("readable", "")
        score = p.get("score", 0.0)
        line = f"  {i}. [score={score:.2f}] {readable}"
        # Add PMID citations if available
        prov_chain = p.get("provenance_chain", [])
        pmids = set()
        for step in prov_chain:
            pmids.update(step.get("pmids", []))
        if pmids:
            line += f"  (PMIDs: {', '.join(sorted(pmids)[:3])})"
        path_lines.append(line)

    paths_block = "\n".join(path_lines)

    prompt = f"""{SYSTEM_PROMPT}

{constraint_context}

=== RANKED MECHANISTIC PATHS ===
{paths_block}
=== END PATHS ===

Based ONLY on the above paths, provide a mechanistic explanation covering:
1. The central biological mechanism
2. Key regulatory relationships
3. Clinical/therapeutic relevance
4. Confidence level (based on path scores)

Explanation:"""
    return prompt


def build_comparison_prompt(gene: str) -> str:
    """
    Build an unconstrained baseline prompt (no KG context) for the same gene.
    Used by baseline_llm evaluation.
    """
    return f"""You are a precision oncology expert. Explain the role of the gene {gene} in cancer biology, including:
1. Its primary biological function
2. How mutations affect cancer progression
3. Known therapeutic implications
4. Key interaction partners

Explanation:"""


def _no_paths_prompt() -> str:
    return (
        "No valid knowledge graph paths were found for the query genes. "
        "Unable to generate a constrained mechanistic explanation."
    )


def extract_entities_from_prompt_response(
    response_text: str, known_entities: set
) -> List[str]:
    """
    Identify which KG entities appear in an LLM response.
    Returns list of matched entity names.
    """
    import re
    found = []
    for entity in known_entities:
        pattern = r"\b" + re.escape(entity) + r"\b"
        if re.search(pattern, response_text, re.IGNORECASE):
            found.append(entity)
    return found
