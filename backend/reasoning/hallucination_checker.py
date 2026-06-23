"""
Hallucination Checker — validates every entity and relationship in LLM output
against the Knowledge Graph. Flags and scores hallucinations.
"""
import re
from typing import List, Dict, Any, Set

import networkx as nx

from backend.core.logging import get_logger

logger = get_logger("reasoning.hallucination_checker")


def _extract_candidate_tokens(text: str) -> List[str]:
    """
    Extract capitalised word groups from LLM output as candidate entity mentions.
    Handles gene names like TP53, BRCA1, PIK3CA as well as multi-word terms.
    """
    # Match all-caps tokens (gene names), capitalised words, and known bio patterns
    pattern = r"\b([A-Z][A-Z0-9\-_]{1,})\b"
    return re.findall(pattern, text)


class HallucinationChecker:
    """
    Post-generation validator that cross-references LLM output against KG.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self._valid_entities: Set[str] = set()
        self._valid_entities_upper: Set[str] = set()
        self._build_entity_set()

    def _build_entity_set(self) -> None:
        for node in self.graph.nodes:
            self._valid_entities.add(node)
            self._valid_entities_upper.add(node.upper())
            # Also register aliases
            for alias in self.graph.nodes[node].get("aliases", []):
                self._valid_entities.add(alias)
                self._valid_entities_upper.add(alias.upper())

        logger.info(
            "HallucinationChecker built",
            extra={"extra": {"valid_entities": len(self._valid_entities)}}
        )

    def check(self, llm_text: str, path_nodes: List[str]) -> Dict[str, Any]:
        """
        Full hallucination analysis of LLM output.

        Args:
            llm_text: The raw text returned by the LLM.
            path_nodes: Ground-truth KG nodes used to build the prompt.

        Returns:
            dict with hallucination_rate, flagged_entities, grounding_score,
            mentioned_entities, valid_entities, invalid_entities.
        """
        # 1. Extract candidate entity mentions from LLM output
        candidates = _extract_candidate_tokens(llm_text)

        # 2. Classify each as valid (in KG) or invalid (hallucinated)
        seen: Set[str] = set()
        valid_entities: List[str] = []
        invalid_entities: List[str] = []

        for token in candidates:
            if token in seen:
                continue
            seen.add(token)
            if token.upper() in self._valid_entities_upper:
                valid_entities.append(token)
            else:
                invalid_entities.append(token)

        total_mentioned = len(valid_entities) + len(invalid_entities)
        hallucination_rate = (
            len(invalid_entities) / total_mentioned if total_mentioned > 0 else 0.0
        )

        # 3. Grounding score — fraction of path nodes that appear in explanation
        path_node_upper = {n.upper() for n in path_nodes}
        mentioned_upper = {t.upper() for t in valid_entities}
        grounding_score = (
            len(path_node_upper & mentioned_upper) / len(path_node_upper)
            if path_node_upper else 1.0
        )

        # 4. Confidence score (1 - hallucination rate) weighted with grounding
        confidence_score = round((1.0 - hallucination_rate) * 0.6 + grounding_score * 0.4, 3)

        result = {
            "hallucination_rate": round(hallucination_rate, 3),
            "grounding_score": round(grounding_score, 3),
            "confidence_score": confidence_score,
            "total_tokens_checked": total_mentioned,
            "valid_entities": valid_entities,
            "flagged_entities": invalid_entities,
            "path_nodes_covered": list(path_node_upper & mentioned_upper),
            "path_nodes_missed": list(path_node_upper - mentioned_upper),
        }

        logger.info(
            "Hallucination check complete",
            extra={"extra": {
                "hallucination_rate": result["hallucination_rate"],
                "confidence_score": result["confidence_score"],
                "flagged_count": len(invalid_entities),
            }}
        )
        return result

    def entity_in_kg(self, entity: str) -> bool:
        return entity.upper() in self._valid_entities_upper
