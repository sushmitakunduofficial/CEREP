"""
CEREP Reasoning Module — two-stage graph-constrained reasoning pipeline.

Stage 1: ConstrainedDecoder (vLLM + KGTrieLogitsProcessor + beam search)
Stage 2: FusionDecoder (GPT-4.1 / Claude / Llama 70B synthesis)

Public API:
    ConstraintEngine      — entity/edge validation + KG-Trie compilation
    ConstrainedDecoder    — Stage 1 constrained decoding
    FusionDecoder         — Stage 2 fusion synthesis
    PathExtractor         — KG path extraction and ranking
"""
from backend.reasoning.constraint_engine import ConstraintEngine, KGTrie, TokenKGTrie
from backend.reasoning.constrained_decoder import ConstrainedDecoder, ConstrainedDecoderResult
from backend.reasoning.fusion_decoder import FusionDecoder, FusionResult
from backend.reasoning.path_extractor import PathExtractor

__all__ = [
    "ConstraintEngine",
    "KGTrie",
    "TokenKGTrie",
    "ConstrainedDecoder",
    "ConstrainedDecoderResult",
    "FusionDecoder",
    "FusionResult",
    "PathExtractor",
]
