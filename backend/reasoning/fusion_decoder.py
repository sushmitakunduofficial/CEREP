"""
Fusion Decoder — Stage 2 of CEREP's two-stage reasoning pipeline.

Implements the Fusion-in-Decoder paradigm:
    1. Receives multiple validated constrained paths from Stage 1
    2. Constructs a structured context block with all valid paths + evidence
    3. Sends to a general-purpose LLM (GPT-4.1 / Claude / Llama 70B)
    4. Synthesizes a coherent, clinically-actionable narrative
    5. No KG-Trie constraint on this stage (paths are already validated)

The fusion decoder focuses on synthesis quality, not constraint enforcement.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger("reasoning.fusion_decoder")
settings = get_settings()


@dataclass
class FusionResult:
    """Result from Stage 2 fusion decoding."""
    narrative: str                              # synthesized clinical narrative
    constrained_paths_used: int = 0            # number of Stage 1 paths consumed
    model_used: str = ""                       # model identifier
    provider: str = ""                         # openai | ollama | vllm
    tokens_used: int = 0
    evidence_pmids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "constrained_paths_used": self.constrained_paths_used,
            "model": self.model_used,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "evidence_pmids": self.evidence_pmids,
            "confidence": self.confidence,
        }


class FusionDecoder:
    """Stage 2 — synthesizes constrained reasoning paths into clinical narrative.

    Tries providers in order:
        1. OpenAI-compatible API (GPT-4.1 / Claude via API)
        2. vLLM with large model (Llama 70B)
        3. Ollama (fallback)
    """

    SYSTEM_PROMPT = """You are CEREP (Computational Explainable Reasoning Engine for Precision Oncology), a clinical reasoning AI that synthesizes validated molecular pathway information into actionable clinical narratives.

Your task is to create a coherent, clinically-grounded explanation from the validated reasoning paths provided below. These paths have been extracted from a curated biomedical knowledge graph and verified through graph-constrained decoding — every entity and relationship is biologically validated.

Requirements:
1. ONLY reference entities and relationships from the provided paths
2. Explain the molecular mechanism step-by-step
3. Connect findings to clinical actionability (drug sensitivities, treatment implications)
4. Cite PMIDs where provided
5. Note confidence levels and evidence quality
6. Use precise biomedical terminology
7. Structure your response with clear sections: Molecular Mechanism, Clinical Significance, Treatment Implications

Do NOT speculate beyond the provided evidence. Do NOT introduce entities not in the paths."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._provider = provider or settings.fusion_provider
        self._model = model or settings.fusion_model
        self._api_key = api_key or settings.fusion_api_key
        self._base_url = base_url or settings.fusion_base_url

    async def synthesize(
        self,
        constrained_outputs: List[Dict[str, Any]],
        query: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> FusionResult:
        """Synthesize constrained outputs into a coherent clinical narrative.

        Args:
            constrained_outputs: Stage 1 outputs (text + metadata)
            query: Original user query / gene set
            evidence: Optional evidence from the evidence layer
            provenance: Provenance chain from path extraction

        Returns:
            FusionResult with the synthesized narrative
        """
        # Build the fusion prompt
        prompt = self._build_fusion_prompt(
            constrained_outputs, query, evidence, provenance
        )

        # Try providers in order
        result = await self._try_openai_compatible(prompt)
        if result:
            result.constrained_paths_used = len(constrained_outputs)
            return result

        result = await self._try_vllm_large(prompt)
        if result:
            result.constrained_paths_used = len(constrained_outputs)
            return result

        result = await self._try_ollama(prompt)
        if result:
            result.constrained_paths_used = len(constrained_outputs)
            return result

        # No provider available
        return FusionResult(
            narrative=self._build_fallback_narrative(constrained_outputs, query),
            constrained_paths_used=len(constrained_outputs),
            model_used="none",
            provider="fallback",
        )

    def _build_fusion_prompt(
        self,
        constrained_outputs: List[Dict[str, Any]],
        query: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the structured fusion prompt."""
        sections = [
            f"=== PATIENT QUERY ===",
            f"Genes/variants of interest: {query}",
            "",
            f"=== VALIDATED REASONING PATHS (from Stage 1 constrained decoding) ===",
        ]

        for i, output in enumerate(constrained_outputs):
            text = output.get("text", "")
            score = output.get("score", 0)
            sections.append(f"\n--- Path {i+1} (confidence: {score:.3f}) ---")
            sections.append(text)

        if provenance:
            sections.append("\n=== PROVENANCE CHAIN ===")
            for p in provenance[:10]:
                pmids = p.get("pmids", [])
                pmid_str = ", ".join(f"PMID:{pm}" for pm in pmids) if pmids else "no PMID"
                sections.append(
                    f"  {p.get('source', '?')} → {p.get('target', '?')} "
                    f"[{p.get('predicate', '?')}] ({pmid_str})"
                )

        if evidence:
            sections.append("\n=== BIOLOGICAL EVIDENCE ===")
            for ev in evidence[:10]:
                sections.append(
                    f"  [{ev.get('source', '?')}] {ev.get('gene', '?')}: "
                    f"{ev.get('summary', '')}"
                )

        sections.append("\n=== TASK ===")
        sections.append(
            "Synthesize the above validated paths and evidence into a coherent "
            "clinical narrative. Explain the molecular mechanism, its clinical "
            "significance, and treatment implications."
        )

        return "\n".join(sections)

    # ── OpenAI-compatible API ────────────────────────────────────────────────

    async def _try_openai_compatible(self, prompt: str) -> Optional[FusionResult]:
        """Try GPT-4.1 / Claude / any OpenAI-compatible API."""
        if not self._api_key:
            return None

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            body = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": settings.fusion_temperature,
                "max_tokens": settings.fusion_max_tokens,
            }

            async with httpx.AsyncClient(timeout=settings.fusion_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                if response.status_code != 200:
                    logger.warning(f"OpenAI API returned {response.status_code}: {response.text[:200]}")
                    return None

                data = response.json()

            narrative = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            logger.info(f"Fusion via OpenAI API ({self._model}): {tokens} tokens")

            return FusionResult(
                narrative=narrative,
                model_used=self._model,
                provider="openai",
                tokens_used=tokens,
            )

        except Exception as exc:
            logger.info(f"OpenAI API unavailable: {exc}")
            return None

    # ── vLLM Large Model ─────────────────────────────────────────────────────

    async def _try_vllm_large(self, prompt: str) -> Optional[FusionResult]:
        """Try vLLM with a large model (Llama 70B) for fusion."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                health = await client.get(f"{settings.vllm_base_url}/health")
                if health.status_code != 200:
                    return None

            # Use chat completions format
            body = {
                "model": settings.vllm_model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": settings.fusion_temperature,
                "max_tokens": settings.fusion_max_tokens,
            }

            async with httpx.AsyncClient(timeout=settings.fusion_timeout) as client:
                response = await client.post(
                    f"{settings.vllm_base_url}/v1/chat/completions",
                    json=body,
                )
                if response.status_code != 200:
                    return None

                data = response.json()

            narrative = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return FusionResult(
                narrative=narrative,
                model_used=settings.vllm_model,
                provider="vllm",
                tokens_used=tokens,
            )

        except Exception as exc:
            logger.info(f"vLLM large model unavailable: {exc}")
            return None

    # ── Ollama Fallback ──────────────────────────────────────────────────────

    async def _try_ollama(self, prompt: str) -> Optional[FusionResult]:
        """Try Ollama as a fusion fallback."""
        try:
            from backend.reasoning.llm_wrapper import OllamaClient
            client = OllamaClient()
            if not await client.is_available():
                return None

            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
            response = await client.generate(full_prompt)

            return FusionResult(
                narrative=response.text,
                model_used=settings.llm_model,
                provider="ollama",
                tokens_used=len(response.text.split()),
            )

        except Exception as exc:
            logger.info(f"Ollama unavailable for fusion: {exc}")
            return None

    # ── Structured Fallback (no LLM available) ───────────────────────────────

    def _build_fallback_narrative(
        self, constrained_outputs: List[Dict[str, Any]], query: str
    ) -> str:
        """Build a structured narrative without LLM when no provider is available."""
        lines = [
            f"## Molecular Analysis: {query}",
            "",
            "### Validated Reasoning Paths",
            f"CEREP identified {len(constrained_outputs)} validated reasoning path(s) "
            "from the biomedical knowledge graph.",
            "",
        ]
        for i, output in enumerate(constrained_outputs):
            text = output.get("text", "No path data")
            score = output.get("score", 0)
            lines.append(f"**Path {i+1}** (confidence: {score:.3f}):")
            lines.append(f"> {text}")
            lines.append("")

        lines.append("### Note")
        lines.append(
            "No LLM backend was available for narrative synthesis. "
            "The above paths are raw graph-constrained outputs. "
            "Configure vLLM, OpenAI API, or Ollama for full clinical narrative generation."
        )
        return "\n".join(lines)
