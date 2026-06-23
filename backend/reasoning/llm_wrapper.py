"""
LLM Wrapper — async HTTP client to Ollama /api/generate.
Handles streaming aggregation, timeout, retry, and structured response parsing.
"""
import asyncio
import json
from typing import Optional, Dict, Any

import httpx

from backend.core.config import get_settings
from backend.core.logging import get_logger

settings = get_settings()
logger = get_logger("reasoning.llm_wrapper")


class LLMResponse:
    """Structured LLM output."""
    def __init__(
        self,
        text: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
    ):
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


class OllamaClient:
    """Async HTTP client for Ollama /api/generate."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries

    async def generate(self, prompt: str) -> LLMResponse:
        """
        Send prompt to Ollama and return aggregated response.
        Retries up to max_retries on connection errors.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._call_ollama(payload)
            except httpx.ConnectError as e:
                last_error = f"Ollama connection refused (attempt {attempt + 1}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
            except httpx.TimeoutException as e:
                last_error = f"Ollama timeout after {self.timeout}s: {e}"
                logger.warning(last_error)
                break
            except Exception as e:
                last_error = f"Unexpected LLM error: {e}"
                logger.error(last_error)
                break

        # Fallback: return error response without crashing the pipeline
        return LLMResponse(
            text="",
            model=self.model,
            error=last_error or "LLM unavailable",
        )

    async def _call_ollama(self, payload: dict) -> LLMResponse:
        """Internal: make the actual streaming HTTP call to Ollama."""
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    full_text += chunk.get("response", "")
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        break

        logger.info(
            "LLM generation complete",
            extra={"extra": {
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "text_length": len(full_text),
            }}
        )
        return LLMResponse(
            text=full_text.strip(),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def is_available(self) -> bool:
        """Health-check Ollama availability."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
