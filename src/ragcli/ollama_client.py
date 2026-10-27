"""
HTTP client for Ollama LLM and embedding generation.
"""

import logging
from typing import Any
import httpx
from ragcli.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def check_health(self) -> dict[str, Any]:
        """Check Ollama service status and listed models."""
        result: dict[str, Any] = {
            "available": False,
            "models": [],
            "llm_ready": False,
            "embedding_ready": False,
            "error": None,
        }
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    result["available"] = True
                    models = resp.json().get("models", [])
                    model_names = [m.get("name") for m in models]
                    result["models"] = model_names

                    # Check configured LLM model
                    result["llm_ready"] = any(
                        settings.llm_model == name or name.startswith(settings.llm_model)
                        for name in model_names
                    )
                    # Check configured embedding model
                    result["embedding_ready"] = any(
                        settings.embedding_model == name or name.startswith(settings.embedding_model)
                        for name in model_names
                    )
                else:
                    result["error"] = f"Ollama returned HTTP {resp.status_code}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Call Ollama LLM generation endpoint with temperature 0 for determinism."""
        selected_model = model or settings.llm_model
        payload: dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096,
                "num_predict": 1024,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        timeout_config = httpx.Timeout(600.0, connect=30.0, read=600.0)
        with httpx.Client(timeout=timeout_config) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    def get_embedding(self, text: str, model: str | None = None) -> list[float]:
        """Generate embedding vector for a single text chunk with safe size guard."""
        selected_model = model or settings.embedding_model
        # Guard: Ollama embedding models often have a 2048 token limit
        # 3000 chars is safely below ~750 tokens
        safe_text = text[:3200] if len(text) > 3200 else text

        payload = {
            "model": selected_model,
            "prompt": safe_text,
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.base_url}/api/embeddings", json=payload)
            if resp.status_code != 200 and len(safe_text) > 1500:
                # Fallback to tighter character window if model batch size is constrained
                payload["prompt"] = safe_text[:1500]
                resp = client.post(f"{self.base_url}/api/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding", [])

    def get_embeddings_batch(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        """Generate embeddings sequentially or in small batches."""
        return [self.get_embedding(text, model) for text in texts]
