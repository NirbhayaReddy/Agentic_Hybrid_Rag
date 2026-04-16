import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from src.config import Settings
from src.services.ollama.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client with the same interface as OllamaClient.

    Drop-in replacement — all agent nodes and routers work unchanged.
    """

    def __init__(self, settings: Settings):
        self.api_key = settings.openai_api_key
        self.default_model = settings.openai_model
        self.prompt_builder = RAGPromptBuilder()
        self._async_client = AsyncOpenAI(api_key=self.api_key)

    def get_langchain_model(self, model: str, temperature: float = 0.0) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance for use in agent nodes."""
        return ChatOpenAI(
            api_key=self.api_key,
            model=model or self.default_model,
            temperature=temperature,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity."""
        try:
            models = await self._async_client.models.list()
            return {
                "status": "healthy",
                "message": "OpenAI API is reachable",
                "provider": "openai",
            }
        except Exception as e:
            raise Exception(f"OpenAI health check failed: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models."""
        try:
            response = await self._async_client.models.list()
            return [{"id": m.id} for m in response.data]
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate a RAG answer using retrieved chunks via OpenAI."""
        model_to_use = model or self.default_model
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)

        try:
            response = await self._async_client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            answer_text = response.choices[0].message.content or ""

            sources = []
            seen_urls: set = set()
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id", "")
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                    if pdf_url not in seen_urls:
                        sources.append(pdf_url)
                        seen_urls.add(pdf_url)

            citations = list({c.get("arxiv_id") for c in chunks if c.get("arxiv_id")})

            return {
                "answer": answer_text,
                "sources": sources,
                "confidence": "medium",
                "citations": citations[:5],
            }

        except Exception as e:
            logger.error(f"OpenAI RAG answer failed: {e}")
            raise Exception(f"Failed to generate RAG answer via OpenAI: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer using retrieved chunks via OpenAI."""
        model_to_use = model or self.default_model
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)

        try:
            stream = await self._async_client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield {"response": delta, "done": False}
            yield {"response": "", "done": True}

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise Exception(f"Failed to stream RAG answer via OpenAI: {e}")
