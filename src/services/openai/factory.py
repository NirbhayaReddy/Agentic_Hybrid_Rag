from functools import lru_cache

from src.config import get_settings
from src.services.openai.client import OpenAIClient


@lru_cache(maxsize=1)
def make_openai_client() -> OpenAIClient:
    """Create and return a singleton OpenAI client instance."""
    settings = get_settings()
    return OpenAIClient(settings)
