from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # API keys
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Model
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # Chroma
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "agent_knowledge")

    # Retrieval
    default_top_k: int = int(os.getenv("TOP_K", "5"))
    default_min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.80"))

SETTINGS = Settings()
