from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "agent_knowledge")

SETTINGS = Settings()
