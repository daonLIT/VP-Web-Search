# app/config.py
"""
설정 관리
"""
from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """애플리케이션 설정"""

    # API Keys
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Model
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))

    # Chroma (optional, for future use)
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "research_data"))

    # Search
    default_max_results: int = field(default_factory=lambda: int(os.getenv("MAX_RESULTS", "3")))
    default_search_depth: str = field(default_factory=lambda: os.getenv("SEARCH_DEPTH", "basic"))

    # Crawling
    crawl_timeout: int = field(default_factory=lambda: int(os.getenv("CRAWL_TIMEOUT", "10")))
    max_content_length: int = field(default_factory=lambda: int(os.getenv("MAX_CONTENT_LENGTH", "3000")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Webhook - VP2로 결과 전송
    webhook_url: str = field(default_factory=lambda: os.getenv("WEBHOOK_URL", ""))
    webhook_timeout: int = field(default_factory=lambda: int(os.getenv("WEBHOOK_TIMEOUT", "30")))


# 싱글톤 설정 인스턴스
SETTINGS = Settings()
