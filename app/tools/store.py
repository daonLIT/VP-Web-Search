from typing import Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from app.config import SETTINGS

def get_chroma(embeddings: Embeddings) -> Chroma:
    # LangChain Chroma 벡터스토어 사용
    # (persist_directory, collection_name로 로컬 영속화)
    return Chroma(
        collection_name=SETTINGS.chroma_collection,
        persist_directory=SETTINGS.chroma_persist_dir,
        embedding_function=embeddings,
    )
