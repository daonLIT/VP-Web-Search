from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from app.config import SETTINGS

def get_chroma(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=SETTINGS.chroma_collection,
        persist_directory=SETTINGS.chroma_persist_dir,
        embedding_function=embeddings,
    )
