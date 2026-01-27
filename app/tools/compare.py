from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma

def compare_tool(
    vectordb: Chroma,
    query: str,
    top_k: int = 5,
    min_relevance: float = 0.75,
) -> Tuple[str, List[Document], List[float]]:
    """
    similarity_search_with_relevance_scores:
    - (Document, relevance_score[0~1]) 형태로 반환되는 API 사용
    """
    results = vectordb.similarity_search_with_relevance_scores(query, k=top_k)
    docs = [d for (d, _s) in results]
    scores = [float(_s) for (_d, _s) in results]
    route = "HIT" if any(s >= min_relevance for s in scores) else "MISS"
    return route, docs, scores
