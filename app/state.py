from typing import Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict, total=False):
    incoming: Dict[str, Any]          # 외부 시스템 payload (원문)
    query: str                        # 검색용 질의(키워드)
    judgment: Dict[str, Any]          # 판단 결과(카테고리/도메인/최근성 등)
    route: Literal["HIT", "MISS"]     # compare 결과
    hits: List[Document]              # VectorDB에서 찾은 문서들
    hit_scores: List[float]           # relevance scores
    web_results: List[Dict[str, Any]] # Tavily 결과(raw)
    report: str                       # 요약/정리 결과(외부 전달용)
    to_store: List[Document]          # 저장할 문서(chunks)
    outgoing: Dict[str, Any]          # 외부로 보낼 payload
