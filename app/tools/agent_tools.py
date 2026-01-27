from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from datetime import datetime, timezone

from langchain_tavily import TavilySearch  # ✅ deprecated 경고 없는 최신 경로 :contentReference[oaicite:6]{index=6}

def build_tools(vectordb: Chroma) -> List[Any]:
    """
    vectordb 같은 상태(의존성)가 필요한 도구는
    tool factory(클로저)로 생성하는 게 실전에서 제일 깔끔함.
    """

    @tool("vector_search")
    def vector_search(
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.80,
    ) -> Dict[str, Any]:
        """
        ChromaDB(VectorDB)에서 query와 유사한 문서를 검색한다.
        반환:
          - route: HIT/MISS
          - hits: [{content, metadata, score}]
        """
        results = vectordb.similarity_search_with_relevance_scores(query, k=top_k)
        hits = []
        scores = []
        for doc, score in results:
            scores.append(float(score))
            hits.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })
        route = "HIT" if any(s >= float(min_relevance) for s in scores) else "MISS"
        return {"route": route, "query": query, "hits": hits, "scores": scores}

    tavily = TavilySearch(
        max_results=5,
        topic="general",            # general/news/finance :contentReference[oaicite:7]{index=7}
        include_answer=True,
        include_raw_content=True,   # 원문 근거 저장용 :contentReference[oaicite:8]{index=8}
    )

    @tool("web_search")
    def web_search(
        query: str,
        topic: str = "general",                 # "news"로 주면 뉴스 성향 :contentReference[oaicite:9]{index=9}
        max_results: int = 5,
        time_range: Optional[str] = None,       # "day","week","month","year" :contentReference[oaicite:10]{index=10}
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[str] = None,     # "basic"/"advanced" :contentReference[oaicite:11]{index=11}
    ) -> List[Dict[str, Any]]:
        """
        Tavily로 웹 검색.
        note: include_answer/include_raw_content는 신뢰성/성능 이유로 invocation에서 못 바꾸는 제한이 있음. :contentReference[oaicite:12]{index=12}
        """
        # TavilySearch는 일부 파라미터를 invoke args로 받음(문서 참조). :contentReference[oaicite:13]{index=13}
        args: Dict[str, Any] = {"query": query}

        if topic:
            args["topic"] = topic
        if max_results:
            args["max_results"] = max_results
        if time_range:
            args["time_range"] = time_range
        if include_domains:
            args["include_domains"] = include_domains
        if exclude_domains:
            args["exclude_domains"] = exclude_domains
        if search_depth:
            args["search_depth"] = search_depth

        return tavily.invoke(args)

    @tool("store_web_results")
    def store_web_results(
        query: str,
        web_results: List[Dict[str, Any]],
        kind: str = "web",
    ) -> Dict[str, Any]:
        """
        web_search 결과를 ChromaDB에 저장.
        - content: raw_content 우선, 없으면 content
        - metadata: source/url/title/fetched_at/query/kind
        """
        now = datetime.now(timezone.utc).isoformat()

        docs: List[Document] = []
        for r in web_results or []:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or r.get("source") or "").strip()
            content = (r.get("raw_content") or r.get("content") or "").strip()
            if not content:
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url or "tavily",
                        "title": title,
                        "fetched_at": now,
                        "query": query,
                        "kind": kind,
                    },
                )
            )

        if docs:
            vectordb.add_documents(docs)

        return {"stored": len(docs), "collection": getattr(vectordb, "_collection_name", None)}

    return [vector_search, web_search, store_web_results]
