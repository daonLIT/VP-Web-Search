from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from hashlib import sha256

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch, TavilyExtract


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_tavily_search_output(output: Any) -> List[Dict[str, Any]]:
    """
    TavilySearch.invoke 결과는 보통 dict {'results': [...]} 형태. (버전에 따라 list일 수도 있음)
    - list면 그대로
    - dict면 output['results'] 사용
    """
    if output is None:
        return []
    if isinstance(output, list):
        return output
    if isinstance(output, dict):
        res = output.get("results")
        return res if isinstance(res, list) else []
    return []


def build_tools(vectordb: Chroma) -> List[Any]:
    # -----------------------------
    # 1) Vector search (있지만 web-only 그래프에서는 안 씀)
    # -----------------------------
    @tool("vector_search")
    def vector_search(
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.80,
    ) -> Dict[str, Any]:
        """
        (호환/디버그용) ChromaDB(VectorDB)에서 query 유사 문서를 검색한다.
        web-only 그래프에서는 호출하지 않지만, 도구 등록을 위해 남겨둔다.

        반환:
        - route: "HIT" | "MISS"
        - query: str
        - hits: [{content, metadata, score}]
        - scores: [float]
        """
        results = vectordb.similarity_search_with_relevance_scores(query, k=int(top_k))
        hits: List[Dict[str, Any]] = []
        scores: List[float] = []
        for doc, score in results:
            s = float(score)
            scores.append(s)
            hits.append({"content": doc.page_content, "metadata": doc.metadata, "score": s})
        route = "HIT" if any(s >= float(min_relevance) for s in scores) else "MISS"
        return {"route": route, "query": query, "hits": hits, "scores": scores}

    # -----------------------------
    # 2) Tavily search (SNIPPETS ONLY)
    # -----------------------------
    tavily_snippets = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,  # ✅ 모델로 원문 안 올림
        search_depth="basic",
    )

    @tool("web_search_snippets")
    def web_search_snippets(
        query: str,
        topic: str = "general",
        max_results: int = 5,
        time_range: Optional[str] = None,       # day/week/month/year
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[str] = None,     # basic/advanced
    ) -> List[Dict[str, Any]]:
        """
        Tavily로 웹 검색을 수행하되, 모델 컨텍스트 토큰 폭발을 막기 위해
        raw_content(본문)는 절대 포함하지 않고 '짧은 스니펫'만 반환한다.

        사용 목적:
        - 에이전트가 어떤 URL/문서가 유의미한지 빠르게 판단
        - 후속 단계에서 web_fetch_and_store로 원문을 추출/저장하기 위한 후보 수집

        입력:
        - query: 검색 질의
        - topic: "general" | "news" | "finance"
        - max_results: 결과 개수(기본 5)
        - time_range: "day" | "week" | "month" | "year"
        - include_domains / exclude_domains: 도메인 필터
        - search_depth: "basic" | "advanced"

        출력(리스트):
        - 각 항목: {title, url, content, score}
        - content는 요약/스니펫 수준의 짧은 텍스트만 포함
        """
        args: Dict[str, Any] = {"query": query}

        # invocation에서 바꿀 수 있는 파라미터만 세팅
        if topic:
            args["topic"] = topic
        if time_range:
            args["time_range"] = time_range
        if include_domains:
            args["include_domains"] = include_domains
        if exclude_domains:
            args["exclude_domains"] = exclude_domains
        if search_depth:
            args["search_depth"] = search_depth

        raw_out = tavily_snippets.invoke(args)
        results = _normalize_tavily_search_output(raw_out)

        cleaned: List[Dict[str, Any]] = []
        for r in results[:5]:
            cleaned.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "url": (r.get("url") or "").strip(),
                    "content": (r.get("content") or "").strip(),
                    "score": r.get("score"),
                }
            )
        return cleaned

    # -----------------------------
    # 3) Search -> Extract -> Store (가장 안정)
    # -----------------------------
    tavily_for_urls = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=False,
        include_raw_content=False,  # ✅ URL/스니펫만
        search_depth="basic",
    )
    tavily_extract = TavilyExtract(extract_depth="advanced", include_images=False)

    @tool("web_fetch_and_store")
    def web_fetch_and_store(
        query: str,
        topic: str = "general",
        max_results: int = 5,
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[str] = None,
        kind: str = "web",
        dedup: bool = True,
    ) -> Dict[str, Any]:
        """
        ✅ 안정 버전:
        웹 수집/저장 전용 도구.

        Search -> Extract -> Store 2단계로 동작하여,
        Tavily 검색 결과에서 URL을 얻은 뒤 TavilyExtract로 본문(content)을 추출하고,
        그 본문을 ChromaDB에 저장한다.

        중요:
        - 본문(raw_content/content)은 모델에게 반환하지 않는다.
        (모델 컨텍스트 토큰 증가/비용 증가/429 위험을 방지)
        - 모델에게는 저장 결과(저장 개수, 스킵 개수, 사용한 sources 등)만 반환한다.

        입력:
        - query: 검색 질의
        - topic: "general" | "news" | "finance"
        - max_results: URL 후보 개수
        - time_range: "day" | "week" | "month" | "year"
        - include_domains / exclude_domains: 도메인 필터
        - search_depth: "basic" | "advanced"
        - kind: 저장 메타데이터 구분값(기본 "web")
        - dedup: url+content_hash 기반 중복 제거 여부

        출력(dict):
        - stored: 저장된 문서 수
        - skipped: 본문 없음/중복 등으로 스킵된 수
        - kind/query: 기록용 필드
        - sources: (최대 max_results) [{title, url}]
        """
        now = datetime.now(timezone.utc).isoformat()

        # (1) URL 수집
        args: Dict[str, Any] = {"query": query}
        if topic:
            args["topic"] = topic
        if time_range:
            args["time_range"] = time_range
        if include_domains:
            args["include_domains"] = include_domains
        if exclude_domains:
            args["exclude_domains"] = exclude_domains
        if search_depth:
            args["search_depth"] = search_depth

        search_out = tavily_for_urls.invoke(args)
        search_results = _normalize_tavily_search_output(search_out)

        urls: List[str] = []
        sources: List[Dict[str, str]] = []
        for r in search_results[:5]:
            u = (r.get("url") or "").strip()
            if u:
                urls.append(u)
                sources.append({"title": (r.get("title") or "").strip(), "url": u})

        if not urls:
            return {"stored": 0, "skipped": 0, "kind": kind, "query": query, "note": "no_urls_from_search"}

        # (2) Extract로 본문 뽑기
        extract_out = tavily_extract.invoke({"urls": urls})
        # TavilyExtract는 보통 dict 형태로 오며, urls별 content를 포함
        # 버전차를 대비해 최대한 유연하게 파싱
        extracted_items: List[Tuple[str, str]] = []

        if isinstance(extract_out, dict):
            # 예상: {"results": [{"url":..., "content":...}, ...]} 또는 {"url":..., "content":...} 변형
            if isinstance(extract_out.get("results"), list):
                for item in extract_out["results"]:
                    if isinstance(item, dict):
                        u = (item.get("url") or "").strip()
                        c = (item.get("content") or "").strip()
                        if u and c:
                            extracted_items.append((u, c))
            else:
                u = (extract_out.get("url") or "").strip()
                c = (extract_out.get("content") or "").strip()
                if u and c:
                    extracted_items.append((u, c))

        # (3) 저장 (dedup은 url+hash)
        docs: List[Document] = []
        seen_keys = set()
        skipped = 0

        for u, content in extracted_items:
            content_hash = _hash_text(content[:20000])
            key = f"{u}::{content_hash}"
            if dedup and key in seen_keys:
                skipped += 1
                continue
            seen_keys.add(key)

            title = ""
            for s in sources:
                if s["url"] == u:
                    title = s.get("title", "")
                    break

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": u,
                        "title": title,
                        "fetched_at": now,
                        "query": query,
                        "kind": kind,
                        "content_hash": content_hash,
                    },
                )
            )

        if docs:
            vectordb.add_documents(docs)

        stored = len(docs)
        # 추출 실패(0개)일 때도 최소 스니펫 저장 fallback을 하고 싶으면 여기서 추가 가능

        return {"stored": stored, "skipped": skipped, "kind": kind, "query": query, "sources": sources[:max_results]}

    # -----------------------------
    # 4) 호환용 web_search
    # -----------------------------
    @tool("web_search")
    def web_search(
        query: str,
        topic: str = "general",
        max_results: int = 5,
        time_range: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        호환(Backward compatibility)용 웹 검색 도구.

        기존 코드/프롬프트가 web_search(...)를 호출하더라도 깨지지 않도록,
        내부적으로 web_search_snippets(...)를 호출해 동일한 형식의 '짧은 스니펫' 결과를 반환한다.

        주의:
        - 원문 추출/저장은 하지 않는다.
        - 저장이 필요하면 web_fetch_and_store(...)를 별도로 호출해야 한다.
        """
        return web_search_snippets(
            query=query,
            topic=topic,
            max_results=max_results,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            search_depth=search_depth,
        )
    
    @tool("report_write_and_store")
    def report_write_and_store(
        query_used: str,
        sources: List[Dict[str, str]],
        snippets: List[Dict[str, Any]],
        stored: int = 0,
        skipped: int = 0,
        report_kind: str = "web_report",
    ) -> Dict[str, Any]:
        """
        웹 검색 결과(스니펫/링크)를 LLM이 '리포트'로 요약/정리하고,
        원본 링크 목록과 함께 ChromaDB에 1개 문서로 저장한다.

        입력:
        - query_used: 실제 사용한 검색 쿼리
        - sources: [{title, url}]
        - snippets: [{title,url,content,score}]
        - stored/skipped: (선택) 원문 저장 도구 결과를 함께 기록
        출력:
        - stored_report: 1(성공) 또는 0
        - report_id: content_hash
        """

        # LLM (짧게, 구조화)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)

        # 스니펫 텍스트를 너무 길게 보내지 않도록 컷
        lines = []
        for i, s in enumerate(snippets[:8], 1):
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            content = (s.get("content") or "").strip()
            content = content[:800]  # 각 스니펫 길이 제한
            lines.append(f"{i}. {title}\n- url: {url}\n- snippet: {content}")

        prompt = f"""
    너는 조사 리포트를 작성하는 분석가다.
    주제: {query_used}

    아래는 웹 검색 결과의 스니펫이다. 이것만을 근거로:
    - 핵심 사실/이슈를 5~10개 불릿으로 요약
    - 용어/배경이 필요하면 2~4줄 설명
    - 앞으로 더 확인하면 좋을 질문 3개
    - 마지막에 Sources 섹션에 원본 링크를 목록으로 포함

    출력은 마크다운으로 작성.

    [검색 스니펫]
    {chr(10).join(lines)}
    """.strip()

        report_md = llm.invoke(prompt).content

        sources_trim = sources[:10]
        # Sources를 항상 붙여서 저장 (LLM이 빠뜨려도 보장)
        if sources_trim:
            src_lines = ["\n\n## Sources"]
            for s in sources_trim:
                title = (s.get("title") or "").strip()
                url = (s.get("url") or "").strip()
                if url:
                    src_lines.append(f"- [{title or url}]({url})")
            report_md = report_md.rstrip() + "\n" + "\n".join(src_lines) + "\n"

        now = datetime.now(timezone.utc).isoformat()
        report_hash = _hash_text(report_md)

        
        # Chroma에 "리포트 1개 문서" 저장
        doc = Document(
            page_content=report_md,
            metadata={
                "kind": report_kind,
                "query": query_used,
                "created_at": now,
                "sources_json": json.dumps(sources_trim, ensure_ascii=False),  # ✅ 문자열
                "sources_count": len(sources_trim),                            # ✅ int
                "snippets_count": int(len(snippets)),                          # ✅ int
                "raw_stored_count": int(stored),
                "raw_skipped_count": int(skipped),
                "content_hash": report_hash,
            },
        )
        vectordb.add_documents([doc])

        return {"stored_report": 1, "report_id": report_hash}

    return [vector_search, web_search_snippets, web_fetch_and_store, web_search, report_write_and_store]
