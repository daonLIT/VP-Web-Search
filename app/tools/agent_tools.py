from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from hashlib import sha256
import random
from pathlib import Path

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
        max_results=15,
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
        CACHE_PATH = Path(".cache") / "recent_search_urls.json"
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        def _load_recent_urls(limit: int = 200) -> list[str]:
            if not CACHE_PATH.exists():
                return []
            try:
                data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [str(x) for x in data][-limit:]
            except Exception:
                pass
            return []

        def _save_recent_urls(urls: list[str], limit: int = 200) -> None:
            try:
                CACHE_PATH.write_text(
                    json.dumps(urls[-limit:], ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

        def _is_hub_url(u: str) -> bool:
            u = u.lower().rstrip("/")
            bad_contains = ["/tag/", "/tags/", "/topic/", "/topics/"]
            bad_exact_end = ["/news", "/crypto/bitcoin/news", "/symbols/btcusd/news"]
            if any(x in u for x in bad_contains):
                return True
            if any(u.endswith(x) for x in bad_exact_end):
                return True
            # 도메인 단 메인
            if u in {"https://coinness.com", "https://coinness.com/"}:
                return True
            return False
        
        queries = [
            query,
            f"{query} 기관 사칭",
            f"{query} 가족 사칭",
            f"{query} 스미싱 문자 링크",
            f"{query} 앱 설치 유도",
        ]

        all_results = []
        seen = set()

        for q in queries:
            args: Dict[str, Any] = {"query": q}

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

            for r in results:
                url = (r.get("url") or "").strip()
                if not url or _is_hub_url(url):
                    continue
                if url in seen:
                    continue
                seen.add(url)
                all_results.append(r)

        recent_urls = _load_recent_urls()
        recent_set = set(recent_urls)

        # 1) 최근에 쓴 URL은 우선 제외
        fresh_pool = []
        for r in all_results:
            u = (r.get("url") or "").strip()
            if u and (u not in recent_set):
                fresh_pool.append(r)

        # 2) fresh_pool이 너무 적으면(새 결과가 부족하면) 전체 풀로 fallback
        pool = fresh_pool if len(fresh_pool) >= int(max_results) else all_results

        # 3) 실행마다 다른 결과가 나오도록 shuffle 후 max_results 만큼 선택
        random.shuffle(pool)
        picked = pool[: int(max_results)]

        cleaned: List[Dict[str, Any]] = []
        picked_urls: list[str] = []
        for r in picked:
            url = (r.get("url") or "").strip()
            cleaned.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "content": (r.get("content") or "").strip()[:800],
                    "score": r.get("score"),
                }
            )
            if url:
                picked_urls.append(url)

        # 4) 이번에 뽑은 URL을 캐시에 누적 저장(최근 URL 회피용)
        _save_recent_urls(recent_urls + picked_urls)

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
        너는 보이스피싱 최신 수법을 '유형별 지식베이스'로 정리하는 분석가다.
        주제는 반드시: 보이스피싱 최신 수법

        아래 웹 검색 스니펫과 링크를 근거로, 최신 수법을 '유형(type)' 단위로 분류해서
        반드시 아래 JSON 스키마로만 출력하라. (마크다운/설명 금지, JSON만)

        [JSON 스키마]
        {{
        "topic": "보이스피싱 최신 수법",
        "as_of": "{datetime.now(timezone.utc).date().isoformat()}",
        "types": [
            {{
            "type": "유형명(예: 기관 사칭, 가족/지인 사칭, 대출 사기, 택배/문자 링크, 몸캠/협박, 알바/구인, 중고거래, 투자/코인 등)",
            "keywords": ["주요 키워드1","키워드2",],
            "scenario": ["1) 단계별 시나리오", "2) ...", "3) ...", ...],
            "red_flags": ["의심 신호 1", "의심 신호 2",],
            "recommended_actions": ["대응 1", "대응 2",]
            }}
        ],
        "sources": [
            {{"title":"...","url":"..."}}
        ]
        }}

        규칙:
        - types는 가능한 한 5~12개 사이로 뽑아라.
        - scenario는 반드시 5~7 단계의 리스트로 써라.
        - sources는 제공된 링크만 사용하라.
        - 스니펫 근거가 약하면 type은 넣되 scenario/keywords를 보수적으로 작성하라.

        [검색 스니펫]
        {chr(10).join(lines)}
        """.strip()

        report_md = llm.invoke(prompt).content
        report_json_str = llm.invoke(prompt).content.strip()

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
        report_hash = _hash_text(report_json_str)

        
        # Chroma에 "리포트 1개 문서" 저장
        doc = Document(
            page_content=report_json_str,   # ✅ JSON을 그대로 저장
            metadata={
                "kind": "voicephishing_types_v1",
                "query": "보이스피싱 최신 수법",
                "created_at": now,
                "sources_count": int(len(sources_trim)),
                "snippets_count": int(len(snippets or [])),
                "content_hash": report_hash,
            },
        )
        vectordb.add_documents([doc])

        return {"stored_report": 1, "report_id": report_hash, "kind": "voicephishing_types_v1"}
    
    @tool("store_snippets_only")
    def store_snippets_only(
        query_used: str,
        snippets: List[Dict[str, Any]],
        kind: str = "voicephishing_snippet_v1",
    ) -> Dict[str, Any]:
        """
        LLM 없이 웹 검색 스니펫(title/url/content)만 ChromaDB에 저장한다.
        - 기사 1개 = 문서 1개
        - metadata는 Chroma 제약(원시타입)만 사용한다.
        """
        now = datetime.now(timezone.utc).isoformat()

        stored = 0
        skipped = 0
        docs: List[Document] = []

        for s in (snippets or []):
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            content = (s.get("content") or "").strip()
            snippet_id = _hash_text(url)  # ✅ URL 기반 고유 ID

            if not url:
                skipped += 1
                continue

            # 너무 길면 자름 (저장용)
            content = content[:600]

            payload = {
                "topic": "보이스피싱 최신 수법",
                "query_used": query_used,
                "article": {"title": title, "url": url},
                "snippet": content,
                "created_at": now,
                "snippet_id": snippet_id,
            }
            page_content = json.dumps(payload, ensure_ascii=False)

            content_hash = _hash_text(url + "|" + content)

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "kind": kind,
                        "query": query_used,
                        "title": title[:200],
                        "url": url,
                        "created_at": now,
                        "content_hash": content_hash,
                        # ✅ 추가
                        "snippet_id": snippet_id,
                        "processed": False,
                        "used_in_report_id": "", 
                    },
                )
            )
            stored += 1

        if docs:
            vectordb.add_documents(docs)

        return {"stored": stored, "skipped": skipped, "kind": kind}
    

    # =========================
    # 1) LOAD: 수집된 스니펫 로드
    # =========================
    @tool("load_collected_snippets")
    def load_collected_snippets(
        limit: int = 5,
        kind: str = "voicephishing_snippet_v1",
        only_unprocessed: bool = True,
    ) -> Dict[str, Any]:
        """
        ChromaDB에 저장된 수집(snippet) 문서를 가져온다.
        요약/리포트 단계에서 LLM 입력으로 사용한다.

        반환:
        {
        "count": N,
        "items": [
            {
            "doc_id": "...",        # Chroma 내부 문서 ID
            "snippet_id": "...",    # URL 기반 고유 ID(없으면 None)
            "title": "...",
            "url": "...",
            "created_at": "...",
            "payload_json": "..."   # store_snippets_only가 저장한 page_content(JSON 문자열)
            }, ...
        ]
        }
        """
        # langchain_chroma는 내부에 _collection(Chroma Collection)을 들고 있음
        col = vectordb._collection  # private지만 실무에서 많이 씀

        if only_unprocessed:
            where = {
                "$and": [
                    {"kind": {"$eq": kind}},
                    {"processed": {"$eq": False}},
                ]
            }
        else:
            where = {"kind": {"$eq": kind}}

        data = col.get(where=where, limit=int(limit), include=["documents", "metadatas"])

        items: List[Dict[str, Any]] = []
        for doc_id, content, meta in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
            items.append(
                {
                    "doc_id": doc_id,  # chroma 내부 id (있으면 유용)
                    "snippet_id": meta.get("snippet_id"),
                    "title": meta.get("title"),
                    "url": meta.get("url"),
                    "created_at": meta.get("created_at"),
                    "payload_json": content,  # store_snippets_only가 넣은 JSON 문자열
                }
            )

        return {"count": len(items), "items": items}
    
    # ==========================================
    # 2) WRITE+STORE: 스니펫들 -> 리포트 저장
    # ==========================================
    @tool("write_report_from_snippets_and_store")
    def write_report_from_snippets_and_store(
        query_used: str,
        snippet_items: List[Dict[str, Any]],
        report_kind: str = "voicephishing_report_v1",
    ) -> Dict[str, Any]:
        """
        수집된 snippet 여러 개를 기반으로 LLM이 요약 리포트를 작성하고 ChromaDB에 저장한다.
        리포트는 수집 문서들과 연결될 수 있도록 source_snippet_ids_json을 metadata에 저장한다.

        반환:
        {"stored_report": 1, "report_id": "...", "source_count": N}
        """
        # LLM 1회만
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=20, max_retries=1)

        # LLM 입력용 요약: payload_json에서 snippet만 뽑아 짧게 구성
        normalized: List[Dict[str, Any]] = []
        source_doc_ids: List[str] = []
        source_snippet_ids: List[str] = []

        for it in snippet_items:
            doc_id = (it.get("doc_id") or "").strip()
            source_doc_ids.append(doc_id)

            sid = it.get("snippet_id") or ""
            if not sid:
                # snippet_id가 없던 구 데이터 대비: url로 만들어줌
                # (가능하면 수집 단계에서 snippet_id를 항상 넣도록 권장)
                try:
                    payload_tmp = json.loads(it.get("payload_json") or "{}")
                    url_tmp = ((payload_tmp.get("article") or {}).get("url") or it.get("url") or "").strip()
                except Exception:
                    url_tmp = (it.get("url") or "").strip()
                sid = _hash_text(url_tmp) if url_tmp else _hash_text(doc_id or json.dumps(it, ensure_ascii=False))
            source_snippet_ids.append(sid)

            try:
                payload = json.loads(it.get("payload_json") or "{}")
            except Exception:
                payload = {}

            title = ((payload.get("article") or {}).get("title") or it.get("title") or "").strip()
            url = ((payload.get("article") or {}).get("url") or it.get("url") or "").strip()
            snippet = (payload.get("snippet") or "").strip()

            normalized.append(
                {
                    "snippet_id": sid,
                    "title": str(title)[:160],
                    "url": str(url),
                    "snippet": str(snippet)[:700],
                }
            )

        if not normalized:
            return {"stored_report": 0, "report_id": None, "source_count": 0, "reason": "no_snippets"}

        now = datetime.now(timezone.utc).isoformat()

        prompt = f"""
    너는 보이스피싱 최신 수법을 분석해 '리포트'로 정리하는 분석가다.
    입력은 여러 개의 뉴스 스니펫이며, 각 스니펫에는 snippet_id가 있다.

    출력 형식(반드시 지켜라):
    - 유형 단위로 섹션을 나누어 작성:
    유형: ...
    주요 키워드: ... (여러 개)
    시나리오:
        1. ...
        2. ...
        3. ...
    근거 snippet_id: ["...","..."]  (이 유형을 뒷받침하는 snippet_id들을 반드시 포함)

    - 마지막에는 아래 예시처럼 "종합 분석 문단" 1개를 작성하라:
    (예시 톤) "피해자는 권위와 전문성을 인지하여 ..."

    규칙:
    - 스니펫에서 확인 가능한 정보만 사용하고 과장/창작 금지
    - "근거 snippet_id"는 반드시 JSON 배열 형태로 표기
    - 전체 출력은 한국어 텍스트(마크다운 허용), 코드펜스 금지

    [입력 스니펫들]
    {json.dumps(normalized, ensure_ascii=False)}
    """.strip()

        report_text = llm.invoke(prompt).content.strip()

        now = datetime.now(timezone.utc).isoformat()
        report_id = _hash_text(report_text + "|" + now)

        doc = Document(
            page_content=report_text,
            metadata={
                "kind": report_kind,
                "query": query_used,
                "created_at": now,
                "report_id": report_id,
                # Chroma metadata 제약 때문에 JSON 문자열로 저장
                "source_snippet_ids_json": json.dumps(source_snippet_ids, ensure_ascii=False),
                "source_doc_ids_json": json.dumps(source_doc_ids, ensure_ascii=False),
                "source_count": int(len(source_snippet_ids)),
            },
        )
        vectordb.add_documents([doc])

        return {"stored_report": 1, "report_id": report_id, "source_count": len(source_snippet_ids)}
    
    @tool("mark_snippets_processed")
    def mark_snippets_processed(
        doc_ids: List[str],
        report_id: str,
        kind: str = "voicephishing_snippet_v1",
    ) -> Dict[str, Any]:
        """
        수집(snippet) 문서를 processed=True로 업데이트하고,
        어떤 report_id에서 사용했는지 report_id도 기록한다.

        doc_ids는 load_collected_snippets가 돌려준 items[*].doc_id 리스트를 넣는다.
        """

        col = vectordb._collection

        # Chroma update는 ids 기준으로 가능
        # metadata는 전체를 덮어쓸 수 있으니, 기존 metadata를 먼저 가져와 병합하는 방식이 안전
        data = col.get(ids=doc_ids, include=["metadatas"])
        old_metas = data.get("metadatas", []) or []

        new_metas: List[Dict[str, Any]] = []
        for meta in old_metas:
            meta = dict(meta or {})
            meta["kind"] = kind
            meta["processed"] = True
            meta["used_in_report_id"] = str(report_id or "")
            meta["processed_at"] = datetime.now(timezone.utc).isoformat()
            new_metas.append(meta)

        col.update(ids=doc_ids, metadatas=new_metas)
        return {"updated": len(doc_ids), "report_id": report_id}

    return [vector_search, web_search_snippets, web_fetch_and_store, web_search, report_write_and_store, store_snippets_only, load_collected_snippets, write_report_from_snippets_and_store, mark_snippets_processed]
