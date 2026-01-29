from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from hashlib import sha256
import random
from pathlib import Path

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Any, Optional
import time

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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
    @tool("search_existing_guidance")
    def search_existing_guidance(
        phishing_type: str,
        scenario_hint: str = "",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        DB에 저장된 보이스피싱 리포트/스니펫에서 특정 유형의 지침을 검색한다.
        
        입력:
        - phishing_type: 보이스피싱 유형 (예: "검경 사칭", "기관 사칭", "가족 사칭")
        - scenario_hint: 시나리오 힌트 (예: "검찰 사칭해서 현금 편취")
        - top_k: 반환할 최대 결과 수
        
        출력:
        {
            "found": bool,
            "count": int,
            "guidances": [
                {
                    "type": str,
                    "keywords": [str],
                    "scenario": [str],
                    "red_flags": [str],
                    "recommended_actions": [str],
                    "source_id": str,
                    "relevance_score": float
                }
            ]
        }
        """
        col = vectordb._collection
        
        # 검색 쿼리 구성
        search_query = f"{phishing_type} {scenario_hint}".strip()
        
        # 1) 리포트 kind에서 검색
        report_where = {"kind": {"$eq": "voicephishing_report_v1"}}
        report_data = col.get(where=report_where, limit=50, include=["documents", "metadatas"])
        
        # 2) 유사도 검색 (벡터 검색)
        vector_results = vectordb.similarity_search_with_relevance_scores(
            search_query, 
            k=top_k * 2,
            filter={"kind": "voicephishing_report_v1"}
        )
        
        guidances = []
        seen_ids = set()
        
        # 벡터 검색 결과 우선 처리
        for doc, score in vector_results:
            content = doc.page_content
            meta = doc.metadata
            
            # 리포트 텍스트 파싱 (유형별 섹션 추출)
            try:
                # 리포트가 구조화된 텍스트라고 가정
                type_match = _extract_type_from_report(content, phishing_type)
                if type_match and type_match["type"] not in seen_ids:
                    seen_ids.add(type_match["type"])
                    type_match["source_id"] = meta.get("report_id", "")
                    type_match["relevance_score"] = float(score)
                    guidances.append(type_match)
                    
                    if len(guidances) >= top_k:
                        break
            except Exception:
                continue
        
        return {
            "found": len(guidances) > 0,
            "count": len(guidances),
            "guidances": guidances
        }


    def _extract_type_from_report(report_text: str, target_type: str) -> Optional[Dict[str, Any]]:
        """
        리포트 텍스트에서 특정 유형의 정보를 추출한다.
        """
        import re
        
        # 유형 섹션 찾기 (예: "유형: 검경 사칭")
        type_pattern = rf"유형:\s*([^\n]+)"
        type_matches = list(re.finditer(type_pattern, report_text))
        
        for match in type_matches:
            found_type = match.group(1).strip()
            
            # 유형이 일치하는지 확인 (부분 일치 허용)
            if target_type.lower() in found_type.lower() or found_type.lower() in target_type.lower():
                # 해당 유형 섹션 추출
                section_start = match.start()
                
                # 다음 "유형:" 또는 문서 끝까지
                next_match = None
                for m in type_matches:
                    if m.start() > section_start:
                        next_match = m
                        break
                
                section_end = next_match.start() if next_match else len(report_text)
                section = report_text[section_start:section_end]
                
                # 섹션에서 정보 추출
                keywords = _extract_field(section, r"주요 키워드:\s*([^\n]+)")
                scenario = _extract_scenario(section)
                red_flags = _extract_list_field(section, r"의심 신호|근거 snippet_id")
                
                return {
                    "type": found_type,
                    "keywords": keywords,
                    "scenario": scenario,
                    "red_flags": red_flags,
                    "recommended_actions": []  # 리포트에 따라 추가
                }
        
        return None


    def _extract_field(text: str, pattern: str) -> List[str]:
        """정규식으로 필드 추출 후 리스트로 변환"""
        import re
        match = re.search(pattern, text)
        if match:
            content = match.group(1).strip()
            # 쉼표나 공백으로 구분
            return [k.strip() for k in re.split(r'[,，、]', content) if k.strip()]
        return []


    def _extract_scenario(text: str) -> List[str]:
        """시나리오 단계 추출"""
        import re
        scenario_pattern = r"시나리오:\s*((?:\d+[\.\)]\s*[^\n]+\n?)+)"
        match = re.search(scenario_pattern, text)
        if match:
            steps = match.group(1).strip().split('\n')
            return [re.sub(r'^\d+[\.\)]\s*', '', s).strip() for s in steps if s.strip()]
        return []


    def _extract_list_field(text: str, header_pattern: str) -> List[str]:
        """리스트 형태 필드 추출"""
        import re
        pattern = rf"{header_pattern}:?\s*((?:[-\*•]\s*[^\n]+\n?)+)"
        match = re.search(pattern, text)
        if match:
            items = match.group(1).strip().split('\n')
            return [re.sub(r'^[-\*•]\s*', '', item).strip() for item in items if item.strip()]
        return []


    @tool("generate_targeted_guidance")
    def generate_targeted_guidance(
        phishing_type: str,
        scenario: str,
        victim_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        특정 유형과 시나리오에 맞춘 웹 검색을 수행하고,
        피해자 프로필을 고려한 맞춤형 지침을 생성한다.
        
        입력:
        - phishing_type: 보이스피싱 유형
        - scenario: 시나리오 설명
        - victim_profile: 피해자 특성 (선택)
        
        출력:
        {
            "type": str,
            "keywords": [str],
            "scenario": [str],
            "red_flags": [str],
            "recommended_actions": [str],
            "sources": [{title, url}]
        }
        """
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
        
        # 검색 쿼리 구성
        base_queries = [
            f"보이스피싱 {phishing_type} 수법",
            f"{phishing_type} {scenario}",
            f"{phishing_type} 시나리오",
        ]
        
        # 피해자 프로필 기반 추가 키워드
        if victim_profile:
            age = victim_profile.get("age")
            occupation = victim_profile.get("occupation")
            if age:
                base_queries.append(f"{phishing_type} {age}대 피해")
            if occupation:
                base_queries.append(f"{phishing_type} {occupation} 대상")
        
        # 웹 검색 수행
        all_snippets = []
        all_sources = []
        
        for query in base_queries[:3]:  # 최대 3개 쿼리
            args = {
                "query": query,
                "topic": "news",
                "max_results": 3,
                "time_range": "month",
            }
            
            raw_out = tavily_snippets.invoke(args)
            results = _normalize_tavily_search_output(raw_out)
            
            for r in results:
                url = (r.get("url") or "").strip()
                if url and not _is_hub_url(url):
                    all_snippets.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": (r.get("content") or "")[:600],
                    })
                    all_sources.append({"title": r.get("title", ""), "url": url})
        
        # LLM으로 지침 생성
        snippet_text = "\n\n".join([
            f"출처: {s['title']}\nURL: {s['url']}\n내용: {s['content']}"
            for s in all_snippets[:8]
        ])
        
        victim_context = ""
        if victim_profile:
            victim_context = f"\n피해자 특성: {json.dumps(victim_profile, ensure_ascii=False)}"
        
        prompt = f"""
    너는 보이스피싱 수법 분석 전문가다.
    아래 웹 검색 결과를 기반으로 '{phishing_type}' 유형의 상세 지침을 생성하라.

    시나리오 힌트: {scenario}{victim_context}

    출력 형식 (반드시 JSON):
    {{
    "type": "{phishing_type}",
    "keywords": ["키워드1", "키워드2", ...],
    "scenario": [
        "1단계: ...",
        "2단계: ...",
        "3단계: ...",
        ...
    ],
    "red_flags": ["의심 신호1", "의심 신호2", ...],
    "recommended_actions": ["대응법1", "대응법2", ...]
    }}

    규칙:
    - scenario는 5~7단계로 구체적으로 작성
    - 검색 결과에 근거한 내용만 포함
    - 피해자 특성을 고려한 맞춤형 내용 작성

    [검색 결과]
    {snippet_text}
    """.strip()
        
        response = llm.invoke(prompt).content.strip()
        
        # JSON 파싱
        try:
            # 코드 블록 제거
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            guidance = json.loads(response)
            guidance["sources"] = all_sources[:5]
            
            return guidance
        except Exception as e:
            # 파싱 실패 시 기본 구조 반환
            return {
                "type": phishing_type,
                "keywords": [phishing_type, scenario],
                "scenario": [scenario],
                "red_flags": [],
                "recommended_actions": [],
                "sources": all_sources[:5],
                "error": str(e)
            }


    @tool("store_guidance_to_db")
    def store_guidance_to_db(
        guidance: Dict[str, Any],
        source_system: str = "external_request",
    ) -> Dict[str, Any]:
        """
        생성된 지침을 DB에 저장한다.
        
        입력:
        - guidance: generate_targeted_guidance 출력
        - source_system: 요청 출처
        
        출력:
        {"stored": 1, "guidance_id": "..."}
        """
        now = datetime.now(timezone.utc).isoformat()
        
        # JSON 문자열로 저장
        content = json.dumps(guidance, ensure_ascii=False)
        guidance_id = _hash_text(content)
        
        doc = Document(
            page_content=content,
            metadata={
                "kind": "voicephishing_guidance_v1",
                "phishing_type": guidance.get("type", ""),
                "source_system": source_system,
                "created_at": now,
                "guidance_id": guidance_id,
            }
        )
        
        vectordb.add_documents([doc])
        
        return {"stored": 1, "guidance_id": guidance_id}
    
    @tool("crawl_site_for_phishing_cases")
    def crawl_site_for_phishing_cases(
        site_url: str,
        keywords: List[str] = None,
        max_articles: int = 10,
        article_selector: Optional[str] = None,
        title_selector: Optional[str] = None,
        link_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        특정 사이트의 목록 페이지에서 보이스피싱 관련 글을 필터링하고 링크를 수집한다.
        
        입력:
        - site_url: 크롤링할 사이트 URL (예: 경찰청 공지사항 목록 페이지)
        - keywords: 필터링 키워드 (기본: ["보이스피싱", "전화금융사기", "스미싱", "피싱"])
        - max_articles: 최대 수집 글 수
        - article_selector: 글 목록 CSS 셀렉터 (선택, 자동 감지 시도)
        - title_selector: 제목 CSS 셀렉터 (선택)
        - link_selector: 링크 CSS 셀렉터 (선택)
        
        출력:
        {
            "site_url": str,
            "found_count": int,
            "articles": [
                {"title": str, "url": str, "matched_keywords": [str]},
                ...
            ]
        }
        """
        if keywords is None:
            keywords = [
                "보이스피싱", "전화금융사기", "스미싱", "피싱", 
                "메신저피싱", "사기", "금융사기", "텔레그램"
            ]
        
        try:
            # User-Agent 설정 (차단 방지)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(site_url, headers=headers, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 자동 셀렉터 감지 또는 지정된 셀렉터 사용
            articles = []
            
            if article_selector:
                # 사용자 지정 셀렉터
                items = soup.select(article_selector)
            else:
                # 자동 감지: 일반적인 게시판 패턴들
                items = (
                    soup.select('tr') or  # 테이블 기반
                    soup.select('li') or  # 리스트 기반
                    soup.select('article') or
                    soup.select('.board-list tr') or
                    soup.select('.notice-list li')
                )
            
            filtered_articles = []
            
            for item in items[:100]:  # 최대 100개까지만 탐색
                # 제목 추출
                if title_selector:
                    title_elem = item.select_one(title_selector)
                else:
                    # 자동 감지
                    title_elem = (
                        item.select_one('td.title') or
                        item.select_one('.title') or
                        item.select_one('a') or
                        item.select_one('h3') or
                        item.select_one('h4')
                    )
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                
                # 키워드 필터링
                matched_keywords = [kw for kw in keywords if kw in title]
                if not matched_keywords:
                    continue
                
                # 링크 추출
                if link_selector:
                    link_elem = item.select_one(link_selector)
                else:
                    # 자동 감지
                    link_elem = title_elem if title_elem.name == 'a' else item.select_one('a')
                
                if not link_elem:
                    continue
                
                href = link_elem.get('href', '')
                if not href:
                    continue
                
                # 상대 URL → 절대 URL 변환
                full_url = urljoin(site_url, href)
                
                filtered_articles.append({
                    "title": title,
                    "url": full_url,
                    "matched_keywords": matched_keywords
                })
                
                if len(filtered_articles) >= max_articles:
                    break
            
            return {
                "site_url": site_url,
                "found_count": len(filtered_articles),
                "articles": filtered_articles
            }
        
        except Exception as e:
            return {
                "site_url": site_url,
                "found_count": 0,
                "articles": [],
                "error": str(e)
            }


    @tool("extract_article_content")
    def extract_article_content(
        article_url: str,
        content_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        개별 글의 본문 내용을 추출한다.
        
        입력:
        - article_url: 글 상세 페이지 URL
        - content_selector: 본문 CSS 셀렉터 (선택, 자동 감지 시도)
        
        출력:
        {
            "url": str,
            "title": str,
            "content": str,
            "extracted": bool
        }
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(article_url, headers=headers, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 제목 추출
            title_elem = (
                soup.select_one('h1') or
                soup.select_one('h2.title') or
                soup.select_one('.subject') or
                soup.select_one('.post-title')
            )
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # 본문 추출
            if content_selector:
                content_elem = soup.select_one(content_selector)
            else:
                # 자동 감지: 일반적인 본문 패턴들
                content_elem = (
                    soup.select_one('div.content') or
                    soup.select_one('div.post-content') or
                    soup.select_one('div.article-body') or
                    soup.select_one('div#content') or
                    soup.select_one('article') or
                    soup.select_one('.view-content') or
                    soup.select_one('.board-view')
                )
            
            if not content_elem:
                # fallback: body에서 script/style 제거 후 추출
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                content_elem = soup.select_one('body')
            
            # 텍스트 추출 및 정제
            content = content_elem.get_text(separator='\n', strip=True) if content_elem else ""
            
            # 과도한 공백 제거
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            return {
                "url": article_url,
                "title": title,
                "content": content[:5000],  # 최대 5000자
                "extracted": bool(content)
            }
        
        except Exception as e:
            return {
                "url": article_url,
                "title": "",
                "content": "",
                "extracted": False,
                "error": str(e)
            }


    @tool("crawl_and_extract_batch")
    def crawl_and_extract_batch(
        site_url: str,
        keywords: List[str] = None,
        max_articles: int = 10,
        delay_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """
        사이트 크롤링 + 본문 추출을 한번에 처리한다.
        
        입력:
        - site_url: 목록 페이지 URL
        - keywords: 필터링 키워드
        - max_articles: 최대 수집 글 수
        - delay_seconds: 요청 간 지연 시간 (서버 부하 방지)
        
        출력:
        {
            "site_url": str,
            "crawled_count": int,
            "extracted_count": int,
            "articles": [
                {
                    "title": str,
                    "url": str,
                    "content": str,
                    "matched_keywords": [str]
                },
                ...
            ]
        }
        """
        # 1단계: 목록에서 관련 글 수집
        crawl_result = crawl_site_for_phishing_cases.invoke({
            "site_url": site_url,
            "keywords": keywords,
            "max_articles": max_articles
        })
        
        if crawl_result.get("found_count", 0) == 0:
            return {
                "site_url": site_url,
                "crawled_count": 0,
                "extracted_count": 0,
                "articles": [],
                "note": "no_articles_found"
            }
        
        # 2단계: 각 글의 본문 추출
        articles_with_content = []
        
        for article in crawl_result.get("articles", []):
            # 서버 부하 방지를 위한 지연
            time.sleep(delay_seconds)
            
            extract_result = extract_article_content.invoke({"article_url": article["url"]})
            
            if extract_result.get("extracted"):
                articles_with_content.append({
                    "title": article["title"],
                    "url": article["url"],
                    "content": extract_result["content"],
                    "matched_keywords": article.get("matched_keywords", [])
                })
        
        return {
            "site_url": site_url,
            "crawled_count": crawl_result.get("found_count", 0),
            "extracted_count": len(articles_with_content),
            "articles": articles_with_content
        }


    @tool("generate_guidance_from_crawled_articles")
    def generate_guidance_from_crawled_articles(
        articles: List[Dict[str, Any]],
        target_type: Optional[str] = None,
        force_generate: bool = True,
    ) -> Dict[str, Any]:
        """
        크롤링한 글들로부터 보이스피싱 지침을 생성한다.
        force_generate=True면 무조건 최소 1개 유형 생성
        
        입력:
        - articles: crawl_and_extract_batch의 articles 결과
        - target_type: 특정 유형으로 한정 (선택)
        
        출력:
        {
            "guidance": {...},
            "source_articles": [{title, url}, ...]
        }
        """
        if not articles:
            return {
                "guidance": {"types": []},
                "source_articles": [],
                "note": "no_articles_provided"
            }

        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=40)
        
        # 글 내용 요약
        article_summaries = []
        for i, article in enumerate(articles[:15], 1):
            title = article.get("title", "")
            content = article.get("content", "")[:1500]  # 각 글당 1500자 제한
            
            article_summaries.append(
                f"{i}. 제목: {title}\n내용: {content}\n"
            )
        
        articles_text = "\n---\n".join(article_summaries)
        
        type_instruction = f"특히 '{target_type}' 유형에 집중하라." if target_type else ""
        
        prompt = f"""
    너는 보이스피싱 수법 분석 전문가다.
    아래는 공식 기관에서 크롤링한 보이스피싱 관련 글들이다.

    중요 지침:
    1. 글이 직접적인 사례가 아니더라도, 언급된 수법/패턴을 추출하라
    2. "예방", "주의", "조심" 등의 맥락에서 나온 수법 설명도 포함
    3. 최소 1개 이상의 유형은 반드시 추출하라
    4. 구체적 사례가 없으면 일반적인 패턴이라도 정리하라

    {type_instruction}

    출력 형식 (JSON만, 코드블록 없이):
    {{
    "types": [
        {{
        "type": "유형명 (예: 기관 사칭, 가족 사칭, 대출 사기, AI 음성 사칭 등)",
        "keywords": ["핵심키워드1", "핵심키워드2", ...],
        "scenario": [
            "1단계: 초기 접근 (어떻게 연락하는가)",
            "2단계: 신뢰 구축 (어떻게 믿게 만드는가)",
            "3단계: 정보 획득 (무엇을 요구하는가)",
            "4단계: 압박 전술 (어떻게 급박하게 만드는가)",
            "5단계: 금전 요구 (돈을 어떻게 빼가는가)"
        ],
        "red_flags": [
            "의심할 수 있는 신호 1",
            "의심할 수 있는 신호 2",
            ...
        ],
        "recommended_actions": [
            "즉시 취할 행동 1",
            "즉시 취할 행동 2",
            ...
        ],
        "real_cases": [
            "글에서 언급된 사례나 패턴 요약 1",
            "글에서 언급된 사례나 패턴 요약 2"
        ]
        }}
    ]
    }}

    규칙:
    1. types는 최소 1개, 최대 5개
    2. scenario는 정확히 5단계 (부족하면 일반적 패턴으로 채워라)
    3. real_cases가 없으면 글에서 언급된 예방법/주의사항이라도 요약
    4. 글이 보도자료/공지라도 그 안에서 수법 정보 추출

    예시 해석:
    - "AI 음성으로 보이스피싱 예방" → AI 음성 사칭 수법이 있다는 의미
    - "가족 사칭 주의" → 가족 사칭 유형 추출
    - "계좌 이체 요구 조심" → 금전 요구 단계에 포함

    [크롤링한 글들]
    {articles_text}
    """.strip()
        
        # LLM 호출 전 로깅
        print(f"\n📊 LLM에 전달할 내용:")
        print(f"   - 글 개수: {len(articles)}")
        print(f"   - 총 텍스트 길이: {len(articles_text)} 자")
        print(f"   - 첫 번째 글 미리보기: {articles[0].get('title', '')[:50]}...")
        
        try:
            response = llm.invoke(prompt).content.strip()

            # 응답 로깅
            print(f"\n🤖 LLM 응답 미리보기:")
            print(response[:300] + "...")
            
            # JSON 추출
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            guidance_data = json.loads(response)

            # types 검증
            types_list = guidance_data.get("types", [])

            # types가 비어있으면 기본 템플릿
            if force_generate and len(types_list) == 0:
                print("⚠️  LLM이 유형을 생성하지 않음 → 강제 생성")
                
                # 글 제목에서 키워드 추출
                all_titles = " ".join([a.get("title", "") for a in articles])
                extracted_keywords = []
                
                keyword_patterns = [
                    "보이스피싱", "스미싱", "피싱", "메신저", "AI", "음성",
                    "가족", "검찰", "경찰", "금융", "은행", "대출", "투자"
                ]
                
                for kw in keyword_patterns:
                    if kw in all_titles:
                        extracted_keywords.append(kw)
                # 최소한 언급된 키워드로 1개 유형 생성
                fallback_type = {
                    "type": target_type or "보이스피싱 일반",
                    "keywords": extracted_keywords[:5] or ["보이스피싱"],
                    "scenario": [
                        "1단계: 공공기관/금융기관 사칭 전화",
                        "2단계: 범죄 연루/계좌 문제 등 위기 조성",
                        "3단계: 개인정보 요구",
                        "4단계: 즉시 조치 압박",
                        "5단계: 계좌 이체 또는 앱 설치 유도"
                    ],
                    "red_flags": [
                        "출처 불명 전화/문자",
                        "긴급 상황 강조",
                        "개인정보/금융정보 요구"
                    ],
                    "recommended_actions": [
                        "통화 즉시 종료",
                        "경찰청 182 신고",
                        "공식 기관 번호로 재확인"
                    ],
                    "real_cases": [
                        f"크롤링한 {len(articles)}개 글에서 언급된 예방법 기반"
                    ]
                }
                
                guidance_data["types"] = [fallback_type]
            
            # 출처 정보 추가
            source_articles = [
                {"title": a.get("title", ""), "url": a.get("url", "")}
                for a in articles[:10]
            ]
            
            return {
                "guidance": guidance_data,
                "source_articles": source_articles
            }
        
        except Exception as e:
            print(f"⚠️  LLM 응답 파싱 실패: {str(e)}")
        
            fallback_guidance = {
                "types": [{
                    "type": "보이스피싱 일반",
                    "keywords": ["보이스피싱", "전화금융사기"],
                    "scenario": [
                        "1단계: 공공기관 사칭",
                        "2단계: 위기 상황 조성",
                        "3단계: 개인정보 요구",
                        "4단계: 즉시 조치 압박",
                        "5단계: 금전 요구"
                    ],
                    "red_flags": ["출처 불명 연락", "긴급 상황 강조"],
                    "recommended_actions": ["통화 종료", "경찰 신고"],
                    "real_cases": [f"{len(articles)}개 글 기반"]
                }]
            }
            
            return {
                "guidance": fallback_guidance,
                "source_articles": [
                    {"title": a.get("title", ""), "url": a.get("url", "")}
                    for a in articles[:5]
                ],
                "error": str(e)
            }

    @tool("store_crawled_guidance")
    def store_crawled_guidance(
        guidance_data: Dict[str, Any],
        site_url: str,
        source_articles: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        크롤링으로 생성한 지침을 DB에 저장한다.
        
        입력:
        - guidance_data: generate_guidance_from_crawled_articles의 guidance
        - site_url: 크롤링한 사이트 URL
        - source_articles: 출처 글 목록
        
        출력:
        {"stored": int, "guidance_ids": [str]}
        """
        now = datetime.now(timezone.utc).isoformat()
        stored_ids = []
        
        types_list = guidance_data.get("types", [])
        
        for type_info in types_list:
            content = json.dumps(type_info, ensure_ascii=False)
            guidance_id = _hash_text(content + now)
            
            doc = Document(
                page_content=content,
                metadata={
                    "kind": "voicephishing_guidance_crawled_v1",
                    "phishing_type": type_info.get("type", ""),
                    "source_site": site_url,
                    "source_articles_json": json.dumps(source_articles, ensure_ascii=False),
                    "created_at": now,
                    "guidance_id": guidance_id,
                }
            )
            
            vectordb.add_documents([doc])
            stored_ids.append(guidance_id)
        
        return {
            "stored": len(stored_ids),
            "guidance_ids": stored_ids
        }
    
    @tool("crawl_site_with_pagination")
    def crawl_site_with_pagination(
        site_url: str,
        keywords: List[str] = None,
        max_articles: int = 30,
        max_pages: int = 5,
        pagination_type: str = "auto",  # auto | url_param | path | next_button
        page_param: str = "page",  # URL 파라미터 이름
        delay_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        """
        여러 페이지를 순회하며 보이스피싱 관련 글을 수집한다.
        
        입력:
        - site_url: 첫 페이지 URL
        - keywords: 필터링 키워드
        - max_articles: 최대 수집 글 수
        - max_pages: 최대 탐색 페이지 수
        - pagination_type: 페이지 넘김 방식
            * auto: 자동 감지 (URL 패턴 분석)
            * url_param: ?page=N 형태
            * path: /notice/N 형태
            * next_button: "다음" 링크 찾기
        - page_param: pagination_type=url_param일 때 사용할 파라미터명
        - delay_seconds: 페이지 간 지연 시간
        
        출력:
        {
            "site_url": str,
            "pages_crawled": int,
            "found_count": int,
            "articles": [{"title": str, "url": str, "matched_keywords": [str]}, ...]
        }
        """
        if keywords is None:
            keywords = [
                "보이스피싱", "전화금융사기", "스미싱", "피싱",
                "메신저피싱", "사기", "금융사기", "텔레그램"
            ]
        
        all_articles = []
        current_url = site_url
        pages_crawled = 0
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 페이지네이션 타입 자동 감지
        if pagination_type == "auto":
            parsed = urlparse(site_url)
            if f'{page_param}=' in parsed.query:
                pagination_type = "url_param"
            elif re.search(r'/\d+/?$', parsed.path):
                pagination_type = "path"
            else:
                pagination_type = "next_button"
        
        for page_num in range(1, max_pages + 1):
            try:
                print(f"📄 크롤링 중: 페이지 {page_num}/{max_pages}")
                
                # 페이지 URL 생성
                if pagination_type == "url_param":
                    # ?page=N 방식
                    parsed = urlparse(site_url)
                    query_params = parse_qs(parsed.query)
                    query_params[page_param] = [str(page_num)]
                    new_query = urlencode(query_params, doseq=True)
                    current_url = urlunparse(parsed._replace(query=new_query))
                    
                elif pagination_type == "path":
                    # /notice/N 방식
                    base_url = re.sub(r'/\d+/?$', '', site_url)
                    current_url = f"{base_url}/{page_num}"
                
                # 페이지 요청
                response = requests.get(current_url, headers=headers, timeout=15)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 글 목록 추출 (기존 로직 재사용)
                items = (
                    soup.select('tr') or
                    soup.select('li') or
                    soup.select('article') or
                    soup.select('.board-list tr') or
                    soup.select('.notice-list li')
                )
                
                page_articles = []
                
                for item in items:
                    # 제목 추출
                    title_elem = (
                        item.select_one('td.title') or
                        item.select_one('.title') or
                        item.select_one('a') or
                        item.select_one('h3') or
                        item.select_one('h4')
                    )
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # 키워드 필터링
                    matched_keywords = [kw for kw in keywords if kw in title]
                    if not matched_keywords:
                        continue
                    
                    # 링크 추출
                    link_elem = title_elem if title_elem.name == 'a' else item.select_one('a')
                    
                    if not link_elem:
                        continue
                    
                    href = link_elem.get('href', '')
                    if not href:
                        continue
                    
                    full_url = urljoin(current_url, href)
                    
                    page_articles.append({
                        "title": title,
                        "url": full_url,
                        "matched_keywords": matched_keywords,
                        "page": page_num
                    })
                
                print(f"   → 발견: {len(page_articles)}개")
                all_articles.extend(page_articles)
                pages_crawled += 1
                
                # 최대 글 수 도달 시 중단
                if len(all_articles) >= max_articles:
                    all_articles = all_articles[:max_articles]
                    break
                
                # 다음 페이지 찾기 (next_button 방식)
                if pagination_type == "next_button":
                    next_link = (
                        soup.select_one('a.next') or
                        soup.select_one('a[rel="next"]') or
                        soup.select_one('.pagination a:contains("다음")') or
                        soup.select_one('.paging a:contains("다음")')
                    )
                    
                    if not next_link:
                        print("   → 다음 페이지 없음, 종료")
                        break
                    
                    next_href = next_link.get('href', '')
                    if not next_href:
                        break
                    
                    current_url = urljoin(current_url, next_href)
                
                # 글이 없으면 종료
                if len(page_articles) == 0:
                    print("   → 글 없음, 종료")
                    break
                
                # 서버 부하 방지
                time.sleep(delay_seconds)
                
            except Exception as e:
                print(f"   ⚠️  페이지 {page_num} 오류: {str(e)}")
                break
        
        return {
            "site_url": site_url,
            "pages_crawled": pages_crawled,
            "found_count": len(all_articles),
            "articles": all_articles
        }


    @tool("crawl_and_extract_batch_multi_page")
    def crawl_and_extract_batch_multi_page(
        site_url: str,
        keywords: List[str] = None,
        max_articles: int = 30,
        max_pages: int = 5,
        pagination_type: str = "auto",
        delay_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        """
        여러 페이지 크롤링 + 본문 추출을 한번에 처리한다.
        
        crawl_and_extract_batch의 다중 페이지 버전
        """
        # 1단계: 여러 페이지에서 글 목록 수집
        crawl_result = crawl_site_with_pagination.invoke({
            "site_url": site_url,
            "keywords": keywords,
            "max_articles": max_articles,
            "max_pages": max_pages,
            "pagination_type": pagination_type,
            "delay_seconds": delay_seconds
        })
        
        if crawl_result.get("found_count", 0) == 0:
            return {
                "site_url": site_url,
                "pages_crawled": 0,
                "crawled_count": 0,
                "extracted_count": 0,
                "articles": [],
                "note": "no_articles_found"
            }
        
        # 2단계: 본문 추출
        articles_with_content = []
        
        print(f"\n📝 본문 추출 시작: {crawl_result.get('found_count')}개 글")
        
        for i, article in enumerate(crawl_result.get("articles", []), 1):
            print(f"   {i}/{crawl_result.get('found_count')}: {article['title'][:30]}...")
            
            time.sleep(delay_seconds)
            
            extract_result = extract_article_content.invoke({"article_url": article["url"]})
            
            if extract_result.get("extracted"):
                articles_with_content.append({
                    "title": article["title"],
                    "url": article["url"],
                    "content": extract_result["content"],
                    "matched_keywords": article["matched_keywords"],
                    "page": article.get("page", 1)
                })
        
        print(f"✅ 본문 추출 완료: {len(articles_with_content)}개")
        
        return {
            "site_url": site_url,
            "pages_crawled": crawl_result.get("pages_crawled", 0),
            "crawled_count": crawl_result.get("found_count", 0),
            "extracted_count": len(articles_with_content),
            "articles": articles_with_content
        }
    
    # -----------------------------
    # 기존 함수들
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

    return [vector_search, 
            web_search_snippets, 
            web_fetch_and_store, 
            web_search, 
            report_write_and_store, 
            store_snippets_only, 
            load_collected_snippets, 
            write_report_from_snippets_and_store, 
            mark_snippets_processed,
            generate_targeted_guidance,
            store_guidance_to_db,
            search_existing_guidance,
            # 크롤링 도구 추가
            crawl_site_for_phishing_cases,
            extract_article_content,
            crawl_and_extract_batch,
            crawl_site_with_pagination,
            crawl_and_extract_batch_multi_page,
            generate_guidance_from_crawled_articles,
            store_crawled_guidance
            ]
