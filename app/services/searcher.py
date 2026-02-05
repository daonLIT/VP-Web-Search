# app/services/searcher.py
"""
웹 검색 서비스
- Tavily API를 사용한 검색
- 본문 크롤링 지원
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch

from app.schemas import SearchResult
from app.config import SETTINGS

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    웹 검색 서비스
    - Tavily를 사용한 검색
    - 선택적 본문 크롤링
    """

    def __init__(
        self,
        max_results_per_query: int = 3,
        search_depth: str = "basic",
        crawl_timeout: int = 10,
    ):
        self.max_results_per_query = max_results_per_query
        self.search_depth = search_depth
        self.crawl_timeout = crawl_timeout

        self.tavily = TavilySearch(
            max_results=max_results_per_query,
            topic="general",
            include_answer=False,
            include_raw_content=False,
            search_depth=search_depth,
        )

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def search(
        self,
        queries: List[str],
        extract_content: bool = True,
        max_total_results: int = 15,
    ) -> List[SearchResult]:
        """
        여러 쿼리로 웹 검색 수행

        Args:
            queries: 검색 쿼리 리스트
            extract_content: 본문 크롤링 여부
            max_total_results: 최대 총 결과 수

        Returns:
            SearchResult 리스트
        """
        # 중복 쿼리 제거
        unique_queries = list(dict.fromkeys(queries))
        logger.info(f"Searching with {len(unique_queries)} unique queries")

        # 1단계: URL 수집
        all_items = self._collect_urls(unique_queries)

        if not all_items:
            logger.warning("No search results found")
            return []

        # 중복 URL 제거
        seen_urls = set()
        unique_items = []
        for item in all_items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                unique_items.append(item)

        unique_items = unique_items[:max_total_results]
        logger.info(f"Collected {len(unique_items)} unique URLs")

        # 2단계: 본문 크롤링 (선택적)
        if extract_content:
            results = self._crawl_contents(unique_items)
        else:
            results = [
                SearchResult(
                    title=item["title"],
                    url=item["url"],
                    content=item["snippet"],
                    query=item["query"],
                    content_type="snippet",
                )
                for item in unique_items
            ]

        return results

    def _collect_urls(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Tavily로 URL 수집"""
        all_items = []

        for query in queries:
            try:
                raw_out = self.tavily.invoke({"query": query})

                # 결과 정규화
                if isinstance(raw_out, dict):
                    results = raw_out.get("results", [])
                elif isinstance(raw_out, list):
                    results = raw_out
                else:
                    results = []

                for r in results[:self.max_results_per_query]:
                    url = (r.get("url") or "").strip()
                    if url:
                        all_items.append({
                            "url": url,
                            "title": (r.get("title") or "")[:150],
                            "snippet": (r.get("content") or "")[:500],
                            "query": query,
                        })

                logger.debug(f"Query '{query}': {len(results)} results")

            except Exception as e:
                logger.error(f"Search failed for '{query}': {e}")

        return all_items

    def _crawl_contents(
        self,
        items: List[Dict[str, Any]],
        max_workers: int = 5,
    ) -> List[SearchResult]:
        """병렬로 본문 크롤링"""
        logger.info(f"Crawling {len(items)} URLs")
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(self._crawl_single, item): item
                for item in items
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Crawl failed for {item['url']}: {e}")
                    # 폴백: 스니펫 사용
                    results.append(SearchResult(
                        title=item["title"],
                        url=item["url"],
                        content=item["snippet"],
                        query=item["query"],
                        content_type="snippet",
                    ))

        logger.info(f"Crawling complete: {len(results)} results")
        return results

    def _crawl_single(self, item: Dict[str, Any]) -> SearchResult:
        """단일 URL 크롤링"""
        try:
            response = requests.get(
                item["url"],
                headers=self.headers,
                timeout=self.crawl_timeout,
            )
            response.raise_for_status()

            # 인코딩 처리
            if response.encoding is None or response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding or "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")

            # 본문 추출
            content = self._extract_content(soup)

            if content and len(content) >= 100:
                return SearchResult(
                    title=item["title"],
                    url=item["url"],
                    content=content[:3000],
                    query=item["query"],
                    content_type="full_crawled",
                )

        except requests.Timeout:
            logger.warning(f"Timeout: {item['url']}")
        except Exception as e:
            logger.warning(f"Crawl error: {e}")

        # 폴백
        return SearchResult(
            title=item["title"],
            url=item["url"],
            content=item["snippet"],
            query=item["query"],
            content_type="snippet",
        )

    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """HTML에서 본문 추출"""
        # 1. 시맨틱 태그 시도
        content_elem = soup.select_one("article")

        # 2. 일반적인 클래스/ID 시도
        if not content_elem:
            selectors = [
                "div.content",
                "div.post-content",
                "div.article-body",
                "div.entry-content",
                "div#content",
                "main",
                "div.post_content",
                "div.article_body",
            ]
            for selector in selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    break

        # 3. body 폴백
        if not content_elem:
            content_elem = soup.select_one("body")

        if not content_elem:
            return None

        # 불필요한 요소 제거
        for tag in content_elem(["script", "style", "nav", "header",
                                  "footer", "aside", "iframe", "noscript"]):
            tag.decompose()

        # 텍스트 추출 및 정제
        content = content_elem.get_text(separator="\n", strip=True)
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = re.sub(r" +", " ", content)

        return content

    def search_single_query(
        self,
        query: str,
        extract_content: bool = True,
    ) -> List[SearchResult]:
        """단일 쿼리로 검색"""
        return self.search(
            queries=[query],
            extract_content=extract_content,
            max_total_results=self.max_results_per_query,
        )
