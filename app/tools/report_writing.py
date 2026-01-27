from typing import Any, Dict, List
from langchain_core.documents import Document
from datetime import datetime, timezone

def report_writing_tool(web_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    기본 구현: Tavily 결과를 '근거 링크 + 핵심 bullet'로 정리하고,
    Chroma에 저장할 Document 리스트로 변환.
    """
    now = datetime.now(timezone.utc).isoformat()

    bullets = []
    to_store: List[Document] = []

    for r in web_results or []:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("raw_content") or r.get("content") or "").strip()

        if title or url:
            bullets.append(f"- {title} ({url})")

        if content:
            to_store.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url or "tavily",
                        "title": title,
                        "fetched_at": now,
                        "kind": "web",
                    },
                )
            )

    report = "웹 검색 결과 요약\n" + "\n".join(bullets) if bullets else "웹 검색 결과가 비어있습니다."
    return {"report": report, "to_store": to_store}
