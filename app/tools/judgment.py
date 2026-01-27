from typing import Any, Dict

def judgment_tool(incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    외부 시스템이 보낸 판단 정보를 바탕으로:
    - query(키워드)
    - 최근성 필요(needs_web)
    - 특정 사이트 제한(domains)
    등을 추출/정규화
    """
    text = (incoming.get("text") or incoming.get("message") or "").strip()
    keywords = incoming.get("keywords") or []

    # 매우 단순 규칙: 최근/뉴스 키워드가 있으면 웹 우선
    needs_web = any(k in text for k in ["오늘", "어제", "최근", "속보", "뉴스", "발표", "업데이트"])
    domains = incoming.get("domains")  # ["example.com"] 같은 도메인 제한을 원하면 사용

    query = " ".join(keywords).strip() if keywords else text[:200]
    return {
        "query": query,
        "needs_web": bool(needs_web),
        "domains": domains,
    }
