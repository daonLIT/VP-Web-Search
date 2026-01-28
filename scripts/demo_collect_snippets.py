import json
from app.tools.store import get_chroma
from app.tools.agent_tools import build_tools

def main():
    vectordb = get_chroma()
    tools = build_tools(vectordb)
    tool_map = {t.name: t for t in tools}

    query_used = "보이스피싱 최신 수법"

    # 1) 검색(스니펫만)
    snippets = tool_map["web_search_snippets"].invoke(
        {
            "query": query_used,
            "topic": "news",
            "time_range": "week",
            "search_depth": "advanced",
            # 필요하면 SNS 제외
            "exclude_domains": ["x.com", "instagram.com", "youtube.com", "namu.wiki", "blog.naver.com"],
            "max_results": 5,
        }
    )

    # 2) 저장(LLM 없음)
    res = tool_map["store_snippets_only"].invoke(
        {
            "query_used": query_used,
            "snippets": snippets,
            "kind": "voicephishing_snippet_v1",
        }
    )

    out = {
        "query_used": query_used,
        "found": len(snippets),
        "stored": res.get("stored"),
        "skipped": res.get("skipped"),
        "kind": res.get("kind"),
        "sources": [{"title": s.get("title"), "url": s.get("url")} for s in snippets],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
