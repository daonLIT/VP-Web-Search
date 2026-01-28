from __future__ import annotations
from typing import Any, Dict

import time, random
from openai import RateLimitError

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

from app.tools.agent_tools import build_tools

SYSTEM_PROMPT_WEBONLY = """\
너는 '웹 수집 전용 에이전트'다. (중요: VectorDB 조회 금지)

절대 금지:
- vector_search 호출 금지
- web_fetch_and_store 호출 금지
- report_write_* 호출 금지
- 어떤 요약/리포트도 작성하지 마라(LLM 출력으로 길게 쓰지 마라)

반드시 수행 절차:
1) web_search_snippets를 호출한다.
   - query는 항상 "보이스피싱 최신 수법"
   - topic="news"
   - time_range="week"
   - max_results=5
   - search_depth="advanced"
   - exclude_domains는 ["x.com","instagram.com","youtube.com","namu.wiki","blog.naver.com"] 를 기본으로 사용

2) 반환된 snippets를 그대로 store_snippets_only(query_used, snippets, kind="voicephishing_snippet_v1")로 저장한다.

최종 출력:
아래 JSON 스키마로만 출력한다.
{
  "query_used": "...",
  "found": number,
  "stored": number,
  "skipped": number,
  "kind": "...",
  "sources": [{"title":"...","url":"..."}]
}
"""

def build_webonly_agent_graph(vectordb, model_name: str) -> Any:
    # ✅ tools는 build_tools에서 가져오되, web-only로 제한한다.
    all_tools = build_tools(vectordb)
    tools = [t for t in all_tools if t.name in ("web_search_snippets", "store_snippets_only")]

    tool_node = ToolNode(tools)

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        timeout=30,
        max_retries=0,  # ✅ 내부 재시도는 끄고, 아래에서 우리가 제어
    ).bind_tools(tools)

    def agent_node(state: MessagesState) -> Dict[str, Any]:
        messages = state["messages"]
        if not messages or getattr(messages[0], "type", None) != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT_WEBONLY}] + messages

        # ✅ 429 대비 짧은 재시도(너무 오래 끌지 않게)
        max_retries = 6
        base_sleep = 0.4
        for attempt in range(max_retries):
            try:
                resp = llm.invoke(messages)
                return {"messages": [resp]}
            except RateLimitError:
                time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.2))

        # 마지막 시도도 실패하면 그대로 예외
        resp = llm.invoke(messages)
        return {"messages": [resp]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile()
