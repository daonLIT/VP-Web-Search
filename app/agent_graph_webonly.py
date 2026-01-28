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

목표:
- 사용자의 요청을 해석해서 웹검색 쿼리를 정교화하고,
- web_search_snippets로 짧은 결과를 확인한 뒤,
- web_fetch_and_store로 원문(raw_content)을 DB에 저장한다.
- 마지막에 저장 결과와 근거 URL 목록을 JSON으로 출력한다.

규칙(강제):
1) vector_search는 절대 호출하지 마라. (DB 조회 금지)
2) 항상 먼저 web_search_snippets(...)를 호출해 snippets를 얻어라.
3) 다음으로 web_fetch_and_store(...)를 호출해 원문 저장을 시도하라. (stored/skipped를 얻어라)
4) 마지막으로 report_write_and_store(query_used, sources, snippets, stored, skipped)를 호출해
   리포트를 생성하고 DB에 저장하라.
5) 최종 출력은 report_write_and_store의 결과 + sources를 포함한 JSON만 출력하라.
   스키마:
   {
     "query_used": "...",
     "topic": "...",
     "time_range": "...",
     "stored": number,
     "skipped": number,
     "sources": [{"title":"...","url":"..."}]
   }
"""

def build_webonly_agent_graph(vectordb, model_name: str) -> Any:
    # ✅ tools는 build_tools에서 가져오되, web-only로 제한한다.
    all_tools = build_tools(vectordb)
    tools = [t for t in all_tools if t.name in ("web_search_snippets", "web_fetch_and_store", "report_write_and_store")]

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
