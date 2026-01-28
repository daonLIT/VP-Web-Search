# app/agent_graph_summarize.py
from __future__ import annotations

from typing import Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.agent_tools import build_tools


SYSTEM_PROMPT_SUMMARIZE = """\
너는 수집된 보이스피싱 스니펫을 읽고 리포트를 저장하는 요약 에이전트다.

절차:
1) load_collected_snippets(limit=10, only_unprocessed=true)를 호출한다.
2) items가 0이면 "no_work"를 출력하고 종료한다.
3) items가 있으면 write_report_from_snippets_and_store(query_used="보이스피싱 최신 수법", snippet_items=items)를 호출한다.
4) 성공하면 mark_snippets_processed(doc_ids=[items의 doc_id들], report_id=...)를 호출한다.
5) 최종 출력은 아래 JSON:
{
  "status": "ok" | "no_work",
  "used": number,
  "report_id": "...",
  "updated": number
}
""".strip()


def build_summarize_agent_graph(vectordb, model_name: Optional[str] = None):
    """
    요약 전용 에이전트 그래프:
    - DB에서 수집된 snippet 문서를 읽고
    - LLM이 리포트를 생성해 별도 kind로 저장
    - 사용한 snippet 문서를 processed=True로 업데이트
    """

    all_tools = build_tools(vectordb)

    # 요약 단계에서 허용할 tool만 남김
    allow = {"load_collected_snippets", "write_report_from_snippets_and_store", "mark_snippets_processed"}
    tools = [t for t in all_tools if t.name in allow]

    if not tools:
        raise RuntimeError(
            "No summarize tools found. Ensure build_tools() returns: "
            "load_collected_snippets, write_report_from_snippets_and_store, mark_snippets_processed"
        )

    # 모델명은 외부에서 주거나, 기본값 사용
    llm = ChatOpenAI(
        model=(model_name or "gpt-4o-mini"),
        temperature=0,
        timeout=90,
        max_retries=3,
    ).bind_tools(tools)

    def agent_node(state: MessagesState):
        # system + 기존 메시지로 호출
        messages = [SystemMessage(content=SYSTEM_PROMPT_SUMMARIZE)] + state["messages"]
        resp = llm.invoke(messages)
        return {"messages": [resp]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")

    # agent 출력에 tool_calls가 있으면 tools로, 없으면 종료
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    # tool 실행 후 다시 agent로
    graph.add_edge("tools", "agent")

    return graph.compile()
