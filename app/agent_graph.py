from __future__ import annotations
from typing import Any, Dict

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
import time
import random
from openai import RateLimitError

from app.tools.agent_tools import build_tools

SYSTEM_PROMPT = """\
너는 '웹 검색 컨텍스트 엔진' 에이전트다.
목표: 사용자의 요청을 보고, 가능한 한 기존 VectorDB(Chroma)에서 유사 정보를 찾고(HIT), 없거나 최신성이 필요하면(MISS) 웹검색으로 보강한다.

규칙(중요):
1) 항상 먼저 vector_search(query)를 호출해서 내부 지식(HIT/MISS)을 확인한다.
2) vector_search가 HIT이고 score가 충분하면, 웹검색을 하지 말고 내부 근거로 답한다.
3) vector_search가 MISS이거나, 사용자가 '최근/뉴스/오늘/어제/속보/업데이트' 등 최신성을 요구하면 web_search를 호출한다.
   - 뉴스 성격이면 topic="news"를 우선 고려한다.
4) web_search를 했다면 store_web_results로 저장한다(중복/품질은 나중에 개선 가능).
5) 최종 출력은 반드시 JSON 한 덩어리로만 출력한다. (설명 텍스트 금지)
   스키마:
   {
     "route": "HIT" | "MISS",
     "query": "...",
     "answer": "...",
     "evidence": [
       {"source": "...", "title": "...", "score": 0.0, "snippet": "..."}
     ]
   }
"""

def build_agent_graph(vectordb, model_name: str) -> Any:
    tools = build_tools(vectordb)
    tool_node = ToolNode(tools)

    llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(tools)

    def agent_node(state: MessagesState) -> Dict[str, Any]:
        messages = state["messages"]
        if not messages or getattr(messages[0], "type", None) != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # ✅ rate limit 자동 재시도
        max_retries = 6
        base_sleep = 0.3  # seconds
        for attempt in range(max_retries):
            try:
                resp = llm.invoke(messages)
                return {"messages": [resp]}
            except RateLimitError as e:
                # 지수 백오프 + 지터
                sleep_s = (base_sleep * (2 ** attempt)) + random.uniform(0, 0.2)
                time.sleep(sleep_s)

        # 마지막까지 실패하면 예외 재발생
        resp = llm.invoke(messages)
        return {"messages": [resp]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)

    g.add_edge(START, "agent")

    # tool_calls 있으면 tools로, 없으면 END로 종료 :contentReference[oaicite:15]{index=15}
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    # tools 실행 후 다시 agent로 돌아가서 다음 행동 결정
    g.add_edge("tools", "agent")

    return g.compile()
