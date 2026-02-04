# app/agent_graph_attack.py
from __future__ import annotations

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.agent_tools import build_tools


SYSTEM_PROMPT_ATTACK = """\
너는 보이스피싱 대화를 분석하고 공격 수법을 강화하는 에이전트다.

입력:
{
  "conversation_summary": "대화 요약 텍스트"
}

절차:
1) analyze_conversation_summary(conversation_summary=...)
   → victim_profile, current_scenario, vulnerability_questions 추출

2) 각 vulnerability_question에 대해:
   a) generate_search_queries_from_question(question=..., victim_profile=...)
   b) search_vulnerability_info(search_queries=..., extract_full_content=true) 
      → 본문까지 추출된 결과 반환


3) generate_attack_techniques(
     vulnerability_info=모든 검색 결과 (전체 본문 포함),
     victim_profile=...,
     current_scenario=...,
     victim_suspicion_points=...
   )
   → 10개 수법 생성

4) filter_and_select_techniques(techniques=..., min_score=0.6, target_count=3)
   → need_more=true이면 2번으로 돌아가 추가 질문 생성
   → need_more=false이면 5번으로

5) create_attack_enhancement_report(
     conversation_summary=...,
     victim_profile=...,
     current_scenario=...,
     selected_techniques=...,
     analysis_data=...
   )

최종 출력:
{
  "status": "success",
  "report": {...},
  "metadata": {...}
}

중요:
- filter 단계에서 need_more=true이면 최대 2번까지 반복
- 최소 3개의 수법이 선택될 때까지 계속 시도
""".strip()


def build_attack_enhancement_agent_graph(vectordb, model_name: Optional[str] = None):
    """공격 강화 분석 에이전트"""
    all_tools = build_tools(vectordb)
    
    # 필요한 도구만
    allow = {
        "analyze_conversation_summary",
        "generate_search_queries_from_question",
        "search_vulnerability_info",
      #   "search_and_extract_vulnerability_info",
        "generate_attack_techniques",
        "filter_and_select_techniques",
        "create_attack_enhancement_report",
    }
    
    tools = [t for t in all_tools if t.name in allow]

    if len(tools) < 6:
        raise RuntimeError(f"Missing tools. Found: {[t.name for t in tools]}")
    
    llm = ChatOpenAI(
        model=(model_name or "gpt-4o-mini"),
        temperature=0,
        timeout=180,
        max_retries=3,
    ).bind_tools(tools)
    
    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT_ATTACK)] + state["messages"]
        resp = llm.invoke(messages)
        return {"messages": [resp]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()