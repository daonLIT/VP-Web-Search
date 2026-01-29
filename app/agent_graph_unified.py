# app/agent_graph_unified.py
from __future__ import annotations

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.agent_tools import build_tools


SYSTEM_PROMPT_UNIFIED = """\
너는 보이스피싱 지침을 제공하는 통합 에이전트다.

입력:
{
  "phishing": true,
  "type": "검경 사칭",
  "scenario": "...",
  "victim_profile": {...}
}

절차:
1) search_existing_guidance(phishing_type=type, scenario_hint=scenario)
   - found=true → guidances[0] 사용 → format_guidance_response(..., source="database") → 종료

2) found=false →
   a) search_and_crawl_combined(
        phishing_type=type,
        scenario=scenario,
        victim_profile=victim_profile
      )
   
   b) generate_unified_guidance(
        phishing_type=type,
        scenario=scenario,
        web_results=결과.web_search_results,
        crawled_articles=결과.crawled_articles,
        victim_profile=victim_profile
      )
   
   c) store_guidance_to_db(guidance=생성결과.guidance)
   
   d) format_guidance_response(
        guidance=생성결과.guidance,
        source="unified_search",
        guidance_id=저장ID
      )

최종 출력:
{
  "status": "found_in_db" | "generated_new",
  "guidance": {...},
  "guidance_id": "...",
  "source": "database" | "unified_search",
  "sources_used": {
    "web_search": 5,
    "crawl": 3
  }
}
""".strip()


def build_unified_agent_graph(vectordb, model_name: Optional[str] = None):
    """통합 에이전트 그래프"""
    all_tools = build_tools(vectordb)
    
    allow = {
        "search_existing_guidance",
        "search_and_crawl_combined",
        "generate_unified_guidance",
        "store_guidance_to_db",
        "format_guidance_response",
    }
    tools = [t for t in all_tools if t.name in allow]
    
    llm = ChatOpenAI(
        model=(model_name or "gpt-4o"),
        temperature=0,
        timeout=180,
        max_retries=3,
    ).bind_tools(tools)
    
    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT_UNIFIED)] + state["messages"]
        resp = llm.invoke(messages)
        return {"messages": [resp]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()