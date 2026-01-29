# app/agent_graph_guidance.py
# 최종 그래프
from __future__ import annotations

from typing import Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.agent_tools import build_tools

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_GUIDANCE = """\
너는 다른 시스템의 요청을 받아 보이스피싱 수법 지침을 제공하는 에이전트다.

입력 형식:
{
  "phishing": true,
  "type": "검경 사칭",
  "scenario": "검찰 사칭해서 현금 편취",
  "victim_profile": {...}  // 선택
}

절차:
1) search_existing_guidance(phishing_type=type, scenario_hint=scenario)를 호출해 DB에서 검색
2) found=true이고 count>=1이면:
   - guidances[0]를 그대로 반환
   - 출력 JSON 형식:
   {
     "status": "found_in_db",
     "guidance": {...},
     "source": "database"
   }
3) found=false이면:
   - generate_targeted_guidance(phishing_type=type, scenario=scenario, victim_profile=...)를 호출
   - store_guidance_to_db(guidance=생성결과)를 호출해 저장
   - 출력 JSON 형식:
   {
     "status": "generated_new",
     "guidance": {...},
     "guidance_id": "...",
     "source": "web_search"
   }

최종 출력은 반드시 위 JSON 형식으로만 작성하라.
""".strip()


def build_guidance_agent_graph(vectordb, model_name: Optional[str] = None):
    """
    외부 시스템 요청 처리용 에이전트 그래프
    """
    all_tools = build_tools(vectordb)
    
    # 필요한 도구만 선택
    allow = {
        "search_existing_guidance",
        "generate_targeted_guidance", 
        "store_guidance_to_db"
    }
    tools = [t for t in all_tools if t.name in allow]
    
    if len(tools) < 3:
        raise RuntimeError(f"Missing tools. Found: {[t.name for t in tools]}")
    
    llm = ChatOpenAI(
        model=(model_name or "gpt-4o"),
        temperature=0,
        timeout=90,
        max_retries=3,
    ).bind_tools(tools)
    
    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT_GUIDANCE)] + state["messages"]
        logger.info("🤖 Agent 호출 중...")
        resp = llm.invoke(messages)

        # 도구 호출 로깅
        if hasattr(resp, 'tool_calls') and resp.tool_calls:
            for tc in resp.tool_calls:
                logger.info(f"🔧 도구 호출: {tc.get('name', 'unknown')}")

        return {"messages": [resp]}
    
    def tools_node_wrapper(state: MessagesState):
        logger.info("⚙️  도구 실행 중...")
        tool_node = ToolNode(tools)
        result = tool_node.invoke(state)
        logger.info("✅ 도구 실행 완료")
        return result
    
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node_wrapper)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()