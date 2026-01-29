# app/agent_graph_crawl.py
from __future__ import annotations

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.agent_tools import build_tools


SYSTEM_PROMPT_CRAWL = """\
너는 특정 사이트에서 보이스피싱 최신 사례를 크롤링하고 지침을 생성하는 에이전트다.

입력 형식:
{
  "site_url": "https://example.com/notices",
  "keywords": ["보이스피싱", "스미싱"],  // 선택
  "max_articles": 30,  // 선택 (기본 30)
  "max_pages": 5,  // 선택 (기본 5페이지)
  "pagination_type": "auto",  // auto | url_param | path | next_button
  "target_type": "검경 사칭"  // 선택
}

절차:
1) crawl_and_extract_batch_multi_page(
     site_url=site_url,
     keywords=keywords,
     max_articles=max_articles,
     max_pages=max_pages,
     pagination_type=pagination_type
   )
   - extracted_count가 0이면 {"status": "no_articles"} 반환 후 종료

2) generate_guidance_from_crawled_articles(
     articles=크롤링결과.articles,
     target_type=target_type
   )

3) store_crawled_guidance(
     guidance_data=생성결과.guidance,
     site_url=site_url,
     source_articles=생성결과.source_articles
   )

4) 최종 JSON 출력:
{
  "status": "success",
  "site_url": "...",
  "pages_crawled": 5,
  "crawled_count": 30,
  "extracted_count": 25,
  "types_generated": 3,
  "guidance_ids": ["...", "...", "..."],
  "guidance": {...},
  "source_articles": [...]
}
""".strip()


def build_crawl_agent_graph(vectordb, model_name: Optional[str] = None):
    """사이트 크롤링 전용 에이전트"""
    all_tools = build_tools(vectordb)
    
    allow = {
        "crawl_and_extract_batch_multi_page",
        "generate_guidance_from_crawled_articles",
        "store_crawled_guidance",
    }
    tools = [t for t in all_tools if t.name in allow]
    
    if len(tools) < 3:
        raise RuntimeError(f"Missing crawl tools. Found: {[t.name for t in tools]}")
    
    llm = ChatOpenAI(
        model=(model_name or "gpt-4o"),
        temperature=0,
        timeout=180,
        max_retries=3,
    ).bind_tools(tools)
    
    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT_CRAWL)] + state["messages"]
        resp = llm.invoke(messages)
        return {"messages": [resp]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()