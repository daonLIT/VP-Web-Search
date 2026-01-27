from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.tools.judgment import judgment_tool
from app.tools.compare import compare_tool
from app.tools.search import search_tool
from app.tools.report_writing import report_writing_tool

def build_graph(vectordb):
    g = StateGraph(AgentState)

    def node_judgment(state: AgentState) -> AgentState:
        j = judgment_tool(state["incoming"])
        return {"judgment": j, "query": j["query"]}

    def node_compare(state: AgentState) -> AgentState:
        j = state.get("judgment", {})
        # needs_web이면 강제 MISS로 보내도 됨(“최근성 요구 → 웹” 전략)
        if j.get("needs_web"):
            return {"route": "MISS", "hits": [], "hit_scores": []}

        route, docs, scores = compare_tool(vectordb, state["query"])
        return {"route": route, "hits": docs, "hit_scores": scores}

    def node_search(state: AgentState) -> AgentState:
        results = search_tool(state["query"], max_results=5)
        return {"web_results": results}

    def node_report(state: AgentState) -> AgentState:
        out = report_writing_tool(state.get("web_results", []))
        return {"report": out["report"], "to_store": out["to_store"]}

    def node_store(state: AgentState) -> AgentState:
        docs = state.get("to_store") or []
        if docs:
            vectordb.add_documents(docs)
            # Chroma persist는 integration에 따라 자동/옵션이지만, 보통 persist_directory 사용 시 저장됨
        return {}

    def node_outgoing(state: AgentState) -> AgentState:
        # 외부 시스템으로 보낼 payload 구성
        if state["route"] == "HIT":
            outgoing = {
                "route": "HIT",
                "query": state.get("query"),
                "top_docs": [
                    {"text": d.page_content[:500], "metadata": d.metadata} for d in (state.get("hits") or [])
                ],
                "scores": state.get("hit_scores", []),
            }
        else:
            outgoing = {
                "route": "MISS",
                "query": state.get("query"),
                "report": state.get("report"),
            }
        return {"outgoing": outgoing}

    g.add_node("judgment", node_judgment)
    g.add_node("compare", node_compare)
    g.add_node("search", node_search)
    g.add_node("report", node_report)
    g.add_node("store", node_store)
    g.add_node("outgoing", node_outgoing)

    g.set_entry_point("judgment")
    g.add_edge("judgment", "compare")

    def route_fn(state: AgentState) -> str:
        return "search" if state["route"] == "MISS" else "outgoing"

    # compare 후 HIT면 outgoing, MISS면 search로 분기
    g.add_conditional_edges("compare", route_fn, {"search": "search", "outgoing": "outgoing"})

    g.add_edge("search", "report")
    g.add_edge("report", "store")
    g.add_edge("store", "outgoing")
    g.add_edge("outgoing", END)

    return g.compile()
