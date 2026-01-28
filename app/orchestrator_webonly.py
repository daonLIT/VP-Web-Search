from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from app.config import SETTINGS
from app.tools.store import get_chroma
from app.agent_graph_webonly import build_webonly_agent_graph


@dataclass
class WebOnlyOrchestrator:
    app: Any

    def handle(self, text: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        inputs = {"messages": [HumanMessage(content=text)]}
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

        out_state = self.app.invoke(inputs, config=config)
        final_msg = out_state["messages"][-1]

        return {"final": final_msg.content}


def build_webonly_orchestrator() -> WebOnlyOrchestrator:
    if not SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. (.env 확인)")

    embeddings = OpenAIEmbeddings()  # ✅ 저장 시 임베딩 생성에 사용
    vectordb = get_chroma(embeddings)

    app = build_webonly_agent_graph(vectordb=vectordb, model_name=SETTINGS.model_name)
    return WebOnlyOrchestrator(app=app)
