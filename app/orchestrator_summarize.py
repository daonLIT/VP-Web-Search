# app/orchestrator_summarize.py
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_openai import OpenAIEmbeddings

from app.agent_graph_summarize import build_summarize_agent_graph

# vectordb getter는 프로젝트마다 다를 수 있음
# 아래 import가 안 맞으면 네 프로젝트의 vectordb 모듈/함수명으로 바꿔줘.
from app.tools.store import get_chroma

# settings도 프로젝트마다 다를 수 있음
try:
    from app.config import SETTINGS
except Exception:
    SETTINGS = None


class SummarizeOrchestrator:
    def __init__(self, app):
        self.app = app

    def handle(self, text: str, thread_id: str = "summarize") -> Dict[str, Any]:
        inputs = {"input": text}
        config = {"configurable": {"thread_id": thread_id}}

        out_state = self.app.invoke(inputs, config=config)

        # 마지막 메시지(content)가 보통 최종 JSON 텍스트
        msgs = out_state.get("messages") or []
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str) and content.strip():
                return {"final": content}

        return {"final": out_state}


def build_summarize_orchestrator(model_name: Optional[str] = None) -> SummarizeOrchestrator:
    embeddings = OpenAIEmbeddings()  # ✅ 저장 시 임베딩 생성에 사용
    vectordb = get_chroma(embeddings)

    if model_name is None and SETTINGS is not None:
        model_name = getattr(SETTINGS, "model_name", None)

    app = build_summarize_agent_graph(vectordb=vectordb, model_name=model_name)
    return SummarizeOrchestrator(app)
