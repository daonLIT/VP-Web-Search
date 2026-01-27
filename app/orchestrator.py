from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage

from app.config import SETTINGS
from app.tools.store import get_chroma
from app.agent_graph import build_agent_graph

@dataclass
class Orchestrator:
    app: Any

    def handle(self, text: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        text(외부 시스템에서 온 요청)를 에이전트에 전달하고,
        최종 메시지(모델의 최종 답)를 반환한다.
        """
        inputs = {"messages": [HumanMessage(content=text)]}

        # thread_id는 체크포인터/세션 유지에 사용(지금은 optional)
        config = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        out_state = self.app.invoke(inputs, config=config)

        # 마지막 메시지가 최종 응답(문자열 JSON)
        final_msg = out_state["messages"][-1]
        return {
            "final": final_msg.content,
            "messages": [m.dict() if hasattr(m, "dict") else str(m) for m in out_state["messages"]],
        }

    def stream(self, text: str, thread_id: Optional[str] = None) -> None:
        """
        디버그/관측용: 에이전트가 툴을 어떻게 호출하는지 스트리밍 출력
        """
        inputs = {"messages": [HumanMessage(content=text)]}
        config = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        for event in self.app.stream(inputs, config=config, stream_mode=["values"]):
            # values는 state 전체가 계속 갱신되어 보임 (운영에선 custom/debug로 조절 가능)
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                print("\n--- last message ---")
                print(getattr(last, "content", last))

def build_orchestrator() -> Orchestrator:
    if not SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. (.env 확인)")

    embeddings = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings)

    app = build_agent_graph(vectordb=vectordb, model_name=SETTINGS.model_name)
    return Orchestrator(app=app)
