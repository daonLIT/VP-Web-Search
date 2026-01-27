from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import SETTINGS
from app.graph import build_graph
from app.tools.store import get_chroma

# 임베딩은 프로젝트 상황에 맞게 교체 가능
# (OpenAIEmbeddings 대신 로컬 임베딩을 쓰면 OPENAI_API_KEY 없어도 됨)
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None  # type: ignore


@dataclass
class Orchestrator:
    """
    메인 오케스트레이터:
    - 외부 시스템으로부터 payload를 받는다(여기서는 handle()).
    - LangGraph 앱을 실행한다.
    - 외부 시스템으로 보낼 outgoing payload를 반환한다.

    실제 운영에서는 handle() 앞뒤로:
    - HTTP 서버(FastAPI), 메시지 큐(Kafka/RabbitMQ), 웹훅, gRPC 등으로
      '받기/보내기'를 붙이면 됨.
    """

    app: Any  # LangGraph compiled app
    external_sender: Optional["ExternalSender"] = None

    def handle(self, incoming_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        외부 시스템에서 받은 payload를 처리하고,
        외부 시스템으로 전달할 outgoing payload를 리턴(또는 전송)한다.
        """
        result = self.app.invoke({"incoming": incoming_payload})
        outgoing = result.get("outgoing") or {}

        # "보내기"가 연결되어 있으면 전송까지 수행
        if self.external_sender is not None:
            self.external_sender.send(outgoing)

        return outgoing


class ExternalSender:
    """
    예시용 Sender 인터페이스.
    운영에서는 이 클래스를 상속/구현해서:
    - HTTP POST
    - 메시지 큐 publish
    - 다른 내부 API 호출
    등을 수행하면 됨.
    """

    def send(self, payload: Dict[str, Any]) -> None:
        # 기본 구현: 콘솔 출력(데모)
        print("[ExternalSender] sending payload:")
        print(payload)


def build_app():
    """
    LangGraph 앱(컴파일된 그래프) 생성.

    embeddings / vectordb 초기화는 여기서 한 번만 수행하고,
    Orchestrator가 그 앱을 재사용하도록 구성(성능/일관성).
    """
    if OpenAIEmbeddings is None:
        raise RuntimeError(
            "langchain-openai가 설치되어 있지 않습니다. "
            "requirements.txt에 langchain-openai를 추가하거나 임베딩을 교체하세요."
        )

    # OpenAI 임베딩 사용(로컬 임베딩으로 대체 가능)
    if not SETTINGS.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 비어있습니다. "
            "로컬 임베딩으로 교체하거나 .env에 OPENAI_API_KEY를 설정하세요."
        )

    embeddings = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings)

    return build_graph(vectordb)


def make_orchestrator(sender: Optional[ExternalSender] = None) -> Orchestrator:
    """
    Orchestrator 인스턴스 생성 헬퍼.
    """
    app = build_app()
    return Orchestrator(app=app, external_sender=sender)
