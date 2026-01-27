import json
from app.graph import build_graph
from app.tools.store import get_chroma
from app.orchestrator import make_orchestrator

# 임베딩: 여기서는 예시로 OpenAI 임베딩을 쓰되,
# 로컬 임베딩(sentence-transformers)으로 교체해도 됨.
from langchain_openai import OpenAIEmbeddings
from app.config import SETTINGS

def main():
    if not SETTINGS.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. (로컬 임베딩으로 바꾸려면 demo_run.py 수정)")

    embeddings = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings)

    app = build_graph(vectordb)

    incoming = {
        "text": "최근 발표된 AI 에이전트 웹 검색 관련 뉴스 요약해줘",
        "keywords": ["AI agent", "web search", "Tavily", "LangGraph"],
    }

    out = app.invoke({"incoming": incoming})
    print(json.dumps(out["outgoing"], ensure_ascii=False, indent=2))

    orc = make_orchestrator()
    outgoing = orc.handle({"text": "최근 AI 에이전트 웹 검색 뉴스 알려줘"})
    print(outgoing)
    
if __name__ == "__main__":
    main()
