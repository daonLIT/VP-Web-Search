# app/orchestrator_guidance.py
# 최종 orchestrator
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from langchain_openai import OpenAIEmbeddings

from app.agent_graph_guidance import build_guidance_agent_graph
from app.tools.store import get_chroma

try:
    from app.config import SETTINGS
except Exception:
    SETTINGS = None


class GuidanceOrchestrator:
    def __init__(self, app):
        self.app = app
    
    def handle(self, request: Dict[str, Any], thread_id: str = "guidance") -> Dict[str, Any]:
        """
        외부 시스템 요청 처리
        
        입력:
        {
            "phishing": true,
            "type": "검경 사칭",
            "scenario": "검찰 사칭해서 현금 편취",
            "victim_profile": {...}  // 선택
        }
        
        출력:
        {
            "status": "found_in_db" | "generated_new",
            "guidance": {
                "type": "...",
                "keywords": [...],
                "scenario": [...],
                "red_flags": [...],
                "recommended_actions": [...],
                "sources": [...]
            },
            "guidance_id": "...",  // generated_new인 경우
            "source": "database" | "web_search"
        }
        """
        # 요청을 텍스트로 변환
        input_text = json.dumps(request, ensure_ascii=False)
        
        inputs = {"messages": [{"role": "user", "content": input_text}]}
        config = {"configurable": {"thread_id": thread_id}}
        
        out_state = self.app.invoke(inputs, config=config)
        
        # 마지막 메시지에서 JSON 추출
        msgs = out_state.get("messages") or []
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None)
            
            if isinstance(content, str):
                # JSON 파싱 시도
                try:
                    # 코드 블록 제거
                    if "```" in content:
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    
                    result = json.loads(content.strip())
                    return result
                except Exception:
                    pass
        
        return {
            "status": "error",
            "message": "Failed to parse agent response",
            "raw": str(out_state)
        }


def build_guidance_orchestrator(model_name: Optional[str] = None) -> GuidanceOrchestrator:
    embeddings = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings)
    
    if model_name is None and SETTINGS is not None:
        model_name = getattr(SETTINGS, "model_name", None)
    
    app = build_guidance_agent_graph(vectordb=vectordb, model_name=model_name)
    return GuidanceOrchestrator(app)