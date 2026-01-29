# app/orchestrator_crawl.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from langchain_openai import OpenAIEmbeddings

from app.agent_graph_crawl import build_crawl_agent_graph
from app.tools.store import get_chroma

try:
    from app.config import SETTINGS
except Exception:
    SETTINGS = None


class CrawlOrchestrator:
    def __init__(self, app):
        self.app = app
    
    def handle(self, request: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
        """사이트 크롤링 요청 처리"""
        if thread_id is None:
            thread_id = f"crawl_{hash(request.get('site_url', ''))}"
        
        input_text = json.dumps(request, ensure_ascii=False)
        
        inputs = {"messages": [{"role": "user", "content": input_text}]}
        config = {"configurable": {"thread_id": thread_id}}
        
        out_state = self.app.invoke(inputs, config=config)
        
        msgs = out_state.get("messages") or []
        if msgs:
            last = msgs[-1]
            content = getattr(last, "content", None)
            
            if isinstance(content, str):
                try:
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    result = json.loads(content.strip())
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Parse failed: {str(e)}",
                        "raw": content
                    }
        
        return {"status": "error", "message": "No response"}


def build_crawl_orchestrator(model_name: Optional[str] = None) -> CrawlOrchestrator:
    embeddings = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings)
    
    if model_name is None and SETTINGS is not None:
        model_name = getattr(SETTINGS, "model_name", None)
    
    app = build_crawl_agent_graph(vectordb=vectordb, model_name=model_name)
    return CrawlOrchestrator(app)