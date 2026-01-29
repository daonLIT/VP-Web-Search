# app/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.orchestrator_guidance import build_guidance_orchestrator


router = APIRouter()
guidance_orch = build_guidance_orchestrator()


class GuidanceRequest(BaseModel):
    phishing: bool
    type: str
    scenario: str
    victim_profile: Optional[Dict[str, Any]] = None


class GuidanceResponse(BaseModel):
    status: str
    guidance: Dict[str, Any]
    guidance_id: Optional[str] = None
    source: str


@router.post("/api/guidance", response_model=GuidanceResponse)
async def get_phishing_guidance(request: GuidanceRequest):
    """
    다른 시스템에서 보이스피싱 지침 요청
    
    요청 예시:
    {
        "phishing": true,
        "type": "검경 사칭",
        "scenario": "검찰 사칭해서 현금 편취",
        "victim_profile": {
            "age": 65,
            "occupation": "퇴직자"
        }
    }
    """
    try:
        result = guidance_orch.handle(request.dict())
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))