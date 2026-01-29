# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn

from app.orchestrator_attack import build_attack_enhancement_orchestrator

# FastAPI 앱 생성
app = FastAPI(
    title="VoicePhishing Intelligence API",
    description="보이스피싱 최신 수법 지침 제공 및 크롤링 API",
    version="1.0.0"
)

# CORS 설정 (다른 시스템에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Orchestrator 초기화
print("🚀 Initializing orchestrators...")
attack_orch = build_attack_enhancement_orchestrator(model_name="gpt-4o")
print("✅ Orchestrators ready!")


# ==================== Pydantic 모델 정의 ====================

class GuidanceRequest(BaseModel):
    """지침 요청 모델"""
    phishing: bool
    type: str
    scenario: str
    victim_profile: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "phishing": True,
                "type": "검경 사칭",
                "scenario": "검찰 사칭해서 현금 편취",
                "victim_profile": {
                    "age": 65,
                    "occupation": "퇴직자"
                }
            }
        }


class CrawlRequest(BaseModel):
    """크롤링 요청 모델"""
    site_url: str
    keywords: Optional[List[str]] = None
    max_articles: Optional[int] = 30
    max_pages: Optional[int] = 5
    pagination_type: Optional[str] = "auto"
    target_type: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "site_url": "https://www.kisa.or.kr/402?page=1&searchDiv=10&searchWord=피싱",
                "keywords": ["보이스피싱", "스미싱", "피싱"],
                "max_articles": 20,
                "max_pages": 3,
                "pagination_type": "auto",
                "target_type": None
            }
        }


class GuidanceResponse(BaseModel):
    """지침 응답 모델"""
    status: str
    guidance: Dict[str, Any]
    guidance_id: Optional[str] = None
    source: str


class CrawlResponse(BaseModel):
    """크롤링 응답 모델"""
    status: str
    site_url: str
    pages_crawled: Optional[int] = 0
    crawled_count: Optional[int] = 0
    extracted_count: Optional[int] = 0
    types_generated: Optional[int] = 0
    guidance_ids: Optional[List[str]] = []
    guidance: Optional[Dict[str, Any]] = None
    source_articles: Optional[List[Dict[str, str]]] = []

class AttackEnhancementRequest(BaseModel):
    """공격 강화 분석 요청"""
    conversation_summary: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_summary": "피해자는 30대 중후반 남자 직장인이다. 피싱범은 전화로 연락해 공식 기관 소속을 내세워 신뢰를 얻으려 한다..."
            }
        }

# ==================== API 엔드포인트 ====================

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "service": "VoicePhishing Intelligence API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "guidance": "/api/guidance",
            "crawl": "/api/crawl",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "guidance_orchestrator": "ready",
        "crawl_orchestrator": "ready"
    }


@app.post("/api/attack/enhance")
async def enhance_attack_scenario(request: AttackEnhancementRequest):
    """
    대화 요약 분석 → 취약점 파악 → 강화 수법 생성
    
    **Process:**
    1. 피해자 프로필 추출
    2. 취약점 질문 생성
    3. 웹 검색 (심리학/사회학 관점)
    4. 수법 10개 생성
    5. 적합한 수법 3개 이상 선택
    6. 최종 리포트 작성
    """
    try:
        result = attack_orch.handle(request.dict())
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("message")
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 서버 실행 ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 VoicePhishing Intelligence API Server")
    print("="*60)
    print("📍 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔍 Health: http://localhost:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )