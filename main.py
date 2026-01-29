# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn

from app.orchestrator_guidance import build_guidance_orchestrator
from app.orchestrator_crawl import build_crawl_orchestrator
from app.orchestrator_unified import build_unified_orchestrator

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
guidance_orch = build_guidance_orchestrator(model_name="gpt-4o")
crawl_orch = build_crawl_orchestrator(model_name="gpt-4o")
unified_orch = build_unified_orchestrator(model_name="gpt-4o")
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


@app.post("/api/guidance", response_model=GuidanceResponse)
async def get_phishing_guidance(request: GuidanceRequest):
    """
    보이스피싱 지침 요청
    
    - **DB에 있으면**: 기존 지침 반환
    - **없으면**: 웹 검색 → 생성 → 저장 → 반환
    
    **Request Body:**
```json
    {
        "phishing": true,
        "type": "검경 사칭",
        "scenario": "검찰 사칭해서 현금 편취",
        "victim_profile": {
            "age": 65,
            "occupation": "퇴직자"
        }
    }
```
    
    **Response:**
```json
    {
        "status": "found_in_db" | "generated_new",
        "guidance": {
            "type": "검경 사칭",
            "keywords": [...],
            "scenario": [...],
            "red_flags": [...],
            "recommended_actions": [...]
        },
        "guidance_id": "...",
        "source": "database" | "web_search"
    }
```
    """
    try:
        result = guidance_orch.handle(request.dict())
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown error")
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crawl", response_model=CrawlResponse)
async def crawl_site_for_guidance(request: CrawlRequest):
    """
    특정 사이트 크롤링 → 지침 생성
    
    - 목록 페이지에서 보이스피싱 관련 글 필터링
    - 각 글의 본문 추출
    - LLM으로 지침 생성
    - DB에 저장
    
    **Request Body:**
```json
    {
        "site_url": "https://www.kisa.or.kr/402?page=1",
        "keywords": ["보이스피싱", "스미싱"],
        "max_articles": 20,
        "max_pages": 3,
        "pagination_type": "auto",
        "target_type": null
    }
```
    
    **Response:**
```json
    {
        "status": "success",
        "site_url": "...",
        "pages_crawled": 3,
        "crawled_count": 20,
        "extracted_count": 18,
        "types_generated": 3,
        "guidance_ids": ["...", "...", "..."],
        "guidance": {...},
        "source_articles": [...]
    }
```
    """
    try:
        result = crawl_orch.handle(request.dict())
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown error")
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 병합 엔드포인트
@app.post("/api/guidance/unified")
async def get_phishing_guidance_unified(request: GuidanceRequest):
    """
    **통합 지침 API** (권장)
    
    1. DB 검색
    2. 없으면 → 웹 검색 + 사이트 크롤링 동시 실행
    3. 결과 통합 → 지침 생성 → 저장 → 반환
    
    기존 `/api/guidance`보다 더 많은 출처로 정확한 지침 제공
    """
    try:
        result = unified_orch.handle(request.dict())
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown error")
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