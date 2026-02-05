# main.py
"""
VP-Web-Search API Server

범용 데이터 분석 및 웹 검색 API
- 어떤 형태의 데이터든 받아서 분석/요약
- 자동 검색 쿼리 생성 및 웹 검색
- 취약점 분석 및 기법 생성
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes import router
from app.config import SETTINGS

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클"""
    logger.info("Starting VP-Web-Search API...")
    logger.info(f"Model: {SETTINGS.model_name}")
    yield
    logger.info("Shutting down...")


# FastAPI 앱 생성
app = FastAPI(
    title="VP-Web-Search API",
    description="""
## 범용 데이터 분석 및 웹 검색 API

어떤 형태의 데이터든 받아서:
1. **분석/요약** - 데이터에서 핵심 정보 추출
2. **검색 쿼리 생성** - 관련 정보 검색을 위한 쿼리 자동 생성
3. **웹 검색** - Tavily API를 통한 검색 및 본문 크롤링
4. **기법 생성** - 수집된 정보 기반 분석 기법 생성
5. **리포트 작성** - 종합 리포트 출력

### 지원 데이터 형식
- **텍스트**: 대화 내용, 문서 등
- **JSON 객체**: 구조화된 데이터
- **배열**: 여러 항목의 리스트

### API 엔드포인트
- `POST /api/analyze` - 전체 분석 파이프라인
- `POST /api/analyze/quick` - 빠른 분석 (검색 없이)
- `POST /api/search` - 웹 검색만
- `GET /health` - 헬스 체크
    """,
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)


# 루트 엔드포인트
@app.get("/")
async def root():
    """API 정보"""
    return {
        "service": "VP-Web-Search API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /api/analyze - 전체 분석",
            "quick_analyze": "POST /api/analyze/quick - 빠른 분석",
            "search": "POST /api/search - 웹 검색",
            "health": "GET /health - 헬스 체크",
            "docs": "GET /docs - API 문서",
        },
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 VP-Web-Search API Server v2.0")
    print("=" * 60)
    print(f"📍 Server: http://localhost:8001")
    print(f"📚 Docs: http://localhost:8001/docs")
    print(f"🔍 Health: http://localhost:8001/health")
    print(f"🤖 Model: {SETTINGS.model_name}")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
