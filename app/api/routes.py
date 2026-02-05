# app/api/routes.py
"""
FastAPI 라우터
- 범용 분석 API 엔드포인트
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
)
from app.agents import build_research_agent

logger = logging.getLogger(__name__)

router = APIRouter()

# 에이전트 인스턴스 (lazy init)
_agent = None


def get_agent():
    """에이전트 싱글톤"""
    global _agent
    if _agent is None:
        logger.info("Initializing research agent...")
        _agent = build_research_agent()
        logger.info("Research agent ready")
    return _agent


# ==================== 엔드포인트 ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        services={
            "agent": "ready",
            "analyzer": "ready",
            "searcher": "ready",
        },
        version="2.0.0",
    )


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """
    범용 데이터 분석 API

    어떤 형태의 데이터든 받아서:
    1. 데이터 분석/요약
    2. 검색 쿼리 생성
    3. 웹 검색 수행
    4. 기법 생성
    5. 리포트 작성

    **입력 예시:**

    텍스트 데이터:
    ```json
    {
        "data": "피해자는 30대 직장인 남성이다...",
        "analysis_type": "conversation"
    }
    ```

    JSON 데이터:
    ```json
    {
        "data": {
            "victim_age": 65,
            "occupation": "퇴직자",
            "transcript": "여보세요..."
        }
    }
    ```

    리스트 데이터:
    ```json
    {
        "data": ["대화1", "대화2", "대화3"]
    }
    ```
    """
    try:
        agent = get_agent()
        result = agent.run(request)

        if result.status == "error":
            raise HTTPException(
                status_code=500,
                detail=result.error or "Analysis failed",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/analyze/quick")
async def quick_analyze(request: AnalysisRequest):
    """
    빠른 분석 (웹 검색 없이 요약만)

    웹 검색을 건너뛰고 데이터 분석/요약만 수행합니다.
    """
    try:
        from app.services.analyzer import DataAnalyzer

        analyzer = DataAnalyzer()
        analysis = analyzer.analyze(
            data=request.data,
            analysis_type=request.analysis_type,
            context=request.context,
        )

        return {
            "status": "success",
            "summary": analysis.summary,
            "key_points": analysis.key_points,
            "profile": analysis.extracted_profile.model_dump() if analysis.extracted_profile else None,
            "scenario": analysis.detected_scenario,
            "vulnerabilities": analysis.vulnerability_areas,
            "suggested_queries": analysis.search_queries,
        }

    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/search")
async def search_web(queries: List[str], extract_content: bool = True):
    """
    웹 검색 API

    검색 쿼리 리스트를 받아 웹 검색 수행
    """
    try:
        from app.services.searcher import WebSearcher

        searcher = WebSearcher()
        results = searcher.search(
            queries=queries,
            extract_content=extract_content,
        )

        return {
            "status": "success",
            "count": len(results),
            "results": [r.model_dump() for r in results],
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
