# app/api/routes.py
"""
FastAPI 라우터
- 범용 분석 API 엔드포인트
- VP2 연동 엔드포인트
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

import httpx

from app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    # VP2 연동
    JudgementRequest,
    JudgementResponse,
    ConversationRequest,
    ConversationResponse,
    MethodReportRequest,
    MethodReportResponse,
)
from app.agents import build_research_agent
from app.config import SETTINGS

logger = logging.getLogger(__name__)

router = APIRouter()

# ==================== 수신 데이터 저장소 (메모리) ====================
# 실제 운영에서는 DB로 대체
_received_judgements: Dict[str, Dict[str, Any]] = {}
_received_conversations: Dict[str, Dict[str, Any]] = {}
_analysis_results: Dict[str, Dict[str, Any]] = {}  # 분석 결과 저장
_analyzed_cases: set = set()  # 이미 분석 완료된 case_id 추적
_analyzing_cases: set = set()  # 현재 분석 중인 case_id 추적

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


# ==================== VP2 연동 엔드포인트 ====================

import re

def _generate_received_id(prefix: str = "recv") -> str:
    """수신 ID 생성"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _clean_text_korean_only(text: str) -> str:
    """
    텍스트에서 한글, 숫자, 기본 문장부호만 남기고 나머지 제거
    - 영어 알파벳 제거
    - 특수기호 제거 (괄호, 따옴표 등 일부 허용)
    - 공백 정리
    """
    if not text or not isinstance(text, str):
        return ""

    # 한글(가-힣), 숫자(0-9), 기본 문장부호(.,!?), 공백만 허용
    # 괄호(), 쉼표, 마침표, 물음표, 느낌표는 유지
    cleaned = re.sub(r'[^\uAC00-\uD7A3\u3131-\u3163\u1100-\u11FF0-9\s.,!?()~]', '', text)

    # 연속 공백 제거
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


def _preprocess_turns(turns: List[Dict[str, Any]]) -> List[str]:
    """
    수신된 turns를 전처리하여 한글 위주로 정제
    - 영어/특수기호 제거
    - 순수 문자열 리스트로 반환
    """
    import json

    processed = []

    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue

        text = t.get("text", "")

        # text가 JSON 문자열인 경우 파싱
        if isinstance(text, str) and text.startswith("{"):
            try:
                parsed = json.loads(text)
                text = parsed.get("utterance") or parsed.get("dialogue") or ""
            except json.JSONDecodeError:
                pass
        # text가 dict인 경우 dialogue 추출
        elif isinstance(text, dict):
            text = text.get("dialogue") or text.get("utterance") or ""

        # 한글만 추출
        clean_text = _clean_text_korean_only(str(text))

        if clean_text:  # 빈 텍스트 제외
            processed.append(clean_text)

    return processed


def _preprocess_judgement(judgement: Dict[str, Any]) -> Dict[str, Any]:
    """
    판정 결과 전처리 - 필요한 필드만 유지하고 한글 정제
    """
    if not isinstance(judgement, dict):
        return {}

    result = {
        "phishing": judgement.get("phishing", False),
    }

    # risk 정보
    risk = judgement.get("risk", {})
    if isinstance(risk, dict):
        result["risk"] = {
            "score": risk.get("score", 0),
            "level": risk.get("level", ""),
        }

    # evidence - 한글만 추출
    evidence = judgement.get("evidence", "")
    if evidence:
        result["evidence"] = _clean_text_korean_only(str(evidence))

    # 취약점 - 한글만 추출
    vulnerabilities = judgement.get("victim_vulnerabilities", [])
    if isinstance(vulnerabilities, list):
        result["victim_vulnerabilities"] = [
            _clean_text_korean_only(str(v)) for v in vulnerabilities if v
        ]

    return result


def _format_turns_for_analysis(turns: List[str]) -> str:
    """대화 턴을 분석용 텍스트로 변환"""
    return "\n".join(turns)


async def _send_webhook(payload: Dict[str, Any], webhook_url: Optional[str] = None) -> bool:
    """
    분석 결과를 VP2로 전송 (Webhook)

    Args:
        payload: 전송할 데이터
        webhook_url: 전송할 URL (없으면 설정에서 가져옴)

    Returns:
        성공 여부
    """
    url = webhook_url or SETTINGS.webhook_url

    if not url:
        logger.warning("[Webhook] URL이 설정되지 않음. 전송 스킵.")
        return False

    try:
        async with httpx.AsyncClient(timeout=SETTINGS.webhook_timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code in (200, 201, 202):
                logger.info(f"[Webhook] 전송 성공: {url}, status={response.status_code}")
                return True
            else:
                logger.error(f"[Webhook] 전송 실패: {url}, status={response.status_code}, body={response.text[:200]}")
                return False

    except httpx.TimeoutException:
        logger.error(f"[Webhook] 타임아웃: {url}")
        return False
    except Exception as e:
        logger.error(f"[Webhook] 전송 에러: {url}, error={e}")
        return False


async def _trigger_analysis_background(
    case_id: str,
    turns: List[str],
    judgement: Dict[str, Any],
    scenario: Optional[Dict[str, Any]] = None,
    victim_profile: Optional[Dict[str, Any]] = None,
):
    """백그라운드에서 분석 트리거 (선택적)"""

    # 이미 분석 완료된 케이스인지 확인
    if case_id in _analyzed_cases:
        logger.info(f"[Background] 이미 분석 완료된 케이스, 스킵: case_id={case_id}")
        return

    # 현재 분석 중인 케이스인지 확인
    if case_id in _analyzing_cases:
        logger.info(f"[Background] 이미 분석 진행 중인 케이스, 스킵: case_id={case_id}")
        return

    # 분석 시작 표시
    _analyzing_cases.add(case_id)
    analysis_id = f"analysis_{case_id}_{uuid.uuid4().hex[:8]}"

    try:
        logger.info(f"[Background] 분석 시작: case_id={case_id}")

        # 대화를 텍스트로 변환
        conversation_text = _format_turns_for_analysis(turns)

        # 컨텍스트 구성
        context = {
            "case_id": case_id,
            "judgement": judgement,
            "scenario": scenario or {},
            "victim_profile": victim_profile or {},
        }

        # 에이전트로 분석 실행
        agent = get_agent()
        request = AnalysisRequest(
            data=conversation_text,
            analysis_type="conversation",
            context=context,
        )
        result = agent.run(request)

        # 분석 결과 저장
        report_data = None
        techniques = []
        if result.report:
            report_data = result.report.model_dump() if hasattr(result.report, 'model_dump') else result.report
            techniques = result.report.techniques if hasattr(result.report, 'techniques') else []

        analysis_data = {
            "analysis_id": analysis_id,
            "case_id": case_id,
            "status": result.status,
            "input_turns": turns,
            "input_text": conversation_text,
            "sources": [s.model_dump() for s in result.sources] if result.sources else [],
            "sources_count": len(result.sources) if result.sources else 0,
            "techniques": [t.model_dump() if hasattr(t, 'model_dump') else t for t in techniques] if techniques else [],
            "report": report_data,
            "metadata": result.metadata or {},
            "error": result.error,
            "analyzed_at": datetime.utcnow().isoformat(),
        }

        # 메모리에 저장
        _analysis_results[analysis_id] = analysis_data

        logger.info(f"[Background] 분석 완료: case_id={case_id}, status={result.status}, analysis_id={analysis_id}")

        # Webhook으로 VP2에 전송
        if result.status == "success":
            webhook_payload = {
                "type": "analysis_complete",
                "case_id": case_id,
                "analysis_id": analysis_id,
                "report": report_data,
                "techniques": [t.model_dump() if hasattr(t, 'model_dump') else t for t in techniques] if techniques else [],
                "sources_count": len(result.sources) if result.sources else 0,
                "analyzed_at": datetime.utcnow().isoformat(),
            }
            await _send_webhook(webhook_payload)

            # 분석 완료 표시
            _analyzed_cases.add(case_id)

    except Exception as e:
        logger.error(f"[Background] 분석 실패: case_id={case_id}, error={e}")
        # 에러도 저장
        error_data = {
            "analysis_id": analysis_id,
            "case_id": case_id,
            "status": "error",
            "input_turns": turns,
            "error": str(e),
            "analyzed_at": datetime.utcnow().isoformat(),
        }
        _analysis_results[analysis_id] = error_data

        # 에러도 webhook으로 전송
        error_payload = {
            "type": "analysis_error",
            "case_id": case_id,
            "analysis_id": analysis_id,
            "error": str(e),
            "analyzed_at": datetime.utcnow().isoformat(),
        }
        await _send_webhook(error_payload)

    finally:
        # 분석 중 상태 해제
        _analyzing_cases.discard(case_id)


@router.post("/api/v1/judgements", response_model=JudgementResponse)
async def receive_judgement(
    request: JudgementRequest,
    background_tasks: BackgroundTasks,
    auto_analyze: bool = True
):
    """
    VP2로부터 판정+대화 데이터 수신

    - 감정 라벨이 제거된 순수 대화 내용
    - 판정 결과 (phishing, risk, evidence 등)
    - 선택적으로 백그라운드 분석 트리거
    - 자동 전처리: 한글/숫자만 유지, 영어/특수기호 제거

    **파라미터:**
    - auto_analyze: True면 수신 후 자동으로 분석 트리거
    """
    try:
        received_id = _generate_received_id("jdg")

        # ★ 전처리: 한글만 추출, 영어/특수기호 제거
        processed_turns = _preprocess_turns(request.turns)
        processed_judgement = _preprocess_judgement(request.judgement)

        logger.info(
            f"[VP2] 전처리: 원본 {len(request.turns)}턴 → 정제 {len(processed_turns)}턴"
        )

        # 수신 데이터 저장 (전처리된 데이터)
        stored_data = {
            "received_id": received_id,
            "case_id": request.case_id,
            "round_no": request.round_no,
            "turns": processed_turns,  # 전처리된 turns
            "turns_original_count": len(request.turns),
            "judgement": processed_judgement,  # 전처리된 judgement
            "scenario": request.scenario,
            "victim_profile": request.victim_profile,
            "source": request.source,
            "received_at": datetime.utcnow().isoformat(),
        }
        _received_judgements[received_id] = stored_data

        logger.info(
            f"[VP2] 판정 수신: case_id={request.case_id}, "
            f"round={request.round_no}, turns={len(processed_turns)}, "
            f"phishing={processed_judgement.get('phishing')}"
        )

        # 선택적 백그라운드 분석 (전처리된 데이터 사용)
        analysis_triggered = False
        if auto_analyze:
            background_tasks.add_task(
                _trigger_analysis_background,
                case_id=request.case_id,
                turns=processed_turns,  # 전처리된 turns
                judgement=processed_judgement,  # 전처리된 judgement
                scenario=request.scenario,
                victim_profile=request.victim_profile,
            )
            analysis_triggered = True
            logger.info(f"[VP2] 백그라운드 분석 트리거: case_id={request.case_id}")

        return JudgementResponse(
            ok=True,
            received_id=received_id,
            case_id=request.case_id,
            round_no=request.round_no,
            turns_count=len(request.turns),
            message="판정 데이터 수신 완료",
            analysis_triggered=analysis_triggered,
        )

    except Exception as e:
        logger.error(f"[VP2] 판정 수신 실패: {e}")
        raise HTTPException(status_code=500, detail=f"판정 수신 실패: {e}")


@router.post("/api/v1/conversations", response_model=ConversationResponse)
async def receive_conversation(request: ConversationRequest):
    """
    VP2로부터 대화 데이터 수신 (기존 호환)
    """
    try:
        received_id = _generate_received_id("conv")

        stored_data = {
            "received_id": received_id,
            "case_id": request.case_id,
            "round_no": request.round_no,
            "turns": request.turns,
            "scenario": request.scenario,
            "victim_profile": request.victim_profile,
            "guidance": request.guidance,
            "judgement": request.judgement,
            "received_at": datetime.utcnow().isoformat(),
        }
        _received_conversations[received_id] = stored_data

        logger.info(
            f"[VP2] 대화 수신: case_id={request.case_id}, "
            f"round={request.round_no}, turns={len(request.turns)}"
        )

        return ConversationResponse(
            ok=True,
            received_id=received_id,
            message="대화 데이터 수신 완료",
        )

    except Exception as e:
        logger.error(f"[VP2] 대화 수신 실패: {e}")
        raise HTTPException(status_code=500, detail=f"대화 수신 실패: {e}")


@router.post("/api/v1/methods/report", response_model=MethodReportResponse)
async def request_method_report(request: MethodReportRequest):
    """
    웹 서치 기반 새로운 수법 리포트 생성

    - 시나리오 유형과 키워드 기반 웹 검색
    - 최신 보이스피싱 수법 탐색
    """
    try:
        report_id = _generate_received_id("rpt")

        # 검색 쿼리 생성
        queries = [
            f"{request.scenario_type} 보이스피싱 최신 수법",
            f"{request.scenario_type} 사기 기법 2024",
        ]
        if request.keywords:
            queries.extend([f"보이스피싱 {kw}" for kw in request.keywords[:3]])

        logger.info(
            f"[VP2] 수법 리포트 요청: case_id={request.case_id}, "
            f"scenario={request.scenario_type}, queries={len(queries)}"
        )

        # 웹 검색 수행
        from app.services.searcher import WebSearcher
        searcher = WebSearcher()
        results = searcher.search(queries=queries, extract_content=True)

        # 결과 정리
        sources = list(set(r.url for r in results))
        new_methods = []

        for r in results[:5]:
            new_methods.append({
                "title": r.title,
                "description": r.content[:500] if r.content else "",
                "source": r.url,
            })

        # 요약 생성 (간단)
        summary = f"{request.scenario_type} 관련 최신 수법 {len(new_methods)}건 탐색 완료"

        return MethodReportResponse(
            report_id=report_id,
            new_methods=new_methods,
            sources=sources[:10],
            summary=summary,
            recommendations=[
                f"{request.scenario_type} 시나리오에서 발견된 취약점 활용 권장",
                "검색된 최신 수법 참고하여 공격 전략 보강",
            ],
        )

    except Exception as e:
        logger.error(f"[VP2] 수법 리포트 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"수법 리포트 생성 실패: {e}")


# ==================== 수신 데이터 조회 엔드포인트 ====================

@router.get("/api/v1/judgements")
async def list_received_judgements(limit: int = 50):
    """수신된 판정 목록 조회"""
    items = list(_received_judgements.values())[-limit:]
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }


@router.get("/api/v1/judgements/{received_id}")
async def get_received_judgement(received_id: str):
    """특정 판정 조회"""
    data = _received_judgements.get(received_id)
    if not data:
        raise HTTPException(status_code=404, detail="판정 데이터 없음")
    return {"ok": True, "data": data}


@router.get("/api/v1/conversations")
async def list_received_conversations(limit: int = 50):
    """수신된 대화 목록 조회"""
    items = list(_received_conversations.values())[-limit:]
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }


@router.get("/api/v1/conversations/{received_id}")
async def get_received_conversation(received_id: str):
    """특정 대화 조회"""
    data = _received_conversations.get(received_id)
    if not data:
        raise HTTPException(status_code=404, detail="대화 데이터 없음")
    return {"ok": True, "data": data}


# ==================== 분석 결과 조회 엔드포인트 ====================

@router.get("/api/v1/analysis")
async def list_analysis_results(limit: int = 50):
    """분석 결과 목록 조회"""
    items = list(_analysis_results.values())[-limit:]
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }


@router.get("/api/v1/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """특정 분석 결과 조회"""
    data = _analysis_results.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="분석 결과 없음")
    return {"ok": True, "data": data}


@router.get("/api/v1/analysis/case/{case_id}")
async def get_analysis_by_case(case_id: str):
    """case_id로 분석 결과 조회"""
    results = [v for v in _analysis_results.values() if v.get("case_id") == case_id]
    if not results:
        raise HTTPException(status_code=404, detail="해당 케이스의 분석 결과 없음")
    return {"ok": True, "count": len(results), "items": results}


@router.delete("/api/v1/analysis/case/{case_id}/reset")
async def reset_case_analysis(case_id: str):
    """
    케이스의 분석 상태 리셋
    - 해당 케이스를 다시 분석할 수 있도록 상태 초기화
    """
    was_analyzed = case_id in _analyzed_cases
    was_analyzing = case_id in _analyzing_cases

    _analyzed_cases.discard(case_id)
    _analyzing_cases.discard(case_id)

    logger.info(f"[Reset] 케이스 분석 상태 리셋: case_id={case_id}, was_analyzed={was_analyzed}, was_analyzing={was_analyzing}")

    return {
        "ok": True,
        "message": f"케이스 {case_id}의 분석 상태가 리셋되었습니다.",
        "was_analyzed": was_analyzed,
        "was_analyzing": was_analyzing,
    }


@router.get("/api/v1/analysis/status")
async def get_analysis_status():
    """
    전체 분석 상태 조회
    - 분석 완료된 케이스 목록
    - 현재 분석 중인 케이스 목록
    """
    return {
        "ok": True,
        "analyzed_cases": list(_analyzed_cases),
        "analyzing_cases": list(_analyzing_cases),
        "analyzed_count": len(_analyzed_cases),
        "analyzing_count": len(_analyzing_cases),
    }
