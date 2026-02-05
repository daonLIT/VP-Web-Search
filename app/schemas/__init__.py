# app/schemas/__init__.py
"""
Pydantic 스키마 정의
- 범용적인 입력/출력 구조
- 어떤 데이터든 받아서 분석 가능
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """분석 유형"""
    CONVERSATION = "conversation"      # 대화 분석
    DOCUMENT = "document"              # 문서 분석
    PROFILE = "profile"                # 프로필 분석
    SCENARIO = "scenario"              # 시나리오 분석
    CUSTOM = "custom"                  # 커스텀


class SearchDepth(str, Enum):
    """검색 깊이"""
    BASIC = "basic"
    ADVANCED = "advanced"


# ==================== 입력 스키마 ====================

class AnalysisRequest(BaseModel):
    """
    범용 분석 요청 스키마
    - 어떤 형태의 데이터든 받을 수 있음
    """
    # 필수: 분석할 데이터 (텍스트 또는 구조화된 데이터)
    data: Union[str, Dict[str, Any], List[Any]] = Field(
        ...,
        description="분석할 데이터. 텍스트, JSON 객체, 배열 모두 가능"
    )

    # 선택: 분석 유형 힌트
    analysis_type: Optional[AnalysisType] = Field(
        default=None,
        description="분석 유형 힌트. 없으면 자동 감지"
    )

    # 선택: 추가 컨텍스트
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 컨텍스트 정보 (피해자 프로필, 시나리오 등)"
    )

    # 선택: 검색 설정
    search_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="웹 검색 설정 (max_results, search_depth 등)"
    )

    # 선택: 출력 설정
    output_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="출력 형식 설정"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "data": "피해자는 30대 직장인 남성이다. 검찰을 사칭한 전화를 받았고...",
                    "analysis_type": "conversation",
                    "context": {"scenario": "검경 사칭"}
                },
                {
                    "data": {
                        "victim_age": 65,
                        "victim_occupation": "퇴직자",
                        "call_transcript": "여보세요, 검찰청입니다..."
                    },
                    "analysis_type": "profile"
                },
                {
                    "data": ["대화내용1", "대화내용2", "대화내용3"],
                    "analysis_type": "conversation"
                }
            ]
        }


# ==================== 내부 처리용 스키마 ====================

class ExtractedProfile(BaseModel):
    """추출된 프로필 정보"""
    age_group: Optional[str] = None
    occupation: Optional[str] = None
    gender: Optional[str] = None
    characteristics: List[str] = Field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None


class AnalysisSummary(BaseModel):
    """분석 요약"""
    summary: str
    key_points: List[str] = Field(default_factory=list)
    extracted_profile: Optional[ExtractedProfile] = None
    detected_scenario: Optional[str] = None
    vulnerability_areas: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """검색 결과"""
    title: str
    url: str
    content: str
    query: str
    content_type: str = "snippet"  # snippet, full_crawled
    relevance_score: Optional[float] = None


class GeneratedTechnique(BaseModel):
    """생성된 기법"""
    name: str
    description: str
    application: str
    expected_effect: str
    fit_score: float = Field(ge=0.0, le=1.0)
    source_info: Optional[List[str]] = None


# ==================== 출력 스키마 ====================

class AnalysisReport(BaseModel):
    """분석 리포트"""
    summary: str
    profile: Optional[ExtractedProfile] = None
    vulnerabilities: List[str] = Field(default_factory=list)
    techniques: List[GeneratedTechnique] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    implementation_guide: Optional[str] = None


class AnalysisResponse(BaseModel):
    """
    범용 분석 응답 스키마
    """
    status: str = Field(default="success")

    # 분석 결과
    report: Optional[AnalysisReport] = None

    # 메타데이터
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # 원본 데이터 참조
    sources: List[SearchResult] = Field(default_factory=list)

    # 에러 정보 (실패 시)
    error: Optional[str] = None

    # 타임스탬프
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    services: Dict[str, str]
    version: str


# ==================== VP2 연동 스키마 ====================

class JudgementRequest(BaseModel):
    """
    VP2에서 전송하는 판정+대화 데이터
    - 감정 라벨이 제거된 순수 대화 내용
    - 판정 결과 포함
    """
    case_id: str = Field(..., description="케이스 ID")
    round_no: int = Field(..., ge=1, description="라운드 번호")
    turns: List[Dict[str, Any]] = Field(
        ...,
        description="대화 턴 목록 (감정 라벨 제거됨)"
    )
    judgement: Dict[str, Any] = Field(
        ...,
        description="판정 결과 (phishing, risk, evidence 등)"
    )
    scenario: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="시나리오 정보"
    )
    victim_profile: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="피해자 프로필"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="전송 시각"
    )
    source: Optional[str] = Field(
        default="judgement_trigger",
        description="전송 출처"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "550e8400-e29b-41d4-a716-446655440000",
                "round_no": 1,
                "turns": [
                    {"role": "offender", "text": "안녕하세요, 검찰청입니다.", "turn_index": 0},
                    {"role": "victim", "text": "네, 안녕하세요.", "turn_index": 1}
                ],
                "judgement": {
                    "phishing": False,
                    "evidence": "피해자가 의심하고 있음",
                    "risk": {"score": 25, "level": "low", "rationale": "..."},
                    "victim_vulnerabilities": ["권위에 약함"]
                },
                "scenario": {"type": "검찰사칭"},
                "victim_profile": {"age_group": "60대"}
            }
        }


class JudgementResponse(BaseModel):
    """판정 수신 응답"""
    ok: bool = True
    received_id: str = Field(..., description="수신 ID")
    case_id: str = Field(..., description="케이스 ID")
    round_no: int = Field(..., description="라운드 번호")
    turns_count: int = Field(..., description="수신된 턴 수")
    message: str = Field(default="판정 데이터 수신 완료")
    analysis_triggered: bool = Field(
        default=False,
        description="분석 트리거 여부"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationRequest(BaseModel):
    """
    VP2에서 전송하는 대화 데이터 (기존 호환)
    """
    case_id: str = Field(..., description="케이스 ID")
    round_no: int = Field(..., ge=1, description="라운드 번호")
    turns: List[Dict[str, Any]] = Field(..., description="대화 턴 목록")
    scenario: Dict[str, Any] = Field(default_factory=dict)
    victim_profile: Dict[str, Any] = Field(default_factory=dict)
    guidance: Dict[str, Any] = Field(default_factory=dict)
    judgement: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: Optional[str] = Field(default=None)


class ConversationResponse(BaseModel):
    """대화 수신 응답"""
    ok: bool = True
    received_id: str
    message: Optional[str] = None


class MethodReportRequest(BaseModel):
    """수법 리포트 요청"""
    case_id: str = Field(..., description="케이스 ID")
    scenario_type: str = Field(..., description="시나리오 유형")
    keywords: List[str] = Field(default_factory=list)
    conversation_summary: Optional[str] = Field(default=None)
    request_time: Optional[str] = Field(default=None)


class MethodReportResponse(BaseModel):
    """수법 리포트 응답"""
    report_id: str
    new_methods: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Export
__all__ = [
    "AnalysisType",
    "SearchDepth",
    "AnalysisRequest",
    "ExtractedProfile",
    "AnalysisSummary",
    "SearchResult",
    "GeneratedTechnique",
    "AnalysisReport",
    "AnalysisResponse",
    "HealthResponse",
    # VP2 연동
    "JudgementRequest",
    "JudgementResponse",
    "ConversationRequest",
    "ConversationResponse",
    "MethodReportRequest",
    "MethodReportResponse",
]
