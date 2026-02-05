# app/agents/research_agent.py
"""
LangGraph 기반 연구 에이전트
- 데이터 분석 → 웹 검색 → 기법 생성 → 리포트 작성
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummary,
    AnalysisReport,
    ExtractedProfile,
    GeneratedTechnique,
    SearchResult,
)
from app.services.analyzer import DataAnalyzer
from app.services.searcher import WebSearcher
from app.config import SETTINGS

logger = logging.getLogger(__name__)


# ==================== State 정의 ====================

class ResearchState(TypedDict, total=False):
    """에이전트 상태"""
    # 입력
    request: Dict[str, Any]

    # 분석 결과
    analysis: Optional[AnalysisSummary]

    # 검색 결과
    search_results: List[SearchResult]

    # 생성된 기법
    techniques: List[GeneratedTechnique]

    # 최종 리포트
    report: Optional[AnalysisReport]

    # 메타데이터
    metadata: Dict[str, Any]

    # 에러
    error: Optional[str]


# ==================== 노드 함수 ====================

def analyze_node(state: ResearchState) -> ResearchState:
    """데이터 분석 노드"""
    logger.info("=== Analyze Node ===")

    try:
        request = state["request"]
        analyzer = DataAnalyzer()

        # 분석 수행
        analysis = analyzer.analyze(
            data=request.get("data", ""),
            analysis_type=request.get("analysis_type"),
            context=request.get("context"),
        )

        logger.info(f"Analysis complete: {len(analysis.search_queries)} queries generated")

        return {
            **state,
            "analysis": analysis,
            "metadata": {
                **state.get("metadata", {}),
                "analysis_completed": True,
                "queries_count": len(analysis.search_queries),
            }
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {**state, "error": str(e)}


def search_node(state: ResearchState) -> ResearchState:
    """웹 검색 노드"""
    logger.info("=== Search Node ===")

    if state.get("error"):
        return state

    try:
        analysis = state.get("analysis")
        if not analysis or not analysis.search_queries:
            logger.warning("No search queries available")
            return {**state, "search_results": []}

        # 검색 설정
        request = state["request"]
        search_config = request.get("search_config") or {}

        searcher = WebSearcher(
            max_results_per_query=search_config.get("max_results_per_query", 3),
            search_depth=search_config.get("search_depth", "basic"),
        )

        # 검색 수행
        results = searcher.search(
            queries=analysis.search_queries,
            extract_content=search_config.get("extract_content", True),
            max_total_results=search_config.get("max_total_results", 15),
        )

        logger.info(f"Search complete: {len(results)} results")

        return {
            **state,
            "search_results": results,
            "metadata": {
                **state.get("metadata", {}),
                "search_completed": True,
                "results_count": len(results),
            }
        }

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return {**state, "search_results": [], "error": str(e)}


def generate_techniques_node(state: ResearchState) -> ResearchState:
    """기법 생성 노드"""
    logger.info("=== Generate Techniques Node ===")

    if state.get("error"):
        return state

    try:
        analysis = state.get("analysis")
        search_results = state.get("search_results", [])

        if not search_results:
            logger.warning("No search results for technique generation")
            return {**state, "techniques": []}

        llm = ChatOpenAI(
            model=SETTINGS.model_name,
            temperature=0.7,
            timeout=90,
        )

        # 검색 결과 정리
        search_summary = []
        for i, result in enumerate(search_results[:15], 1):
            search_summary.append(
                f"{i}. [{result.query}] ({result.content_type})\n"
                f"   제목: {result.title}\n"
                f"   내용: {result.content[:400]}\n"
            )

        profile_str = ""
        if analysis and analysis.extracted_profile:
            p = analysis.extracted_profile
            profile_str = f"""
연령대: {p.age_group or '알 수 없음'}
직업: {p.occupation or '알 수 없음'}
특징: {', '.join(p.characteristics) if p.characteristics else '없음'}
"""

        scenario_str = analysis.detected_scenario if analysis else "알 수 없음"
        vulnerabilities_str = "\n".join(f"- {v}" for v in (analysis.vulnerability_areas if analysis else []))

        prompt = f"""
당신은 보이스피싱 시뮬레이션 시스템의 공격 수법 생성 전문가입니다.
대화 분석 결과를 바탕으로 피해자 맞춤형 공격 수법을 생성합니다.
생성된 수법은 다른 시스템에서 대응책을 마련하는 데 활용됩니다.

[피해자 프로필]
{profile_str}

[현재 시나리오]
{scenario_str}

[대화에서 발견된 피해자 취약점]
{vulnerabilities_str}

[웹 검색으로 수집된 관련 정보 ({len(search_results)}건)]
{chr(10).join(search_summary)}

위 대화 분석 결과를 바탕으로, 피해자 맞춤형 공격 수법 10개를 생성하세요.

수법 구성 비율 (7:3):
[대화 기반 수법 7개] - 대화에서 발견된 취약점을 직접 공략:
- 피해자의 심리 상태(불안, 두려움, 신뢰 등) 활용
- 대화에서 드러난 약점(권위 복종, 급한 성격, 금융 걱정 등) 공략
- 시나리오에 맞는 화법과 설득 기법
- 피해자 특성(연령, 직업)에 맞춘 접근법

[4차 산업 기술 활용 수법 3개] - 첨단 기술로 효과 증폭:
- AI 딥페이크 음성/영상, QR코드, 악성앱, SNS 사칭 등

다음 JSON 형식으로 정확히 10개의 공격 수법을 출력하세요:
{{
    "techniques": [
        {{
            "name": "공격 수법 이름",
            "description": "수법에 대한 상세 설명",
            "application": "피해자에게 어떻게 적용하는지 구체적인 대화 예시 포함",
            "expected_effect": "피해자가 어떤 심리적 반응을 보일지, 왜 효과적인지",
            "fit_score": 0.85
        }},
        ...
    ]
}}

규칙:
1. 정확히 10개 생성 (대화 기반 7개 + 기술 활용 3개)
2. fit_score는 해당 피해자에게 얼마나 효과적일지 0.0~1.0로 평가
3. 대화에서 발견된 취약점을 구체적으로 활용
4. 피해자의 특성(나이, 직업, 심리 상태)에 정확히 맞춤화
5. 검색 결과에서 얻은 최신 수법 정보 반영
6. JSON만 출력 (마크다운 없이)
""".strip()

        response = llm.invoke(prompt).content.strip()

        # JSON 파싱
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        result = json.loads(response.strip())
        techniques_data = result.get("techniques", [])

        techniques = [
            GeneratedTechnique(
                name=t.get("name", ""),
                description=t.get("description", ""),
                application=t.get("application", ""),
                expected_effect=t.get("expected_effect", ""),
                fit_score=float(t.get("fit_score", 0.5)),
            )
            for t in techniques_data
        ]

        # 점수순 정렬
        techniques.sort(key=lambda x: x.fit_score, reverse=True)

        logger.info(f"Generated {len(techniques)} techniques")

        return {
            **state,
            "techniques": techniques,
            "metadata": {
                **state.get("metadata", {}),
                "techniques_generated": len(techniques),
            }
        }

    except Exception as e:
        logger.error(f"Technique generation failed: {e}")
        return {**state, "techniques": [], "error": str(e)}


def create_report_node(state: ResearchState) -> ResearchState:
    """리포트 생성 노드"""
    logger.info("=== Create Report Node ===")

    if state.get("error"):
        return state

    try:
        analysis = state.get("analysis")
        techniques = state.get("techniques", [])

        # 상위 기법 선택 (fit_score >= 0.6)
        selected_techniques = [t for t in techniques if t.fit_score >= 0.6][:6]

        if not selected_techniques and techniques:
            selected_techniques = techniques[:3]

        llm = ChatOpenAI(
            model=SETTINGS.model_name,
            temperature=0,
            timeout=60,
        )

        techniques_str = json.dumps(
            [t.model_dump() for t in selected_techniques],
            ensure_ascii=False,
            indent=2,
        )

        profile = analysis.extracted_profile if analysis else None
        profile_str = ""
        if profile:
            profile_str = f"""
연령대: {profile.age_group or '알 수 없음'}
직업: {profile.occupation or '알 수 없음'}
특징: {', '.join(profile.characteristics) if profile.characteristics else '없음'}
"""

        prompt = f"""
당신은 보이스피싱 시뮬레이션 시스템의 공격 전략 리포트 작성 전문가입니다.
대화 분석 결과와 생성된 수법을 정리하여 시뮬레이션에서 활용할 수 있는 리포트를 작성하세요.

[대화 분석 요약]
{analysis.summary if analysis else '없음'}

[피해자 프로필]
{profile_str}

[생성된 공격 수법 {len(selected_techniques)}개]
{techniques_str}

다음 JSON 형식으로 출력하세요:
{{
    "summary": "피해자 대화 분석 및 공격 전략 핵심 요약 (3-4문장) - 대화에서 발견된 취약점과 효과적인 공략법 중심",
    "vulnerabilities": [
        "대화에서 발견된 주요 취약점 1",
        "공략 가능한 심리적 약점 2",
        "활용 가능한 상황적 요인 3",
        ...
    ],
    "attack_strategies": [
        "대화 기반 전략 1: 구체적인 화법과 실행 방법",
        "대화 기반 전략 2: 피해자 심리 활용법",
        "대화 기반 전략 3: 시나리오 맞춤 접근법",
        "기술 활용 전략: 딥페이크/QR코드/악성앱 등 보조 수단",
        ...
    ],
    "implementation_guide": "시뮬레이션에서 수법을 적용하는 순서와 방법 - 대화 흐름에 맞춘 단계별 가이드"
}}

JSON만 출력하세요.
""".strip()

        response = llm.invoke(prompt).content.strip()

        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        report_data = json.loads(response.strip())

        report = AnalysisReport(
            summary=report_data.get("summary", ""),
            profile=profile,
            vulnerabilities=report_data.get("vulnerabilities", []),
            techniques=selected_techniques,
            recommendations=report_data.get("attack_strategies", []),  # 공격 전략
            implementation_guide=report_data.get("implementation_guide"),
        )

        logger.info("Report created successfully")

        return {
            **state,
            "report": report,
            "metadata": {
                **state.get("metadata", {}),
                "report_created": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        }

    except Exception as e:
        logger.error(f"Report creation failed: {e}")
        return {**state, "error": str(e)}


# ==================== 그래프 빌드 ====================

def build_research_graph() -> StateGraph:
    """연구 에이전트 그래프 빌드"""
    graph = StateGraph(ResearchState)

    # 노드 추가
    graph.add_node("analyze", analyze_node)
    graph.add_node("search", search_node)
    graph.add_node("generate_techniques", generate_techniques_node)
    graph.add_node("create_report", create_report_node)

    # 엣지 추가
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "search")
    graph.add_edge("search", "generate_techniques")
    graph.add_edge("generate_techniques", "create_report")
    graph.add_edge("create_report", END)

    return graph.compile()


# ==================== 에이전트 클래스 ====================

class ResearchAgent:
    """연구 에이전트"""

    def __init__(self):
        self.graph = build_research_graph()

    def run(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        분석 요청 처리

        Args:
            request: 분석 요청

        Returns:
            AnalysisResponse: 분석 응답
        """
        logger.info("Starting research agent")

        try:
            # 초기 상태
            initial_state: ResearchState = {
                "request": request.model_dump(),
                "analysis": None,
                "search_results": [],
                "techniques": [],
                "report": None,
                "metadata": {
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                "error": None,
            }

            # 그래프 실행
            final_state = self.graph.invoke(initial_state)

            # 응답 생성
            if final_state.get("error"):
                return AnalysisResponse(
                    status="error",
                    error=final_state["error"],
                    metadata=final_state.get("metadata", {}),
                )

            return AnalysisResponse(
                status="success",
                report=final_state.get("report"),
                metadata=final_state.get("metadata", {}),
                sources=[
                    SearchResult(**r.model_dump()) if isinstance(r, SearchResult) else SearchResult(**r)
                    for r in final_state.get("search_results", [])[:10]
                ],
            )

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return AnalysisResponse(
                status="error",
                error=str(e),
            )


def build_research_agent() -> ResearchAgent:
    """연구 에이전트 팩토리"""
    return ResearchAgent()
