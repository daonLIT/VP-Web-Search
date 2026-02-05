# app/services/analyzer.py
"""
데이터 분석 서비스
- 어떤 형태의 데이터든 받아서 요약/분석
- 검색 쿼리 생성
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_openai import ChatOpenAI

from app.schemas import (
    AnalysisType,
    ExtractedProfile,
    AnalysisSummary,
)
from app.config import SETTINGS

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    범용 데이터 분석기
    - 텍스트, JSON, 리스트 등 어떤 형태든 분석
    - 프로필 추출, 요약, 검색 쿼리 생성
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or SETTINGS.model_name
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.3,
            timeout=60,
            max_retries=2,
        )

    def analyze(
        self,
        data: Union[str, Dict[str, Any], List[Any]],
        analysis_type: Optional[AnalysisType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AnalysisSummary:
        """
        데이터를 분석하여 요약 및 검색 쿼리 생성

        Args:
            data: 분석할 데이터 (텍스트, 딕셔너리, 리스트)
            analysis_type: 분석 유형 힌트
            context: 추가 컨텍스트

        Returns:
            AnalysisSummary: 분석 결과
        """
        # 1. 데이터를 텍스트로 정규화
        normalized_text = self._normalize_data(data)

        # 2. 분석 유형 감지 (힌트가 없으면)
        if analysis_type is None:
            analysis_type = self._detect_analysis_type(normalized_text)

        # 3. LLM으로 분석 수행
        analysis_result = self._perform_analysis(
            text=normalized_text,
            analysis_type=analysis_type,
            context=context or {},
        )

        return analysis_result

    def _normalize_data(self, data: Union[str, Dict[str, Any], List[Any]]) -> str:
        """데이터를 분석 가능한 텍스트로 정규화"""
        if isinstance(data, str):
            return data.strip()

        if isinstance(data, dict):
            # 주요 필드 추출
            text_parts = []

            # 일반적인 텍스트 필드들
            text_fields = [
                "text", "content", "message", "summary",
                "conversation", "transcript", "description",
                "conversation_summary", "call_transcript",
            ]

            for field in text_fields:
                if field in data and data[field]:
                    text_parts.append(f"[{field}]\n{data[field]}")

            # 나머지 필드들
            for key, value in data.items():
                if key not in text_fields and value:
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"[{key}]: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"[{key}]: {', '.join(map(str, value))}")

            return "\n\n".join(text_parts) if text_parts else json.dumps(data, ensure_ascii=False)

        if isinstance(data, list):
            # 리스트의 각 항목을 텍스트로
            parts = []
            for i, item in enumerate(data, 1):
                if isinstance(item, str):
                    parts.append(f"[{i}] {item}")
                elif isinstance(item, dict):
                    parts.append(f"[{i}] {json.dumps(item, ensure_ascii=False)}")
                else:
                    parts.append(f"[{i}] {str(item)}")
            return "\n".join(parts)

        return str(data)

    def _detect_analysis_type(self, text: str) -> AnalysisType:
        """텍스트 내용을 기반으로 분석 유형 감지"""
        text_lower = text.lower()

        # 키워드 기반 감지
        conversation_keywords = ["대화", "전화", "통화", "말했", "대답", "여보세요"]
        profile_keywords = ["나이", "직업", "성별", "연령", "age", "occupation"]
        scenario_keywords = ["시나리오", "수법", "사칭", "scenario"]

        if any(kw in text_lower for kw in conversation_keywords):
            return AnalysisType.CONVERSATION
        if any(kw in text_lower for kw in profile_keywords):
            return AnalysisType.PROFILE
        if any(kw in text_lower for kw in scenario_keywords):
            return AnalysisType.SCENARIO

        return AnalysisType.CUSTOM

    def _perform_analysis(
        self,
        text: str,
        analysis_type: AnalysisType,
        context: Dict[str, Any],
    ) -> AnalysisSummary:
        """LLM을 사용하여 분석 수행"""
        context_str = json.dumps(context, ensure_ascii=False) if context else "없음"

        prompt = f"""
당신은 데이터 분석 전문가입니다.
아래 데이터를 분석하여 구조화된 정보를 추출하세요.

[분석 유형]: {analysis_type.value}
[추가 컨텍스트]: {context_str}

[분석할 데이터]
{text[:4000]}

다음 JSON 형식으로 출력하세요:
{{
    "summary": "데이터의 핵심 내용을 2-3문장으로 요약",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2", ...],
    "extracted_profile": {{
        "age_group": "연령대 (예: 30대, 60대 이상) 또는 null",
        "occupation": "직업 또는 null",
        "gender": "성별 또는 null",
        "characteristics": ["특징1", "특징2"]
    }},
    "detected_scenario": "감지된 시나리오 유형 (예: 검경 사칭, 대출 사기) 또는 null",
    "vulnerability_areas": [
        "취약 영역 1 (예: 권위에 대한 복종)",
        "취약 영역 2 (예: 금융 불안)",
        ...
    ],
    "search_queries": [
        "관련 정보 검색을 위한 쿼리 1",
        "관련 정보 검색을 위한 쿼리 2",
        ... (5-8개)
    ]
}}

규칙:
1. search_queries는 '보이스피싱', '사기' 등의 직접적인 단어를 피하고 심리학/사회학/마케팅 관점의 검색어로 생성
2. 데이터에서 명확히 파악되지 않는 정보는 null로 표시
3. JSON만 출력 (마크다운 코드블록 없이)
""".strip()

        try:
            response = self.llm.invoke(prompt).content.strip()

            # JSON 파싱
            json_text = self._extract_json(response)
            result = json.loads(json_text)

            # AnalysisSummary로 변환
            profile_data = result.get("extracted_profile", {})
            profile = ExtractedProfile(
                age_group=profile_data.get("age_group"),
                occupation=profile_data.get("occupation"),
                gender=profile_data.get("gender"),
                characteristics=profile_data.get("characteristics", []),
            ) if profile_data else None

            return AnalysisSummary(
                summary=result.get("summary", ""),
                key_points=result.get("key_points", []),
                extracted_profile=profile,
                detected_scenario=result.get("detected_scenario"),
                vulnerability_areas=result.get("vulnerability_areas", []),
                search_queries=result.get("search_queries", []),
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # 폴백: 기본 분석
            return AnalysisSummary(
                summary=text[:200] + "..." if len(text) > 200 else text,
                key_points=[],
                search_queries=self._generate_fallback_queries(text),
            )

    def _extract_json(self, text: str) -> str:
        """텍스트에서 JSON 추출"""
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def _generate_fallback_queries(self, text: str) -> List[str]:
        """폴백 검색 쿼리 생성"""
        # 간단한 키워드 추출
        keywords = []
        if "30대" in text or "삼십대" in text:
            keywords.extend(["30대 심리 특성", "밀레니얼 세대 가치관"])
        if "60대" in text or "육십대" in text or "노인" in text:
            keywords.extend(["고령층 심리", "노년기 불안"])
        if "직장인" in text:
            keywords.extend(["직장인 스트레스", "회사원 고민"])
        if "퇴직" in text:
            keywords.extend(["퇴직자 심리", "은퇴 후 불안"])

        return keywords if keywords else ["심리적 취약점", "신뢰 형성 요인"]

    def generate_search_queries(
        self,
        question: str,
        profile: Optional[ExtractedProfile] = None,
        max_queries: int = 5,
    ) -> List[str]:
        """
        특정 질문에 대한 웹 검색 쿼리 생성

        Args:
            question: 답을 찾고자 하는 질문
            profile: 프로필 정보 (컨텍스트)
            max_queries: 최대 쿼리 개수

        Returns:
            검색 쿼리 리스트
        """
        profile_str = ""
        if profile:
            profile_str = f"\n프로필: {profile.age_group or ''} {profile.occupation or ''}"

        prompt = f"""
질문: "{question}"{profile_str}

이 질문에 답하기 위한 웹 검색 쿼리 {max_queries}개를 생성하세요.

규칙:
- "보이스피싱", "사기", "피싱" 등의 단어 사용 금지
- 심리학, 사회학, 마케팅, 소비자 행동 관점의 검색어
- 각 쿼리는 10자 이내로 짧게
- JSON 배열만 출력: ["쿼리1", "쿼리2", ...]
""".strip()

        try:
            response = self.llm.invoke(prompt).content.strip()
            json_text = self._extract_json(response)
            queries = json.loads(json_text)

            if isinstance(queries, list):
                return queries[:max_queries]

        except Exception as e:
            logger.error(f"Query generation failed: {e}")

        # 폴백
        fallback = [f"{question[:10]} 심리", "취약점 요인"]
        if profile and profile.age_group:
            fallback.append(f"{profile.age_group} 특성")
        return fallback
