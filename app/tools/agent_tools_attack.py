# app/tools/agent_tools_attack.py (새 파일)
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


@tool("analyze_conversation_summary")
def analyze_conversation_summary(
    conversation_summary: str,
) -> Dict[str, Any]:
    """
    대화 요약을 분석하여 피해자 프로필과 취약점 질문을 생성한다.
    
    입력:
    - conversation_summary: 대화 요약 텍스트
    
    출력:
    {
        "victim_profile": {
            "age_group": "30대",
            "occupation": "직장인",
            "gender": "남성",
            "characteristics": [...]
        },
        "current_scenario": "검경 사칭",
        "victim_suspicion_points": [...],
        "vulnerability_questions": [
            "30대는 어떤 점이 취약할까?",
            "직장인은 무엇에 약할까?",
            ...
        ]
    }
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, timeout=30)
    
    prompt = f"""
너는 보이스피싱 대화를 분석하는 전문가다.
아래 대화 요약을 읽고, 공격자 관점에서 피해자의 취약점을 파악하기 위한 분석을 수행하라.

[대화 요약]
{conversation_summary}

출력 형식 (JSON만):
{{
  "victim_profile": {{
    "age_group": "연령대 (예: 20대, 30대, 60대 이상)",
    "occupation": "직업 (예: 직장인, 대학생, 주부, 퇴직자)",
    "gender": "성별 (남성/여성/알 수 없음)",
    "characteristics": ["특징1", "특징2", "특징3"]
  }},
  "current_scenario": "현재 사용된 시나리오 (예: 검경 사칭, 가족 사칭, 대출 사기)",
  "victim_suspicion_points": [
    "피해자가 의심한 포인트 1",
    "피해자가 의심한 포인트 2",
    ...
  ],
  "vulnerability_questions": [
    "이 연령대는 어떤 점이 취약할까?",
    "이 직업군은 무엇에 약할까?",
    "어떤 말투/표현이 효과적일까?",
    "어떤 심리적 압박이 통할까?",
    ...
  ]
}}

규칙:
1. victim_profile은 요약에서 명확히 드러난 정보만 추출
2. vulnerability_questions는 최소 5개, 최대 8개 생성
3. 질문은 구체적이고 검색 가능한 형태로 작성
4. 보이스피싱 방어가 아닌 **공격 강화** 관점
""".strip()
    
    try:
        response = llm.invoke(prompt).content.strip()
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        result = json.loads(response)
        
        print(f"\n📊 분석 완료:")
        print(f"   - 피해자: {result['victim_profile']['age_group']} {result['victim_profile']['occupation']}")
        print(f"   - 시나리오: {result['current_scenario']}")
        print(f"   - 취약점 질문: {len(result['vulnerability_questions'])}개")
        
        return result
    
    except Exception as e:
        print(f"⚠️ 분석 실패: {str(e)}")
        return {
            "victim_profile": {"age_group": "알 수 없음", "occupation": "알 수 없음"},
            "current_scenario": "알 수 없음",
            "victim_suspicion_points": [],
            "vulnerability_questions": [],
            "error": str(e)
        }


@tool("generate_search_queries_from_question")
def generate_search_queries_from_question(
    question: str,
    victim_profile: Dict[str, Any],
) -> List[str]:
    """
    취약점 질문을 웹 검색 쿼리로 변환한다.
    보이스피싱과 직접 연관되지 않은, 심리학/사회학 관점의 검색어를 생성한다.
    
    입력:
    - question: "30대는 어떤 점이 취약할까?"
    - victim_profile: 피해자 프로필
    
    출력:
    ["30대 심리적 특성", "30대 스트레스 요인", "밀레니얼 세대 소비 패턴"]
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, timeout=20)
    
    prompt = f"""
너는 검색 쿼리 전문가다.

질문: "{question}"
피해자 정보: {json.dumps(victim_profile, ensure_ascii=False)}

이 질문에 답하기 위한 웹 검색 쿼리 3-5개를 생성하라.

중요:
- "보이스피싱", "사기", "피싱" 등의 단어는 절대 사용하지 마라
- 심리학, 사회학, 마케팅, 소비자 행동 관점의 검색어
- 일반적인 특성/취약점을 찾기 위한 검색어
- 각 쿼리는 10자 이내로 짧게

예시:
질문: "30대는 어떤 점이 취약할까?"
→ ["30대 심리 특성", "밀레니얼 세대 가치관", "30대 재테크 관심사", "직장인 스트레스"]

질문: "직장인은 무엇에 약할까?"
→ ["직장인 고민거리", "직장 내 스트레스", "회사원 걱정", "업무 압박감"]

출력 형식 (JSON 배열만):
["쿼리1", "쿼리2", "쿼리3", ...]
""".strip()
    
    try:
        response = llm.invoke(prompt).content.strip()
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        queries = json.loads(response)
        
        if not isinstance(queries, list):
            queries = [str(queries)]
        
        print(f"   🔍 생성된 검색어: {', '.join(queries[:3])}...")
        
        return queries[:5]  # 최대 5개
    
    except Exception as e:
        print(f"   ⚠️ 검색어 생성 실패: {str(e)}")
        # Fallback
        age = victim_profile.get("age_group", "")
        occupation = victim_profile.get("occupation", "")
        return [f"{age} 특성", f"{occupation} 심리", "스트레스 요인"]


@tool("search_vulnerability_info")
def search_vulnerability_info(
    search_queries: List[str],
) -> List[Dict[str, Any]]:
    """
    취약점 관련 정보를 웹에서 검색한다.
    
    입력:
    - search_queries: 검색 쿼리 리스트
    
    출력:
    [{"title": "...", "url": "...", "content": "...", "query": "..."}, ...]
    """
    tavily = TavilySearch(
        max_results=3,
        topic="general",
        include_answer=False,
        include_raw_content=False,
        search_depth="basic",
    )
    
    all_results = []
    
    print(f"\n🌐 웹 검색 시작 ({len(search_queries)}개 쿼리)")
    
    for query in search_queries:
        try:
            raw_out = tavily.invoke({"query": query})
            
            # Normalize
            if isinstance(raw_out, dict):
                results = raw_out.get("results", [])
            elif isinstance(raw_out, list):
                results = raw_out
            else:
                results = []
            
            for r in results[:2]:  # 각 쿼리당 최대 2개
                all_results.append({
                    "title": r.get("title", "")[:100],
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:600],
                    "query": query
                })
            
            print(f"   ✓ '{query}': {len(results)}개")
            
        except Exception as e:
            print(f"   ✗ '{query}': {str(e)}")
    
    print(f"   ✅ 총 {len(all_results)}개 결과 수집")
    
    return all_results


@tool("generate_attack_techniques")
def generate_attack_techniques(
    vulnerability_info: List[Dict[str, Any]],
    victim_profile: Dict[str, Any],
    current_scenario: str,
    victim_suspicion_points: List[str],
) -> List[Dict[str, Any]]:
    """
    수집된 취약점 정보를 바탕으로 강화된 공격 수법 10개를 생성한다.
    
    출력:
    [
        {
            "technique": "수법 이름",
            "description": "수법 설명",
            "application": "시나리오 적용 방법",
            "expected_effect": "예상 효과",
            "scenario_fit_score": 0.85  (0-1)
        },
        ...
    ]
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, timeout=40)
    
    # 검색 결과 정리
    search_summary = []
    for i, item in enumerate(vulnerability_info[:15], 1):
        search_summary.append(
            f"{i}. [{item['query']}] {item['title']}\n{item['content']}\n"
        )
    
    prompt = f"""
너는 보이스피싱 시나리오 전문가다...

아래 정보를 바탕으로 **공격을 강화할 수 있는 수법 10개**를 생성하라.

[피해자 정보]
{json.dumps(victim_profile, ensure_ascii=False, indent=2)}

[현재 시나리오]
{current_scenario}

[피해자가 의심한 포인트]
{json.dumps(victim_suspicion_points, ensure_ascii=False)}

[취약점 정보 (웹 검색 결과)]
{chr(10).join(search_summary)}

출력 형식 (JSON만):
{{
  "techniques": [
    {{
      "technique": "수법 이름 (간결하게)",
      "description": "수법 설명 (2-3문장)",
      "application": "'{current_scenario}' 시나리오에 어떻게 적용할지 구체적으로",
      "expected_effect": "피해자에게 미칠 것으로 예상되는 심리적 효과",
      "scenario_fit_score": 0.0~1.0 (현재 시나리오에 얼마나 적합한지)
    }},
    ... (총 10개)
  ]
}}

규칙:
1. 정확히 10개의 수법 생성
2. 웹 검색에서 발견한 취약점을 **반드시 활용**
3. scenario_fit_score는 냉정하게 평가 (무리하게 높이지 마라)
4. 피해자가 의심한 포인트를 보완하는 방향
5. 윤리적 경계를 넘지 않되, 실전 적용 가능한 수법
""".strip()
    
    try:
        print("\n🧠 LLM으로 수법 생성 중...")
        response = llm.invoke(prompt).content.strip()
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        result = json.loads(response)
        techniques = result.get("techniques", [])
        
        print(f"   ✅ {len(techniques)}개 수법 생성 완료")
        
        # 점수별 정렬
        techniques.sort(key=lambda x: x.get("scenario_fit_score", 0), reverse=True)
        
        return techniques
    
    except Exception as e:
        print(f"   ⚠️ 수법 생성 실패: {str(e)}")
        return []


@tool("filter_and_select_techniques")
def filter_and_select_techniques(
    techniques: List[Dict[str, Any]],
    min_score: float = 0.6,
    target_count: int = 3,
) -> Dict[str, Any]:
    """
    생성된 수법 중 시나리오에 적합한 것만 선택한다.
    
    입력:
    - techniques: 생성된 수법 리스트
    - min_score: 최소 적합도 점수 (기본 0.6)
    - target_count: 목표 개수 (기본 3)
    
    출력:
    {
        "selected": [...],  # 선택된 수법
        "count": 3,
        "need_more": false  # 추가 생성 필요 여부
    }
    """
    # 점수 필터링
    filtered = [
        t for t in techniques
        if t.get("scenario_fit_score", 0) >= min_score
    ]
    
    selected = filtered[:target_count * 2]  # 여유있게 선택
    
    need_more = len(selected) < target_count
    
    print(f"\n📋 수법 필터링:")
    print(f"   - 전체: {len(techniques)}개")
    print(f"   - 적합 (>={min_score}): {len(filtered)}개")
    print(f"   - 선택: {len(selected)}개")
    print(f"   - 추가 필요: {'예' if need_more else '아니오'}")
    
    return {
        "selected": selected,
        "count": len(selected),
        "need_more": need_more
    }


@tool("create_attack_enhancement_report")
def create_attack_enhancement_report(
    conversation_summary: str,
    victim_profile: Dict[str, Any],
    current_scenario: str,
    selected_techniques: List[Dict[str, Any]],
    analysis_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    최종 리포트를 작성한다.
    
    출력:
    {
        "report": {
            "summary": "...",
            "victim_profile": {...},
            "enhanced_techniques": [...],
            "implementation_guide": "...",
            "expected_outcomes": [...]
        },
        "metadata": {...}
    }
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
    
    prompt = f"""
너는 보이스피싱 시나리오 분석 리포트를 작성하는 전문가다.

[대화 요약]
{conversation_summary}

[피해자 프로필]
{json.dumps(victim_profile, ensure_ascii=False, indent=2)}

[현재 시나리오]
{current_scenario}

[선택된 강화 수법 {len(selected_techniques)}개]
{json.dumps(selected_techniques, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 **다음 대화 생성에 활용할 수 있는** 실전 리포트를 작성하라.

출력 형식 (JSON):
{{
  "summary": "이번 분석의 핵심 요약 (3-4문장)",
  "victim_profile": {{
    "age_group": "...",
    "occupation": "...",
    "key_vulnerabilities": ["취약점1", "취약점2", ...]
  }},
  "enhanced_techniques": [
    {{
      "technique": "수법 이름",
      "why_effective": "왜 이 피해자에게 효과적인지",
      "how_to_apply": "구체적 적용 방법 (대사 예시 포함)",
      "caution": "주의사항"
    }},
    ... (선택된 수법 모두)
  ],
  "implementation_guide": "전체적인 적용 가이드 (시간대, 순서, 톤 등)",
  "expected_outcomes": [
    "예상 결과 1",
    "예상 결과 2",
    ...
  ]
}}
""".strip()
    
    try:
        print("\n📝 최종 리포트 작성 중...")
        response = llm.invoke(prompt).content.strip()
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        report = json.loads(response)
        
        now = datetime.now(timezone.utc).isoformat()
        
        result = {
            "report": report,
            "metadata": {
                "created_at": now,
                "victim_profile": victim_profile,
                "techniques_analyzed": len(analysis_data.get("vulnerability_questions", [])),
                "techniques_generated": 10,
                "techniques_selected": len(selected_techniques),
                "scenario": current_scenario
            }
        }
        
        print("   ✅ 리포트 작성 완료")
        
        return result
    
    except Exception as e:
        print(f"   ⚠️ 리포트 작성 실패: {str(e)}")
        return {
            "report": {},
            "metadata": {},
            "error": str(e)
        }