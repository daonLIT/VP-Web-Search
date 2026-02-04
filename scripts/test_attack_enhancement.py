# scripts/test_attack_enhancement.py
import requests
import json

BASE_URL = "http://localhost:8000"

conversation_summary = """
피해자는 30대 중후반 남자 직장인이다
피싱범은 전화로 연락해 상대가 통화 가능한지 확인한 뒤, 공식 기관 소속을 내세워 신뢰를 얻으려 한다. 이어 특정 사건을 언급하며 사실 확인이 필요하다고 접근하고, 피해자가 모르는 제3자의 정보를 꺼내면서 피해자와 사건을 연결하려는 흐름을 만든다. 피해자가 의심하며 구체적인 설명과 근거를 요구하자, 피싱범은 피해자 명의가 범죄에 연루됐다는 식으로 압박 수위를 높여 불안과 긴급함을 조성한다. 그러나 피해자는 이를 보이스피싱으로 판단하고 더 이상의 대화를 거부하며 통화를 종료한다.
"""

request = {
    "conversation_summary": conversation_summary
}

print("=== 공격 강화 분석 시작 ===\n")
response = requests.post(
    f"{BASE_URL}/api/attack/enhance",
    json=request
)

result = response.json()

print("\n=== 결과 ===")
print(json.dumps(result, ensure_ascii=False, indent=2))

if result.get("status") == "success":
    report = result.get("report", {})
    metadata = result.get("metadata", {})
    
    print(f"\n✅ 분석 완료!")
    print(f"   - 피해자: {metadata.get('victim_profile', {}).get('age_group')} {metadata.get('victim_profile', {}).get('occupation')}")
    print(f"   - 시나리오: {metadata.get('scenario')}")
    print(f"   - 선택된 수법: {metadata.get('techniques_selected')}개")
    
    print(f"\n📋 선택된 수법:")
    for i, tech in enumerate(report.get("enhanced_techniques", []), 1):
        print(f"   {i}. {tech.get('technique')}")