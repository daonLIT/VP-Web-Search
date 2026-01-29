# test_guidance.py (프로젝트 루트에 생성)
from app.orchestrator_guidance import build_guidance_orchestrator
import json


# Orchestrator 초기화
print("=== Orchestrator 초기화 중... ===")
orch = build_guidance_orchestrator(model_name="gpt-4o")

# 테스트 요청
request = {
    "phishing": True,
    "type": "검경 사칭",
    "scenario": "검찰 사칭해서 현금 편취",
    "victim_profile": {
        "age": 65,
        "occupation": "퇴직자"
    }
}

print("\n=== 요청 데이터 ===")
print(json.dumps(request, ensure_ascii=False, indent=2))

# 실행
print("\n=== 에이전트 실행 중... ===")
result = orch.handle(request)

# 결과 분석
print("\n" + "="*60)
print("=== 최종 결과 ===")
print("="*60)

status = result.get("status")
source = result.get("source")

if status == "found_in_db":
    print("✅ DB에서 기존 데이터를 찾았습니다!")
    print(f"   출처: {source}")
elif status == "generated_new":
    print("🔍 DB에 없어서 새로 웹 검색했습니다!")
    print(f"   출처: {source}")
    print(f"   저장된 ID: {result.get('guidance_id')}")
else:
    print(f"⚠️  상태: {status}")

print("\n=== 지침 내용 ===")
guidance = result.get("guidance", {})
print(f"유형: {guidance.get('type')}")
print(f"키워드: {', '.join(guidance.get('keywords', []))}")
print(f"시나리오 단계 수: {len(guidance.get('scenario', []))}")
print(f"출처 수: {len(guidance.get('sources', []))}")

print("\n=== 전체 JSON ===")
print(json.dumps(result, ensure_ascii=False, indent=2))