# scripts/test_crawl.py
import json
from app.orchestrator_crawl import build_crawl_orchestrator

# Orchestrator 초기화
print("=== Crawl Orchestrator 초기화 ===")
orch = build_crawl_orchestrator(model_name="gpt-4o")

# 테스트: 경찰청 사이버안전국 공지사항
request = {
    "site_url": "https://www.kisa.or.kr/402?page=1&searchDiv=10&searchWord=%ED%94%BC%EC%8B%B1&_csrf=ab52158b-23d5-451a-b27f-581be612d456",  # 예시 URL
    "keywords": ["보이스피싱", "전화금융사기", "스미싱"],
    "max_articles": 5,
}

print("\n=== 요청 ===")
print(json.dumps(request, ensure_ascii=False, indent=2))

print("\n=== 크롤링 시작 ===")
result = orch.handle(request)

print("\n=== 결과 ===")
print(json.dumps(result, ensure_ascii=False, indent=2))

if result.get("status") == "success":
    print(f"\n✅ 크롤링 성공!")
    print(f"   수집된 글: {result.get('crawled_count')}")
    print(f"   추출된 글: {result.get('extracted_count')}")
    print(f"   생성된 유형: {result.get('types_generated')}")