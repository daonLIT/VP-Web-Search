# scripts/test_crawl_multi_page.py
import json
from app.orchestrator_crawl import build_crawl_orchestrator

orch = build_crawl_orchestrator(model_name="gpt-4o")

# 여러 페이지 크롤링 테스트
request = {
    "site_url": "https://www.kisa.or.kr/402?page=1&searchDiv=10&searchWord=%ED%94%BC%EC%8B%B1&_csrf=ab52158b-23d5-451a-b27f-581be612d456",
    "keywords": ["보이스피싱", "전화금융사기"],
    "max_articles": 20,  # 최대 20개 글
    "max_pages": 3,  # 최대 3페이지
    "pagination_type": "auto",  # 자동 감지
}

print("=== 다중 페이지 크롤링 시작 ===")
result = orch.handle(request)

print("\n=== 결과 ===")
print(json.dumps(result, ensure_ascii=False, indent=2))

if result.get("status") == "success":
    print(f"\n✅ 성공!")
    print(f"   탐색한 페이지: {result.get('pages_crawled')}")
    print(f"   수집된 글: {result.get('crawled_count')}")
    print(f"   본문 추출: {result.get('extracted_count')}")
    print(f"   생성된 유형: {result.get('types_generated')}")