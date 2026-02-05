# scripts/test_new_api.py
"""
새로운 API 테스트 스크립트
"""
import requests
import json

BASE_URL = "http://localhost:8001"


def test_health():
    """헬스 체크 테스트"""
    print("=" * 50)
    print("Testing /health")
    print("=" * 50)

    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()


def test_quick_analyze():
    """빠른 분석 테스트 (웹 검색 없이)"""
    print("=" * 50)
    print("Testing /api/analyze/quick")
    print("=" * 50)

    # 텍스트 데이터
    payload = {
        "data": """
        피해자는 30대 중후반 남자 직장인이다.
        피싱범은 전화로 연락해 검찰청 수사관을 사칭했다.
        "당신 명의의 계좌가 범죄에 연루되었습니다"라고 협박했고,
        피해자는 처음에는 의심했지만 사건번호를 알려주자 믿기 시작했다.
        """,
        "analysis_type": "conversation"
    }

    resp = requests.post(f"{BASE_URL}/api/analyze/quick", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()


def test_full_analyze():
    """전체 분석 테스트 (웹 검색 포함)"""
    print("=" * 50)
    print("Testing /api/analyze (full)")
    print("=" * 50)

    payload = {
        "data": """
        피해자는 60대 퇴직자 여성이다.
        자녀를 사칭한 전화를 받았다.
        "엄마, 나 급하게 돈이 필요해. 사고가 났어."
        피해자는 목소리가 이상하다고 느꼈지만 걱정이 앞섰다.
        """,
        "analysis_type": "conversation",
        "search_config": {
            "max_results_per_query": 2,
            "extract_content": True
        }
    }

    print("Sending request (this may take a while)...")
    resp = requests.post(f"{BASE_URL}/api/analyze", json=payload, timeout=180)
    print(f"Status: {resp.status_code}")

    result = resp.json()
    # 응답이 길 수 있으므로 요약
    if resp.status_code == 200:
        print("\n=== Report Summary ===")
        if result.get("report"):
            print(f"Summary: {result['report'].get('summary', 'N/A')[:200]}...")
            print(f"Vulnerabilities: {len(result['report'].get('vulnerabilities', []))}")
            print(f"Techniques: {len(result['report'].get('techniques', []))}")
        print(f"Sources: {len(result.get('sources', []))}")
    else:
        print(f"Error: {result}")
    print()


def test_json_data():
    """JSON 데이터 입력 테스트"""
    print("=" * 50)
    print("Testing with JSON data")
    print("=" * 50)

    payload = {
        "data": {
            "victim_age": 45,
            "victim_occupation": "자영업자",
            "victim_gender": "남성",
            "call_transcript": "대출 금리 인하가 가능합니다. 기존 대출 상환 후 재대출하면 됩니다.",
            "scenario_type": "대출 사기"
        }
    }

    resp = requests.post(f"{BASE_URL}/api/analyze/quick", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()


def test_list_data():
    """리스트 데이터 입력 테스트"""
    print("=" * 50)
    print("Testing with list data")
    print("=" * 50)

    payload = {
        "data": [
            "첫 번째 통화: 검찰청이라며 계좌 동결 이야기",
            "두 번째 통화: 금융감독원 직원이라며 안전계좌 언급",
            "세 번째 통화: 원격 앱 설치 요구"
        ]
    }

    resp = requests.post(f"{BASE_URL}/api/analyze/quick", json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VP-Web-Search API Test Suite")
    print("=" * 60 + "\n")

    try:
        # 1. 헬스 체크
        test_health()

        # 2. 빠른 분석 (텍스트)
        test_quick_analyze()

        # 3. JSON 데이터 테스트
        test_json_data()

        # 4. 리스트 데이터 테스트
        test_list_data()

        # 5. 전체 분석 (웹 검색 포함) - 시간이 오래 걸림
        # test_full_analyze()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Make sure it's running on localhost:8001")
    except Exception as e:
        print(f"ERROR: {e}")
