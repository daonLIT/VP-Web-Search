from app import build_orchestrator

def main():
    orc = build_orchestrator()

    text = "최근 AI 에이전트 웹 검색 관련 뉴스 요약해줘. 근거 링크도 같이."
    out = orc.handle(text, thread_id="demo")

    print("\n=== FINAL(JSON string) ===")
    print(out["final"])

if __name__ == "__main__":
    main()
