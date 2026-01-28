from app.orchestrator_webonly import build_webonly_orchestrator

def main():
    orc = build_webonly_orchestrator()
    text = "최근 비트코인 뉴스 찾아줘"
    out = orc.handle(text, thread_id="webonly-demo")
    print(out["final"])

if __name__ == "__main__":
    main()
