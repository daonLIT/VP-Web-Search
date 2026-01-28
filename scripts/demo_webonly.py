from app.orchestrator_webonly import build_webonly_orchestrator
import uuid

def main():
    orc = build_webonly_orchestrator()
    text = "보이스피싱 최신 수법 찾아줘"
    out = orc.handle(text, thread_id=f"webonly-demo-{uuid.uuid4().hex[:8]}")
    print(out["final"])

if __name__ == "__main__":
    main()
