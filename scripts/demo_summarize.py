# scripts/demo_summarize.py
import json
from app.orchestrator_summarize import build_summarize_orchestrator


def main():
    query = "보이스피싱 최신 수법"
    orc = build_summarize_orchestrator()
    out = orc.handle(query, thread_id="summarize-demo")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
