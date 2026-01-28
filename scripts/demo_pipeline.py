# scripts/demo_pipeline.py
import json

from app.orchestrator_webonly import build_webonly_orchestrator
from app.orchestrator_summarize import build_summarize_orchestrator


def main():
    thread_id = "pipeline-demo"
    query = "보이스피싱 최신 수법"

    # 1) 수집 단계(에이전트: web_search_snippets -> store_snippets_only)
    collector = build_webonly_orchestrator()
    collect_out = collector.handle(query, thread_id=thread_id)

    # 2) 요약 단계(에이전트: load_collected_snippets -> write_report_from_snippets_and_store -> mark_snippets_processed)
    summarizer = build_summarize_orchestrator()
    summarize_out = summarizer.handle(query, thread_id=thread_id)

    out = {
        "collect": collect_out,
        "summarize": summarize_out,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
