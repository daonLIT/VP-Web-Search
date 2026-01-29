# scripts/check_processed.py
# “요약에 쓰인 스니펫이 processed=True로 바뀌었는지” 확인 스크립트
import json
from langchain_openai import OpenAIEmbeddings
from app.tools.store import get_chroma

def main():
    emb = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings=emb)
    col = vectordb._collection

    where = {"kind": {"$eq": "voicephishing_snippet_v1"}}
    data = col.get(where=where, limit=20, include=["metadatas", "documents"])
    metas = data.get("metadatas", []) or []
    ids = data.get("ids", []) or []

    # processed 분포
    counts = {"true": 0, "false": 0, "none": 0}
    for m in metas:
        v = (m or {}).get("processed", None)
        if v is True: counts["true"] += 1
        elif v is False: counts["false"] += 1
        else: counts["none"] += 1

    print("processed counts:", counts)
    print("\n=== sample ===")
    for doc_id, m in list(zip(ids, metas))[:5]:
        print(doc_id, "processed=", (m or {}).get("processed"), "used_in_report_id=", (m or {}).get("used_in_report_id"))

if __name__ == "__main__":
    main()
