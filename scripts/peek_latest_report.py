import json
from langchain_openai import OpenAIEmbeddings
from app.tools.store import get_chroma

def main():
    emb = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings=emb)
    col = vectordb._collection

    data = col.get(
        where={"kind": {"$eq": "voicephishing_report_v1"}},
        limit=5,
        include=["metadatas", "documents"]
    )

    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    ids = data.get("ids", []) or []

    if not ids:
        print("no report found")
        return

    # created_at 기준 정렬
    rows = list(zip(ids, metas, docs))
    rows.sort(key=lambda x: (x[1] or {}).get("created_at", ""), reverse=True)

    rid, meta, doc = rows[0]
    print("doc_id:", rid)
    print("report_id:", (meta or {}).get("report_id"))
    print("source_count:", (meta or {}).get("source_count"))
    print("source_snippet_ids_json:", (meta or {}).get("source_snippet_ids_json"))
    print("\n=== report content ===\n")
    print(doc[:2000])

if __name__ == "__main__":
    main()
