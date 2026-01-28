import json
from app.tools.store import get_chroma
from app.config import SETTINGS
from langchain_openai import OpenAIEmbeddings  # 네 프로젝트에 맞게 이름이 다르면 webonly에서 쓰는 임베딩 함수로 바꿔줘

def main():
    emb = OpenAIEmbeddings()
    vectordb = get_chroma(embeddings=emb)
    col = vectordb._collection

    # 최신 리포트 1개 가져오기
    rep = col.get(
        where={"kind": {"$eq": "voicephishing_types_v1"}},
        limit=5,
        include=["documents", "metadatas"],
    )
    rows = list(zip(rep["ids"], rep["metadatas"], rep["documents"]))
    rows.sort(key=lambda x: (x[1] or {}).get("created_at", ""), reverse=True)
    rep_id, rep_meta, rep_doc = rows[0]

    print("LATEST REPORT DOC_ID:", rep_id)
    print("created_at:", (rep_meta or {}).get("created_at"))
    print("source_snippet_ids_json:", (rep_meta or {}).get("source_snippet_ids_json"))

    snippet_ids = json.loads((rep_meta or {}).get("source_snippet_ids_json", "[]"))

    # 스니펫들 중 snippet_id가 일치하는 애들 찾기(전체 훑어서 매칭)
    sn = col.get(
        where={"kind": {"$eq": "voicephishing_snippet_v1"}},
        limit=500,
        include=["metadatas"],
    )

    id2meta = {}
    for _id, m in zip(sn["ids"], sn["metadatas"]):
        sid = (m or {}).get("snippet_id")
        if sid:
            id2meta[sid] = ( _id, m )

    hit = 0
    for sid in snippet_ids:
        if sid in id2meta:
            doc_id, m = id2meta[sid]
            print("HIT snippet_id:", sid, "doc_id:", doc_id, "processed:", m.get("processed"), "used_in_report_id:", m.get("used_in_report_id"))
            hit += 1

    print("matched:", hit, "/", len(snippet_ids))

if __name__ == "__main__":
    main()
