# scripts/check_db.py
# 저장된 데이터 DB에 있는지 확인
from langchain_openai import OpenAIEmbeddings
from app.tools.store import get_chroma

embeddings = OpenAIEmbeddings()
vectordb = get_chroma(embeddings)

# Chroma collection 접근
col = vectordb._collection

# 저장된 guidance 데이터 확인
where = {"kind": {"$eq": "voicephishing_guidance_v1"}}
data = col.get(where=where, limit=10, include=["documents", "metadatas"])

print(f"=== DB에 저장된 guidance 개수: {len(data.get('ids', []))} ===\n")

for doc_id, content, meta in zip(
    data.get("ids", []), 
    data.get("documents", []), 
    data.get("metadatas", [])
):
    print(f"ID: {doc_id}")
    print(f"유형: {meta.get('phishing_type')}")
    print(f"생성일: {meta.get('created_at')}")
    print(f"Guidance ID: {meta.get('guidance_id')}")
    print(f"내용 미리보기: {content[:100]}...")
    print("-" * 60)