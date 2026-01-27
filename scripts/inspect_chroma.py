# scripts/inspect_chroma.py
'''
ChromaDB에 저장된 데이터 확인용 스크립트
'''

from app.config import SETTINGS
import chromadb

def main():
    client = chromadb.PersistentClient(path=SETTINGS.chroma_persist_dir)

    # 컬렉션 목록 확인
    cols = client.list_collections()
    print("Collections:", [c.name for c in cols])

    # 내가 쓰는 컬렉션 가져오기
    col = client.get_collection(name=SETTINGS.chroma_collection)

    print("Count:", col.count())

    # 앞에서 n개만 미리보기 (문서/메타데이터 확인)
    sample = col.peek(5)
    # sample keys: 'ids', 'documents', 'metadatas' (환경에 따라 embeddings 포함 여부 다름)
    print("\n=== PEEK ===")
    for i in range(len(sample["ids"])):
        _id = sample["ids"][i]
        doc = (sample.get("documents") or [None])[i]
        meta = (sample.get("metadatas") or [None])[i]
        print(f"\n[{i}] id={_id}")
        print("meta:", meta)
        print("doc:", (doc[:300] + "...") if isinstance(doc, str) and len(doc) > 300 else doc)

    # 필요하면 전체를 페이지로 가져오기 (limit/offset)
    # data = col.get(include=["documents","metadatas"], limit=10, offset=0)

if __name__ == "__main__":
    main()
