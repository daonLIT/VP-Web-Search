# scripts/check_guidance_db.py
import json
from langchain_openai import OpenAIEmbeddings
from app.tools.store import get_chroma

embeddings = OpenAIEmbeddings()
vectordb = get_chroma(embeddings)

# Chroma collection 접근
col = vectordb._collection

# === 1. 통합 검색으로 생성된 지침 확인 ===
print("="*60)
print("🔍 통합 검색으로 생성된 지침 (voicephishing_guidance_v1)")
print("="*60)

where = {"kind": {"$eq": "voicephishing_guidance_v1"}}
data = col.get(where=where, limit=20, include=["documents", "metadatas"])

guidance_count = len(data.get("ids", []))
print(f"\n총 {guidance_count}개 지침 저장됨\n")

for i, (doc_id, content, meta) in enumerate(zip(
    data.get("ids", []), 
    data.get("documents", []), 
    data.get("metadatas", [])
), 1):
    print(f"[{i}] ID: {doc_id[:30]}...")
    print(f"    유형: {meta.get('phishing_type')}")
    print(f"    생성일: {meta.get('created_at')}")
    print(f"    Guidance ID: {meta.get('guidance_id', 'N/A')[:20]}...")
    print(f"    출처: {meta.get('source_system', 'N/A')}")
    
    # JSON 파싱해서 주요 정보 표시
    try:
        guidance_data = json.loads(content)
        print(f"    키워드: {', '.join(guidance_data.get('keywords', [])[:3])}...")
        print(f"    시나리오 단계: {len(guidance_data.get('scenario', []))}개")
        print(f"    출처 수: {len(guidance_data.get('sources', []))}개")
    except:
        print(f"    내용 미리보기: {content[:80]}...")
    
    print("-" * 60)

# === 2. 크롤링으로 생성된 지침 확인 ===
print("\n" + "="*60)
print("🕷️  크롤링으로 생성된 지침 (voicephishing_guidance_crawled_v1)")
print("="*60)

where_crawled = {"kind": {"$eq": "voicephishing_guidance_crawled_v1"}}
data_crawled = col.get(where=where_crawled, limit=20, include=["documents", "metadatas"])

crawled_count = len(data_crawled.get("ids", []))
print(f"\n총 {crawled_count}개 크롤링 지침 저장됨\n")

for i, (doc_id, content, meta) in enumerate(zip(
    data_crawled.get("ids", []), 
    data_crawled.get("documents", []), 
    data_crawled.get("metadatas", [])
), 1):
    print(f"[{i}] ID: {doc_id[:30]}...")
    print(f"    유형: {meta.get('phishing_type')}")
    print(f"    출처 사이트: {meta.get('source_site', 'N/A')[:50]}...")
    print(f"    생성일: {meta.get('created_at')}")
    
    try:
        guidance_data = json.loads(content)
        print(f"    키워드: {', '.join(guidance_data.get('keywords', [])[:3])}...")
    except:
        pass
    
    print("-" * 60)

# === 3. 전체 통계 ===
print("\n" + "="*60)
print("📊 전체 통계")
print("="*60)

all_kinds = col.get(limit=1000, include=["metadatas"])
kind_counts = {}

for meta in all_kinds.get("metadatas", []):
    kind = meta.get("kind", "unknown")
    kind_counts[kind] = kind_counts.get(kind, 0) + 1

print("\nKind별 문서 수:")
for kind, count in sorted(kind_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {kind}: {count}개")

print(f"\n총 문서 수: {sum(kind_counts.values())}개")