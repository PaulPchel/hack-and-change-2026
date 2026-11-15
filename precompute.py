# precompute.py
import sys
import httpx
from solution import (
    Retriever, load_kb, chunk_doc, find_file,
    EMB_MODEL, CHUNK_SIZE, CACHE_DIR,
)

def main():
    kb_path = find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    chunks = []
    for name, text in load_kb(kb_path):
        chunks.extend(chunk_doc(name, text))
    print(f"chunks: {len(chunks)}")

    with httpx.Client() as client:
        r = Retriever(client, chunks)   # сам загрузит кэш или посчитает
    if r.emb is None:
        print("FAILED: embeddings unavailable", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {r.emb.shape} cached in {CACHE_DIR}/")

if __name__ == "__main__":
    main()