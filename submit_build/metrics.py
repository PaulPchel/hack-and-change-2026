# metrics.py
import csv, os, sys
from collections import Counter
from pathlib import Path
import httpx

from solution import (
    Retriever, load_kb, chunk_doc, find_file,
    TOP_K, BASE_URL, API_KEY,
)

with open("questions.csv", encoding="utf-8", newline="") as f:
    GROUND_TRUTH = csv.reader(f)          # delimiter=";", quotechar='"' по умолчанию
    

def main():
    q_path  = find_file("questions.csv", "train.csv")
    kb_path = find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")

    with open(q_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    chunks = []
    for name, text in load_kb(kb_path):
        chunks.extend(chunk_doc(name, text))

    with httpx.Client() as client:
        r = Retriever(client, chunks)

        # (A) диагностика: сколько раз каждый файл попадал в top-K
        top_hits = Counter()
        # (B) метрика: recall@K по размеченным вопросам
        hits, labelled = 0, 0
        misses = []

        for row in rows:
            qid = int(row["id"])
            query = row["question"]
            results = r.search(query, k=TOP_K)

            # имя файла = c[0]; дедуп на вопрос
            files_in_top = {c[0] for c in results}
            top_hits.update(files_in_top)

            gt = GROUND_TRUTH.get(qid)
            if gt:
                labelled += 1
                if files_in_top & gt:
                    hits += 1
                else:
                    misses.append(qid)

    print(f"\nTOP_K = {TOP_K}, CHUNK_SIZE = {os.getenv('CHUNK_SIZE','1200')}")
    print("Попадания файлов в top-K:", dict(top_hits))
    if labelled:
        print(f"Recall@{TOP_K} = {hits}/{labelled} = {hits/labelled:.2%}")
        if misses:
            print("Промахи по id:", misses)

if __name__ == "__main__":
    main()