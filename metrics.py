import csv
import httpx
from solution import Retriever, load_kb, chunk_doc, find_file

def main():
    # 1. Загружаем эталон (какой файл правильный для каждого вопроса)
    truth_path = find_file("ground_truth.csv")
    ground_truth = {}
    with open(truth_path, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        delim = '\t' if '\t' in first_line else ','
        for row in csv.DictReader(f, delimiter=delim):
            ground_truth[row["id"]] = row["file"]

    # 2. Загружаем вопросы
    q_path = find_file("questions.csv")
    with open(q_path, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        delim = '\t' if '\t' in first_line else ','
        questions = list(csv.DictReader(f, delimiter=delim))

    # 3. Готовим базу знаний
    kb_path = find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    chunks = []
    for name, text in load_kb(kb_path):
        chunks.extend(chunk_doc(name, text))

    # 4. Запускаем поиск
    with httpx.Client() as client:
        retriever = Retriever(client, chunks)
        
        hits = 0
        total = 0
        
        print("--- Оценка поиска ---")
        for row in questions:
            q_id = row.get("id")
            if q_id not in ground_truth:
                continue
                
            total += 1
            query = row["question"]
            target_file = ground_truth[q_id]
            
            # Ищем топ-8 фрагментов
            results = retriever.search(query, k=8)
            
            # Достаем имена файлов из найденных фрагментов
            found_files = [c[0] for c in results]
            
            if target_file in found_files:
                hits += 1
            else:
                print(f" Промах по вопросу {q_id}. Искали: {target_file}")

        
            print(f"\n Метрика (Recall@8): {hits} из {total} ({hits/total*100:.1f}%)")
       

if __name__ == "__main__":
    main()