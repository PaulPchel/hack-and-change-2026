import csv
import httpx
from solution import Retriever, load_kb, chunk_doc

def main():
    client = httpx.Client()
    
    print("Загрузка базы знаний...")
    # Укажите правильный путь к архиву
    docs = load_kb("knowledge-base.zip")
    chunks = []
    for name, text in docs:
        chunks.extend(chunk_doc(name, text))

    print("Построение индексов (может занять время)...")
    retriever = Retriever(client, chunks)
    print("Готово! Генерируем таблицу...\n")

    # Открываем исходные вопросы
    with open("questions.csv", encoding="utf-8-sig", newline="") as f_in:
        rows = list(csv.DictReader(f_in))

    # Создаем новый CSV для таблицы ответов
    with open("ground_truth.csv", "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["id", "question", "expected_file"])

        for row in rows:
            query = row["question"]
            q_id=row["id"]
            # Ищем ответы. Просим только 1 результат (k=1), чтобы взять самого вероятного кандидата
            results = retriever.search(query, k=1)
            
            if results:
                # В solution.py кусок текста — это кортеж (имя_файла, текст).
                # results[0] — лучший результат, results[0][0] — имя файла.
                best_file = results[0][0] 
            else:
                best_file = "НЕ НАЙДЕНО"
                
            # Записываем в таблицу
            writer.writerow([q_id, query, best_file])
            print(f"[{q_id}] {best_file} <- {query[:40]}...")

    print("\nТаблица сохранена в файл ground_truth.csv")
    client.close()

if __name__ == "__main__":
    main()