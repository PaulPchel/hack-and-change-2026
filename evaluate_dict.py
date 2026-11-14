import csv
import httpx
from solution import Retriever, load_kb, chunk_doc

# --- ЗДЕСЬ МЫ ВРУЧНУЮ ЗАДАЕМ СЛОВАРЬ ---
# Ключ (строка) -> Значение (имя файла)
GROUND_TRUTH = {
    "1": "solar_panels.md",
    "2": "solar_panels.md",
    "3": "solar_panels.md",
    "4": "solar_panels.md",
    "5": "smart_metering.md",
    "6": "dashboard.md",
    "7": "dashboard.md",
    "8": "photo_recognition.md",
    "9": "photo_recognition.md",
    "10": "analytics.md",
    "11": "reactive_power.md",
    "12": "wholesale_market.md",
    "13": "consulting.md",
    "14": "solar_panels.md",
    "15": "platform.md",
    "16": "losses.md",
    "17": "tariffs.md",
    "18": "pricing.md"
}
# (Убедитесь, что названия файлов совпадают с вашими реальными!)

def evaluate_with_dict():
    print("Загрузка базы знаний...")
    client = httpx.Client()
    docs = load_kb("knowledge-base.zip")
    chunks = []
    for name, text in docs:
        chunks.extend(chunk_doc(name, text))
    
    retriever = Retriever(client, chunks)
    print("Индексы готовы. Запуск оценки...\n")

    hits = 0
    total = 0

    # Читаем только вопросы из файла
    with open("questions.csv", encoding="utf-8-sig", newline="") as f:
       
            rows = list(csv.DictReader(f))
            
            # Проходимся по каждому словарю (строке) в списке
            for row in rows:
                query = row["question"] # Достаем текст вопроса по названию колонки
                
                # Делаем поиск
            results = retriever.search(query, k=8)
    #         found_files = [res[0] for res in results]
            
    #         if expected_file in found_files:
    #             hits += 1
            
    # print("\n" + "="*40)
    # if total > 0:
    #     metric = (hits / total) * 100
    #     print(f"ИТОГО: {hits} из {total} (Recall@8: {metric:.1f}%)")
    # print("="*40)
            d = {}
            for i in results:
                for j in i:
                    if j[-2:]=='md':
                        d[j] = d.get(j, 0) + 1
    client.close()

if __name__ == "__main__":
    evaluate_with_dict()