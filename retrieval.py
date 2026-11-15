import csv
import httpx
# Импортируем всё нужное из solution.py
from solution import* 
from collections import Counter
# 1. Создаем клиент для работы с API (выносим из цикла)
client = httpx.Client()

print("Шаг 1: Загрузка базы знаний...")
# ВАЖНО: Укажите правильный путь к вашему ZIP-файлу с базой знаний!
# Если он лежит в папке data, то пишите "data/knowledge-base.zip"
try:
    docs = load_kb("knowledge-base.zip") 
    chunks = []
    for name, text in docs:
        # Нарезаем на чанки ТЕКСТ БАЗЫ ЗНАНИЙ, а не вопросы
        chunks.extend(chunk_doc(name, text))
    print(f"Успешно нарезано {len(chunks)} фрагментов базы знаний.\n")
except Exception as e:
    print(f"Ошибка при чтении базы знаний: {e}")
    exit()

print("Шаг 2: Построение индексов (BM25 и эмбеддинги)...")
# ВАЖНО: Создаем Retriever ОДИН РАЗ до цикла по вопросам
retriever = Retriever(client, chunks)
print("Индексы готовы!\n")

print("Шаг 3: Тестирование поиска по вопросам...")
# Открываем файл с вопросами
def search(self, query):
        n = len(self.chunks)
        scores = np.zeros(n)
        bm = np.array(self.bm25.get_scores(toks(query)))
        ranks = {}
        for rank, idx in enumerate(np.argsort(-bm)[:30]):
            ranks[idx] = ranks.get(idx, 0) + 1 / (60 + rank)
        if self.emb is not None:
            q = embed(self.client, [query])
            if q is not None:
                sims = self.emb @ q[0]
                for rank, idx in enumerate(np.argsort(-sims)[:30]):
                    ranks[idx] = ranks.get(idx, 0) + 1 / (60 + rank)
        for idx, s in ranks.items():
            scores[idx] = s
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top if scores[i] > 0]
with open("questions.csv", encoding="utf-8", newline="") as f:
    # DictReader сразу делает первую строку заголовками, поэтому rows[0] - это первый вопрос, а не шапка
    rows = list(csv.DictReader(f))
    
    # Проходимся по каждому словарю (строке) в списке
    for row in rows:
        query = row["question"] # Достаем текст вопроса по названию колонки
        
        # Делаем поиск
        results = retriever.search(query) 
        
        # Выводим информацию
        print(f"Вопрос: {query[:40]}...")
        print(f"Найдено фрагментов: {len(results)}")
        
        # Проверяем условие из задачи (5-8 фрагментов)
        assert 5 <= len(results) <= 8, f"Ожидалось 5–8 фрагментов, получено {len(results)}"
        d = {}
        for i in results:
                        for j in i:
                            if j[-2:]=='md':
                                d[j] = d.get(j, 0) + 1
        
print(d)     

        

print("\nВсе вопросы обработаны успешно!")
client.close() # Закрываем соединение

