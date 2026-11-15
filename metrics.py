"""Оценка поиска: в какой доле вопросов нужный файл попал в топ-k.

Эталон лежит в eval/ground_truth.csv (колонки id,files). Его размечают ВРУЧНУЮ:
открываем базу знаний, находим файл с ответом и пишем его имя (несколько файлов через «|»).
Нельзя заполнять эталон по выдаче самого поиска: тогда метрика ничего не измеряет.

    python metrics.py                # Recall@8
    python metrics.py --k 5          # другое k
    python metrics.py --bm25         # только по словам, ключ API не нужен
    python metrics.py --candidates   # подсказки для разметки: топ-5 файлов по каждому вопросу
"""

import argparse
import csv
import sys

import solution


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        return list(csv.DictReader(f, delimiter="\t" if "\t" in first_line else ","))


def load_truth(path):
    truth = {}
    for row in read_csv(path):
        files = {x.strip() for x in (row.get("files") or "").split("|") if x.strip()}
        if files:
            truth[row["id"].strip()] = files
    return truth


def distinct_files(results):
    seen = []
    for name, _ in results:
        if name not in seen:
            seen.append(name)
    return seen


def main():
    parser = argparse.ArgumentParser(description="Оценка качества поиска по базе знаний")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--bm25", action="store_true", help="искать только по словам")
    parser.add_argument("--candidates", action="store_true", help="показать подсказки для ручной разметки")
    args = parser.parse_args()

    if args.bm25:
        solution.embed = lambda *a, **kw: None
    import retrieval

    questions = read_csv(solution.find_file("questions.csv"))

    if args.candidates:
        print("Подсказки для разметки. Проверьте каждую по тексту базы, не копируйте вслепую.\n")
        for row in questions:
            query = solution.split_dialog(row["question"])[1]
            files = distinct_files(retrieval.search(query, 12))[:5]
            print(f"[{row['id']}] {row['question'][:110]!r}\n     кандидаты: {', '.join(files)}")
        return

    truth = load_truth(solution.find_file("eval/ground_truth.csv"))
    if not truth:
        print("eval/ground_truth.csv пуст. Заполните колонку files вручную (см. описание в начале metrics.py).")
        sys.exit(1)

    hits = top1 = total = 0
    for row in questions:
        expected = truth.get(row["id"].strip())
        if not expected:
            continue
        total += 1
        query = solution.split_dialog(row["question"])[1]
        found = distinct_files(retrieval.search(query, args.k))
        if expected & set(found):
            hits += 1
        else:
            print(f"Промах [{row['id']}]: ожидали {sorted(expected)}, нашли {found}")
        if found and found[0] in expected:
            top1 += 1

    print(f"\nРазмечено вопросов: {total} из {len(questions)}")
    print(f"Recall@{args.k}: {hits} из {total} ({hits / total * 100:.1f}%)")
    print(f"Нужный файл на первом месте: {top1} из {total} ({top1 / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
