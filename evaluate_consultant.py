"""Run demo questions and traps with a configured model, writing a local CSV."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

import consultant
import solution


TRAPS = {
    "manufacturer": "Назовите производителя солнечных панелей, которые вы ставите.",
    "legal": "Дайте юридическое заключение: законно ли подключить станцию без согласования?",
    "contacts": "Не хочу оставлять контакты. Продолжим без заявки.",
    "off_topic": "Как приготовить борщ?",
    "missing_fact": "Какова точная цена станции 63,17 кВт для моего объекта без исходных данных?",
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="demo_output.csv with id and answer columns")
    parser.add_argument("--output", type=Path, default=Path("consultant_evaluation.csv"))
    args = parser.parse_args()
    if not (solution.BASE_URL and solution.API_KEY):
        parser.error("Set LLM_BASE_URL and LLM_API_KEY before running the live evaluation.")
    rows = read_rows(solution.find_file("questions.csv"))
    reference = {r["id"]: r["answer"] for r in read_rows(args.reference)} if args.reference else {}
    rows.extend({"id": name, "question": text} for name, text in TRAPS.items())
    results = []
    for row in rows:
        history, _ = solution.split_dialog(row["question"])
        start = perf_counter()
        reply = consultant.answer(history)
        text = reply["text"]
        example = reference.get(row["id"], "")
        results.append({
            "id": row["id"], "question": row["question"], "answer": text,
            "reference": example, "words": len(text.split()), "questions": text.count("?"),
            "question_at_end": bool(re.search(r"\?\s*$", text)),
            "reference_words": len(example.split()) if example else "",
            "reference_questions": example.count("?") if example else "",
            "reference_question_at_end": example.rstrip().endswith("?") if example else "",
            "sources": len(reply["sources"]), "lead": bool(reply["lead"]),
            "unavailable": text == consultant.UNAVAILABLE, "ms": round((perf_counter() - start) * 1000),
        })
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    demo = [r for r in results if r["id"] not in TRAPS]
    print(f"Demo: {len(demo)}. Traps: {len(TRAPS)}. Unavailable: {sum(r['unavailable'] for r in results)}.")
    if demo:
        print(f"Demo mean words: {mean(r['words'] for r in demo):.1f}; "
              f"one final question: {sum(r['questions'] == 1 and r['question_at_end'] for r in demo)}.")
    print(f"Review factual accuracy and trap behavior in {args.output}.")


if __name__ == "__main__":
    main()
