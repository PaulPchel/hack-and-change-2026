"""МТС Energytool: ИИ-консультант. Точка входа для автоматического сабмита.

Читает DATA_DIR/questions.csv и DATA_DIR/knowledge-base.zip,
пишет submission.csv (id,answer) в рабочий каталог.
Ключи и адреса берутся только из переменных окружения.
"""

import csv
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from rank_bm25 import BM25Okapi
import re

# ---------- настройки из окружения ----------
def env(name, default=""):
    v = os.getenv(name, "").strip()
    return default if not v or v == name else v


def num(name, default, lo, hi, kind=int):
    try:
        return max(lo, min(hi, kind(env(name))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", "data"))
BASE_URL = (env("LLM_BASE_URL") or env("OPENAI_BASE_URL") or env("BASE_URL")).rstrip("/")
API_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "qwen/qwen3.8-27b")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
TOP_K = num("TOP_K", 8, 1, 30)
MAX_TOKENS = num("MAX_TOKENS", 1024, 128, 8192)
TEMPERATURE = num("TEMPERATURE", 0.1, 0.0, 2.0, float)
MAX_WORKERS = num("MAX_WORKERS", 4, 1, 8)

FALLBACK = (
    "В моей базе знаний нет точного ответа на этот вопрос. "
    "Могу передать заявку специалисту МТС Energytool, он уточнит детали."
)

SYSTEM_PROMPT = """Ты — консультант сайта МТС Energytool (учёт и анализ электроэнергии, \
снижение затрат, поиск потерь, солнечные станции).

Правила:
1. Факты о продуктах, ценах, условиях и цифрах бери ТОЛЬКО из блока «Материалы». \
Свои знания об энергетике не используй и ничего не выдумывай.
2. Если в материалах нет ответа, прямо скажи об этом и предложи связаться со специалистом.
3. Если для точного ответа не хватает данных о клиенте (тип объекта, мощность, регион, \
текущие приборы), ответь по сути и задай ОДИН короткий уточняющий вопрос в конце.
4. Отвечай простым языком, без лишних терминов, обычно до 150 слов, без списков и заголовков.
5. Не называй производителей оборудования, подрядчиков, клиентов и не раскрывай \
внутренние алгоритмы и пороги. Не давай юридических заключений.
6. Не гарантируй экономический эффект: точный расчёт делается после изучения данных объекта.
7. Если пользователь не хочет оставлять контакты, не настаивай: объясни, какие данные \
нужны для оценки.
8. Если вопрос не про продукты Energytool и задачи клиента, вежливо скажи, что это вне \
твоей компетенции, и предложи связаться со специалистом.
9. Если передана история диалога, отвечай на последнюю реплику пользователя с её учётом.
Отвечай на русском языке."""


# ---------- загрузка и разбиение базы ----------
def load_kb(path):
    docs = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.endswith((".md", ".txt")):
                text = z.read(name).decode("utf-8", errors="ignore")
                docs.append((name, text))
    return docs


def chunk_doc(name, text, max_len=1200):
    """Режем по заголовкам, длинные разделы — по абзацам. К каждому куску добавляем путь заголовков."""
    chunks, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if len(body) < 40:
            return
        head = f"[{name}] " + " > ".join(path)
        if len(body) <= max_len:
            chunks.append((name, f"{head}\n{body}"))
            return
        part = ""
        for para in re.split(r"\n\s*\n", body):
            if part and len(part) + len(para) > max_len:
                chunks.append((name, f"{head}\n{part.strip()}"))
                part = ""
            part += para + "\n\n"
        if part.strip():
            chunks.append((name, f"{head}\n{part.strip()}"))

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def tokenize(text):
    return [w for w in re.findall(r"[0-9a-zа-яё]+", text.lower())]


def stem(w):
    return w[:6]  # грубый «стемминг» для русского: достаточно для первой версии


def toks(text):
    return [stem(w) for w in tokenize(text) if len(w) > 2]


# ---------- обращения к API ----------
def post(client, path, payload, retries=3):
    for i in range(retries):
        try:
            r = client.post(
                f"{BASE_URL}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=120,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            time.sleep(2 * (i + 1))
    return None


def embed(client, texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        data = post(client, "/embeddings", {"model": EMB_MODEL, "input": texts[i:i + batch]})
        if not data:
            return None
        out.extend(d["embedding"] for d in data["data"])
    m = np.array(out, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class Retriever:
    def __init__(self, client, chunks):
        self.client = client
        self.chunks = chunks
        self.bm25 = BM25Okapi([toks(c[1]) for c in chunks])
        self.emb = embed(client, [c[1] for c in chunks])  # None, если эмбеддинги недоступны

    def search(self, query, k=TOP_K):
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


# ---------- диалог и ответ ----------
def split_dialog(question):
    """Вопрос может быть историей «Пользователь: ... Ассистент: ...». Возвращает messages."""
    if "Пользователь:" not in question:
        return [{"role": "user", "content": question}], question
    parts = re.split(r"(?m)^(Пользователь|Ассистент):\s*", question)
    msgs = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Пользователь" else "assistant"
        msgs.append({"role": role, "content": parts[i + 1].strip()})
    if not msgs:
        return [{"role": "user", "content": question}], question
    # запрос для поиска: последние реплики пользователя
    user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
    return msgs, user_text


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.split("</think>")[-1]
    return re.sub(r"\s+", " ", text).strip()


def answer(client, retriever, question):
    msgs, search_q = split_dialog(question)
    found = retriever.search(search_q)
    context = "\n\n---\n\n".join(c[1] for c in found) or "(ничего не найдено)"
    system = SYSTEM_PROMPT + "\n\nМатериалы:\n" + context
    data = post(
        client,
        "/chat/completions",
        {
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system}] + msgs,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    )
    try:
        text = clean(data["choices"][0]["message"]["content"])
    except (TypeError, KeyError, IndexError):
        text = ""
    return text[:9500] if text else FALLBACK


def find_file(*names):
    for root in (DATA_DIR, Path.cwd(), Path(__file__).resolve().parent):
        for n in names:
            if (root / n).is_file():
                return root / n
    raise FileNotFoundError(names)


def main():
    q_path = find_file("questions.csv", "train.csv")
    kb_path = find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    with open(q_path, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        delim = '\t' if '\t' in first_line else ','
        rows = list(csv.DictReader(f, delimiter=delim))
    chunks = []
    for name, text in load_kb(kb_path):
        chunks.extend(chunk_doc(name, text))

    with httpx.Client() as client:
        retriever = Retriever(client, chunks)

        def work(row):
            try:
                return row["id"], answer(client, retriever, row["question"])
            except Exception as e:  # один сбойный вопрос не должен ронять весь запуск
                print("error", row["id"], e, file=sys.stderr)
                return row["id"], FALLBACK

        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            results = list(pool.map(work, rows))

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "answer"])
        for id_, ans in results:
            w.writerow([id_, ans or FALLBACK])


if __name__ == "__main__":
    main()
