"""Временный консультант, пока нет настоящего consultant.py.

Если в .env заданы LLM_BASE_URL и LLM_API_KEY, отвечает по-настоящему, используя
поиск и промпт из solution.py. Иначе возвращает заготовленный ответ, чтобы
можно было проверить сайт без ключей. Когда команда добавит consultant.py,
этот файл перестанет использоваться (его можно удалить).
"""

import re
import threading

from tracing import Trace

PHONE = re.compile(r"(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
_state = {}
_lock = threading.Lock()


def _retriever(solution):
    with _lock:
        if "r" not in _state:
            import httpx

            chunks = []
            for name, text in solution.load_kb(solution.find_file("knowledge-base.zip")):
                chunks.extend(solution.chunk_doc(name, text))
            _state["client"] = httpx.Client()
            _state["r"] = solution.Retriever(_state["client"], chunks)
    return _state["client"], _state["r"]


def _real(history):
    import solution

    if not (solution.BASE_URL and solution.API_KEY):
        return None
    tr = Trace()
    with tr.step("init_index", "загрузка базы и индекса (при первом запросе)"):
        client, retriever = _retriever(solution)
    search_q = " ".join(m["content"] for m in history if m["role"] == "user")
    with tr.step("search") as s:
        found = retriever.search(search_q)
        s["note"] = f"BM25 + эмбеддинги, фрагментов: {len(found)}"
    context = "\n\n---\n\n".join(c[1] for c in found) or "(ничего не найдено)"
    llm = tr.step("llm", solution.LLM_MODEL)
    llm.__enter__()
    data = solution.post(
        client,
        "/chat/completions",
        {
            "model": solution.LLM_MODEL,
            "messages": [{"role": "system", "content": solution.SYSTEM_PROMPT + "\n\nМатериалы:\n" + context}]
            + history,
            "max_tokens": solution.MAX_TOKENS,
            "temperature": solution.TEMPERATURE,
        },
    )
    llm.__exit__(None, None, None)
    try:
        text = solution.clean(data["choices"][0]["message"]["content"])
    except (TypeError, KeyError, IndexError):
        text = ""
    sources = sorted({c[0] for c in found})
    return {"text": text or solution.FALLBACK, "sources": sources, "lead": None, "trace": tr.steps}


def answer(history):
    real = _real(history)
    if real:
        return real
    last = history[-1]["content"]
    if PHONE.search(last):
        return {
            "text": "Спасибо! Заявка сформирована, менеджер свяжется с вами. (заглушка)",
            "sources": [],
            "lead": {
                "name": "Тестов Тест Тестович",
                "phone": PHONE.search(last).group(0),
                "company": "ООО «Тест»",
                "summary": "Клиент спрашивал: " + history[0]["content"][:200] + " (заглушка)",
            },
            "trace": [{"step": "make_lead", "ms": 0, "note": "тестовая заявка (заглушка)"}],
        }
    return {
        "text": "Это ответ-заглушка: настоящий консультант пока не подключён. "
        "Вы написали: «" + last[:200] + "». Чтобы проверить заявку, отправьте тестовый номер, например +7 900 000 00 00.",
        "sources": ["00-reference/00-product-map.md (заглушка)"],
        "lead": None,
        "trace": [{"step": "stub", "ms": 0, "note": "ответ-заглушка, поиск и модель не вызывались"}],
    }
