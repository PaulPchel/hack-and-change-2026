"""Мини-помощник для трассировки: какие шаги выполнил консультант и сколько они заняли.

Пример использования в consultant.py:

    from tracing import Trace

    def answer(history):
        tr = Trace()
        with tr.step("search") as s:
            found = search(query)
            s["note"] = f"{len(found)} фрагментов"
        with tr.step("llm"):
            text = call_llm(...)
        return {"text": text, "sources": [...], "lead": None, "trace": tr.steps}

Рекомендуемые имена шагов: search, rerank, llm, make_lead.
Не кладите в note промпты, ключи и адреса API.
"""

import time
from contextlib import contextmanager


class Trace:
    def __init__(self):
        self.steps = []

    @contextmanager
    def step(self, name, note=""):
        started = time.perf_counter()
        info = {"note": note}
        try:
            yield info
        finally:
            self.steps.append(
                {"step": name, "ms": int((time.perf_counter() - started) * 1000), "note": info["note"]}
            )
