"""Поиск по базе знаний. Единственное место, где строится индекс.

    from retrieval import search
    search("вопрос клиента", k=8)  ->  [(имя_файла, текст_фрагмента), ...]

Индекс строится один раз при первом вызове и дальше переиспользуется.
Если ключ API не задан, поиск работает только по словам (BM25).
"""

from threading import Lock

import httpx

import solution

_lock = Lock()
_state = {}


def get_retriever():
    """Возвращает общий Retriever или None, если в базе нет фрагментов."""
    with _lock:
        if "retriever" not in _state:
            path = solution.find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
            chunks = [c for name, text in solution.load_kb(path) for c in solution.chunk_doc(name, text)]
            if not chunks:
                return None
            original_embed = solution.embed
            if not (solution.BASE_URL and solution.API_KEY):
                solution.embed = lambda *args, **kwargs: None  # без ключа не ждём недоступный API
            try:
                _state["retriever"] = solution.Retriever(httpx.Client(), chunks)
            finally:
                solution.embed = original_embed
        return _state["retriever"]


def search(query, k=None):
    """Лучшие фрагменты базы знаний для запроса."""
    retriever = get_retriever()
    if retriever is None:
        return []
    found = retriever.search(query, k or solution.TOP_K)
    if retriever.emb is None:  # эмбеддингов нет: оставляем фрагменты, у которых есть общие слова с запросом
        terms = set(solution.toks(query))
        found = [(name, text) for name, text in found if terms & set(solution.toks(text))]
    return found
