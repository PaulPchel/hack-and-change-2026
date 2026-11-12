"""Веб-сервер чата МТС Energytool (FastAPI).

Контракт с консультантом: consultant.answer(history) -> dict
    history: [{"role": "user"|"assistant", "content": "..."}], последнее — вопрос клиента
    возвращает {"text": str, "sources": [имена файлов], "lead": None | {"name","phone","company","summary"}}
"""

import importlib
import importlib.util
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import db

log = logging.getLogger("app")
BASE = Path(__file__).resolve().parent
ERROR_TEXT = "Извините, сейчас не получилось ответить. Попробуйте ещё раз или оставьте заявку специалисту."

# Настоящий consultant.py, если он есть; иначе заглушка (или принудительно USE_STUB=1)
if os.getenv("USE_STUB") != "1" and importlib.util.find_spec("consultant"):
    MODE = "consultant"
    consultant = importlib.import_module("consultant")
else:
    MODE = "stub"
    consultant = importlib.import_module("stub_consultant")

db.init()
app = FastAPI(title="МТС Energytool: ИИ-консультант")


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


def _clean_lead(lead):
    if not isinstance(lead, dict):
        return None
    out = {k: str(lead.get(k) or "").strip() for k in ("name", "phone", "company", "summary")}
    return out if (out["name"] or out["phone"]) else None


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "consultant": MODE}


@app.post("/api/session")
def new_session():
    """«Новый диалог»: пустая сессия. Прежние остаются в базе."""
    return {"session_id": db.new_session()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    sid, text = req.session_id, req.message.strip()
    if not text:
        raise HTTPException(422, "Пустое сообщение")
    if not db.session_exists(sid):
        raise HTTPException(404, "Диалог не найден")

    # Контекст — только сообщения этой сессии
    history = [{"role": m["role"], "content": m["content"]} for m in db.get_messages(sid)]
    history.append({"role": "user", "content": text})
    db.add_message(sid, "user", text)

    started = time.perf_counter()
    try:
        res = consultant.answer(history)
    except Exception:
        log.exception("consultant failed")
        res = {"text": ERROR_TEXT, "sources": [], "lead": None}
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    reply = str(res.get("text") or ERROR_TEXT)
    sources = [str(s) for s in (res.get("sources") or [])]
    lead = _clean_lead(res.get("lead"))

    db.add_message(sid, "assistant", reply, elapsed_ms, sources)
    if lead:
        db.save_lead(sid, lead)
    return {"text": reply, "sources": sources, "elapsed_ms": elapsed_ms, "lead": lead}


@app.get("/api/sessions")
def sessions():
    return db.list_sessions()


@app.get("/api/sessions/{sid}")
def session_messages(sid: str):
    if not db.session_exists(sid):
        raise HTTPException(404, "Диалог не найден")
    return db.get_messages(sid)


@app.get("/api/leads")
def leads():
    return db.list_leads()
