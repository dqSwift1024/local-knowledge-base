"""运行状态与健康检查。"""

import threading
from pathlib import Path

from kb.config import Settings
from kb.service import KnowledgeBase


def register_system(app, kb: KnowledgeBase, settings: Settings, db_lock: threading.Lock) -> None:
    from fastapi.responses import JSONResponse

    @app.get("/status")
    def status() -> JSONResponse:
        with db_lock:
            docs = kb.store.list_docs()
            jobs = kb.store.list_jobs()
            health = kb.health()
        busy = any(j.status in ("pending", "running", "failed") for j in jobs)
        running_doc = next((d for d in docs if d.status == "running"), None)
        return JSONResponse({
            "busy": busy,
            "health": health,
            "runtime": {
                "embedder": kb.embedder.name,
                "answerer": "摘录" if settings.answerer == "offline" else settings.chat_model,
            },
            "active_stage": running_doc.stage if running_doc else "",
            "docs": [
                {"doc_name": d.doc_name, "version": d.version, "is_latest": d.is_latest,
                 "content_hash": d.content_hash,
                 "status": d.status, "stage": d.stage, "chunks": d.chunk_count,
                 "error": d.error}
                for d in docs if d.is_latest
            ],
            "jobs": [
                {"id": j.id, "status": j.status, "stage": j.stage, "attempts": j.attempts,
                 "path": Path(j.path).name, "error": j.error, "updated_at": j.updated_at}
                for j in jobs[-20:]
            ],
        })

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(kb.health())
