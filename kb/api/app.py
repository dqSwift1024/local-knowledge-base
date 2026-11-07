"""应用装配：挂上 JSON 接口，并把独立前端作为静态资源提供。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from kb.api.routes import register_documents, register_qa, register_system
from kb.api.worker import start_worker
from kb.config import Settings, load_settings
from kb.service import KnowledgeBase


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def create_app(settings: Optional[Settings] = None):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    settings = settings or load_settings()
    kb = KnowledgeBase(settings)
    app = FastAPI(title="本地知识库")
    db_lock = threading.Lock()
    start_worker(settings, kb.embedder)

    register_documents(app, kb, settings, db_lock)
    register_qa(app, kb, db_lock)
    register_system(app, kb, settings, db_lock)

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    app.state.kb = kb
    return app
