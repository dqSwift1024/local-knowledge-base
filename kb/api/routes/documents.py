"""文档导入与删除。"""

import threading
import time
from pathlib import Path

from kb.config import Settings
from kb.service import KnowledgeBase


def register_documents(app, kb: KnowledgeBase, settings: Settings, db_lock: threading.Lock) -> None:
    from fastapi import File, UploadFile
    from fastapi.responses import JSONResponse

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)) -> JSONResponse:
        name = Path(file.filename or "upload").name
        inbox = Path(settings.db_path).parent / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        target = inbox / f"{int(time.time() * 1000)}_{name}"
        target.write_bytes(await file.read())
        with db_lock:
            job_id = kb.store.enqueue_job(str(target))
        return JSONResponse({"job_id": job_id, "doc_name": name, "status": "queued"})

    @app.delete("/documents/{content_hash}")
    def delete_document(content_hash: str) -> JSONResponse:
        with db_lock:
            doc = kb.store.get_doc(content_hash)
            if doc is None:
                return JSONResponse({"ok": False, "error": "文档不存在"}, status_code=404)
            deleted = kb.delete_version(content_hash)
        return JSONResponse({
            "ok": deleted,
            "doc_name": doc.doc_name,
            "content_hash": content_hash,
        })
