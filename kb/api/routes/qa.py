"""提问、会话与反馈。"""

import threading

from kb.http import HttpError
from kb.service import KnowledgeBase, SessionError


def register_qa(app, kb: KnowledgeBase, db_lock: threading.Lock) -> None:
    from fastapi import Form
    from fastapi.responses import JSONResponse

    @app.post("/sessions")
    def open_session(user_id: str = Form(...)) -> JSONResponse:
        try:
            with db_lock:
                session_id = kb.open_session(user_id)
        except SessionError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"session_id": session_id, "user_id": user_id.strip()})

    @app.get("/sessions/{session_id}")
    def read_session(session_id: str, user_id: str) -> JSONResponse:
        with db_lock:
            found = kb.get_session(user_id, session_id)
        if found is None:
            return JSONResponse({"detail": "会话不存在"}, status_code=404)
        return JSONResponse(found)

    @app.post("/ask")
    def ask(
        question: str = Form(...),
        user_id: str = Form(""),
        session_id: str = Form(""),
    ) -> JSONResponse:
        try:
            with db_lock:
                if user_id.strip() or session_id.strip():
                    answer = kb.ask(question, user_id, session_id)
                else:
                    answer = kb.ask(question)
        except SessionError:
            return JSONResponse({"detail": "会话不存在"}, status_code=404)
        except HttpError:
            return JSONResponse(
                {"detail": "回答模型响应超时。模型可能正在加载，请过几秒再问一次。"},
                status_code=503,
            )
        return JSONResponse({
            "question": answer.question,
            "text": answer.text,
            "refused": answer.refused,
            "confidence": round(answer.confidence, 3),
            "session_id": answer.session_id,
            "model": answer.model,
            "degraded": answer.degraded,
            "trace": answer.trace,
            "citations": [
                {"chunk_id": c.chunk_id, "doc_name": c.doc_name,
                 "section_path": c.section_path, "page": c.page, "text": c.text}
                for c in answer.citations
            ],
        })

    @app.post("/feedback")
    def feedback(question: str = Form(...), useful: bool = Form(...)) -> JSONResponse:
        with db_lock:
            kb.store.add_feedback(question, useful, False, [])
        return JSONResponse({"ok": True})
