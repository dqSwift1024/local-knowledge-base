"""SQLite 存储层：文档版本、块（父子）、向量、关键词索引、任务、反馈。

单人本地用 SQLite 足够；接口按“增删查”组织，日后换 PostgreSQL + pgvector
时替换本文件即可，上层（检索、问答、管线、任务）不动。
向量以 float32 bytes 存入块表，检索时按最新版加载进 numpy 做余弦；
关键词走 FTS5，中文已在 textutil 里切成字二元组。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kb.model import Chunk
from kb.textutil import index_text


STAGES = ["parse", "clean", "chunk", "embed", "index", "done"]


@dataclass
class DocVersion:
    doc_name: str
    content_hash: str
    version: int
    is_latest: bool
    parser: str
    stage: str
    status: str
    chunk_count: int
    created_at: float
    error: str = ""


@dataclass
class Job:
    id: int
    path: str
    status: str
    stage: str
    content_hash: str
    error: str
    attempts: int
    updated_at: float


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self._db.close()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                content_hash TEXT PRIMARY KEY,
                doc_name     TEXT NOT NULL,
                version      INTEGER NOT NULL,
                is_latest    INTEGER NOT NULL DEFAULT 0,
                parser       TEXT DEFAULT '',
                stage        TEXT NOT NULL DEFAULT 'parse',
                status       TEXT NOT NULL DEFAULT 'pending',
                chunk_count  INTEGER NOT NULL DEFAULT 0,
                error        TEXT DEFAULT '',
                meta         TEXT DEFAULT '{}',
                created_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_docs_name ON docs(doc_name);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                doc_name     TEXT NOT NULL,
                section_path TEXT NOT NULL,
                page         INTEGER,
                text         TEXT NOT NULL,
                parent_text  TEXT DEFAULT '',
                embedding    BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                tokens, chunk_id UNINDEXED, content_hash UNINDEXED
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                path         TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                stage        TEXT NOT NULL DEFAULT 'queued',
                content_hash TEXT DEFAULT '',
                error        TEXT DEFAULT '',
                attempts     INTEGER NOT NULL DEFAULT 0,
                updated_at   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                question   TEXT NOT NULL,
                useful     INTEGER NOT NULL,
                refused    INTEGER NOT NULL,
                citations  TEXT DEFAULT '[]',
                created_at REAL NOT NULL
            );
            """
        )
        self._db.commit()

    # ---- 文档版本 ----

    def get_doc(self, content_hash: str) -> DocVersion | None:
        row = self._db.execute("SELECT * FROM docs WHERE content_hash=?", (content_hash,)).fetchone()
        return _row_to_docversion(row) if row else None

    def register_version(self, doc_name: str, content_hash: str, parser: str, meta: dict) -> DocVersion:
        """登记一次导入。同名新内容视为新版本并置为最新；同内容直接复用。"""
        existing = self.get_doc(content_hash)
        if existing is not None:
            return existing
        row = self._db.execute(
            "SELECT MAX(version) AS v FROM docs WHERE doc_name=?", (doc_name,)
        ).fetchone()
        next_version = (row["v"] or 0) + 1
        self._db.execute("UPDATE docs SET is_latest=0 WHERE doc_name=?", (doc_name,))
        self._db.execute(
            "INSERT INTO docs(content_hash, doc_name, version, is_latest, parser, stage, status, "
            "chunk_count, meta, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (content_hash, doc_name, next_version, 1, parser, "parse", "pending", 0,
             json.dumps(meta, ensure_ascii=False), time.time()),
        )
        self._db.commit()
        return self.get_doc(content_hash)  # type: ignore[return-value]

    def set_stage(self, content_hash: str, stage: str, status: str = "running", error: str = "") -> None:
        self._db.execute(
            "UPDATE docs SET stage=?, status=?, error=? WHERE content_hash=?",
            (stage, status, error, content_hash),
        )
        self._db.commit()

    def list_docs(self) -> list[DocVersion]:
        rows = self._db.execute("SELECT * FROM docs ORDER BY doc_name, version").fetchall()
        return [_row_to_docversion(r) for r in rows]

    def versions_of(self, doc_name: str) -> list[DocVersion]:
        rows = self._db.execute(
            "SELECT * FROM docs WHERE doc_name=? ORDER BY version", (doc_name,)
        ).fetchall()
        return [_row_to_docversion(r) for r in rows]

    def set_latest(self, doc_name: str, version: int) -> bool:
        """把某个历史版本切回最新（回滚）。"""
        row = self._db.execute(
            "SELECT content_hash FROM docs WHERE doc_name=? AND version=?", (doc_name, version)
        ).fetchone()
        if row is None:
            return False
        self._db.execute("UPDATE docs SET is_latest=0 WHERE doc_name=?", (doc_name,))
        self._db.execute(
            "UPDATE docs SET is_latest=1 WHERE doc_name=? AND version=?", (doc_name, version)
        )
        self._db.commit()
        return True

    def delete_version(self, content_hash: str) -> bool:
        """删除某个版本及其块与索引；若删的是最新版，则回退到剩余最高版本。"""
        doc = self.get_doc(content_hash)
        if doc is None:
            return False
        self._clear_chunks(content_hash)
        self._db.execute("DELETE FROM docs WHERE content_hash=?", (content_hash,))
        if doc.is_latest:
            row = self._db.execute(
                "SELECT content_hash FROM docs WHERE doc_name=? ORDER BY version DESC LIMIT 1",
                (doc.doc_name,),
            ).fetchone()
            if row is not None:
                self._db.execute("UPDATE docs SET is_latest=1 WHERE content_hash=?", (row["content_hash"],))
        self._db.commit()
        return True

    def purge_superseded(self, doc_name: str) -> int:
        """清理某文档所有非最新版本，释放空间。返回清理的版本数。"""
        rows = self._db.execute(
            "SELECT content_hash FROM docs WHERE doc_name=? AND is_latest=0", (doc_name,)
        ).fetchall()
        for row in rows:
            self._clear_chunks(row["content_hash"])
            self._db.execute("DELETE FROM docs WHERE content_hash=?", (row["content_hash"],))
        self._db.commit()
        return len(rows)

    # ---- 块与索引（批量、可续跑）----

    def existing_chunk_ids(self, content_hash: str) -> set[str]:
        rows = self._db.execute(
            "SELECT chunk_id FROM chunks WHERE content_hash=?", (content_hash,)
        ).fetchall()
        return {r["chunk_id"] for r in rows}

    def upsert_chunk_batch(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """写入一批块与向量。chunk_id 确定性，故可安全重入（断点续跑）。"""
        for chunk, vector in zip(chunks, embeddings):
            self._db.execute(
                "INSERT OR REPLACE INTO chunks(chunk_id, content_hash, doc_name, section_path, page, "
                "text, parent_text, embedding) VALUES(?,?,?,?,?,?,?,?)",
                (chunk.chunk_id, chunk.content_hash, chunk.doc_name, chunk.section_path, chunk.page,
                 chunk.text, chunk.parent_text, vector.astype(np.float32).tobytes()),
            )
            self._db.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk.chunk_id,))
            self._db.execute(
                "INSERT INTO chunks_fts(tokens, chunk_id, content_hash) VALUES(?,?,?)",
                (index_text(chunk.text), chunk.chunk_id, chunk.content_hash),
            )
        self._db.commit()

    def finalize_chunk_count(self, content_hash: str) -> None:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE content_hash=?", (content_hash,)
        ).fetchone()
        self._db.execute("UPDATE docs SET chunk_count=? WHERE content_hash=?", (row["n"], content_hash))
        self._db.commit()

    def _clear_chunks(self, content_hash: str) -> None:
        self._db.execute("DELETE FROM chunks WHERE content_hash=?", (content_hash,))
        self._db.execute("DELETE FROM chunks_fts WHERE content_hash=?", (content_hash,))

    def clear_chunks(self, content_hash: str) -> None:
        self._clear_chunks(content_hash)
        self._db.commit()

    def load_latest_matrix(self, dim: int) -> tuple[np.ndarray, list[sqlite3.Row]]:
        """加载所有最新版块的向量矩阵。原型级实现（全量加载）。"""
        rows = self._db.execute(
            "SELECT c.chunk_id, c.doc_name, c.section_path, c.page, c.text, c.parent_text, c.embedding "
            "FROM chunks c JOIN docs d ON c.content_hash=d.content_hash WHERE d.is_latest=1"
        ).fetchall()
        if not rows:
            return np.zeros((0, dim), dtype=np.float32), []
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return matrix, rows

    def vector_search(self, query_vec: np.ndarray, top_n: int) -> list[dict]:
        """向量近邻检索。SQLite 版用 numpy 全量余弦；pgvector 版用 HNSW 索引。

        检索接口统一下沉到存储层：上层 Retriever 只调用本方法，
        换成 PgVectorStore 时无需改动检索逻辑。
        """
        matrix, rows = self.load_latest_matrix(query_vec.shape[0])
        if not rows:
            return []
        cosines = matrix @ query_vec
        order = np.argsort(-cosines)[:top_n]
        results: list[dict] = []
        for idx in order:
            row = rows[int(idx)]
            results.append({
                "chunk_id": row["chunk_id"], "doc_name": row["doc_name"],
                "section_path": row["section_path"], "page": row["page"],
                "text": row["text"], "parent_text": row["parent_text"],
                "vec_cos": float(cosines[int(idx)]),
            })
        return results

    def keyword_search(self, match_expr: str, limit: int) -> list[sqlite3.Row]:
        if not match_expr:
            return []
        return self._db.execute(
            "SELECT f.chunk_id, c.doc_name, c.section_path, c.page, c.text, c.parent_text, "
            "bm25(chunks_fts) AS score FROM chunks_fts f JOIN chunks c ON f.chunk_id=c.chunk_id "
            "JOIN docs d ON c.content_hash=d.content_hash "
            "WHERE chunks_fts MATCH ? AND d.is_latest=1 ORDER BY score LIMIT ?",
            (match_expr, limit),
        ).fetchall()

    # ---- 任务队列 ----

    def enqueue_job(self, path: str) -> int:
        cursor = self._db.execute(
            "INSERT INTO jobs(path, status, stage, updated_at) VALUES(?,?,?,?)",
            (path, "pending", "queued", time.time()),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def next_pending_job(self) -> Job | None:
        row = self._db.execute(
            "SELECT * FROM jobs WHERE status IN ('pending','failed') "
            "ORDER BY (status='failed'), id LIMIT 1"
        ).fetchone()
        return _row_to_job(row) if row else None

    def update_job(self, job_id: int, *, status: str, stage: str,
                   content_hash: str = "", error: str = "", bump_attempt: bool = False) -> None:
        attempt_sql = ", attempts=attempts+1" if bump_attempt else ""
        self._db.execute(
            f"UPDATE jobs SET status=?, stage=?, content_hash=?, error=?, updated_at=?{attempt_sql} "
            "WHERE id=?",
            (status, stage, content_hash, error, time.time(), job_id),
        )
        self._db.commit()

    def list_jobs(self) -> list[Job]:
        rows = self._db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        return [_row_to_job(r) for r in rows]

    # ---- 反馈 ----

    def add_feedback(self, question: str, useful: bool, refused: bool, citations: list[str]) -> None:
        self._db.execute(
            "INSERT INTO feedback(question, useful, refused, citations, created_at) VALUES(?,?,?,?,?)",
            (question, int(useful), int(refused), json.dumps(citations, ensure_ascii=False), time.time()),
        )
        self._db.commit()

    def feedback_stats(self) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(useful),0) AS useful FROM feedback"
        ).fetchone()
        return {"total": row["total"], "useful": row["useful"]}

    # ---- 运维 ----

    def health(self) -> dict:
        def count(sql: str) -> int:
            return int(self._db.execute(sql).fetchone()[0])

        return {
            "ok": True,
            "docs": count("SELECT COUNT(*) FROM docs"),
            "latest_docs": count("SELECT COUNT(*) FROM docs WHERE is_latest=1"),
            "chunks": count("SELECT COUNT(*) FROM chunks"),
            "jobs_pending": count("SELECT COUNT(*) FROM jobs WHERE status IN ('pending','failed')"),
            "feedback": count("SELECT COUNT(*) FROM feedback"),
            "db_bytes": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0,
        }

    def backup(self, dest_path: str) -> str:
        """一致性备份：SQLite online backup，运行中也安全。"""
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(dest_path) as dest:
            self._db.backup(dest)
        return dest_path


def _row_to_docversion(row: sqlite3.Row) -> DocVersion:
    return DocVersion(
        doc_name=row["doc_name"], content_hash=row["content_hash"], version=row["version"],
        is_latest=bool(row["is_latest"]), parser=row["parser"], stage=row["stage"],
        status=row["status"], chunk_count=row["chunk_count"], created_at=row["created_at"],
        error=row["error"],
    )


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"], path=row["path"], status=row["status"], stage=row["stage"],
        content_hash=row["content_hash"], error=row["error"], attempts=row["attempts"],
        updated_at=row["updated_at"],
    )
