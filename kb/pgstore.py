"""PostgreSQL + pgvector 存储实现（规模化路径）。

与 kb.store.Store 方法签名一致，向量检索用 pgvector 的 HNSW 索引，
关键词检索用 Postgres 全文检索（对中文同样走 textutil 的字二元组分词）。

注意：本实现需要可用的 PostgreSQL 服务并启用 vector 扩展，以及 psycopg、
pgvector 两个 Python 包。当前开发环境无 Postgres 服务，故本文件已实现但未在
该环境运行；在装好依赖与数据库后，通过 KB_STORE=postgres、KB_PG_DSN=... 启用。
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from kb.config import Settings
from kb.model import Chunk
from kb.store import DocVersion, Job
from kb.textutil import index_text, tokenize


class PgVectorStore:
    def __init__(self, settings: Settings) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "使用 postgres 存储需要 psycopg 与 pgvector：pip install 'psycopg[binary]' pgvector"
            ) from exc
        if not settings.pg_dsn:
            raise RuntimeError("使用 postgres 存储需要设置 KB_PG_DSN")
        self._settings = settings
        self._dim = settings.embed_dim
        self._conn = psycopg.connect(settings.pg_dsn, autocommit=True, row_factory=_dict_row())
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                content_hash TEXT PRIMARY KEY, doc_name TEXT NOT NULL, version INTEGER NOT NULL,
                is_latest BOOLEAN NOT NULL DEFAULT FALSE, parser TEXT DEFAULT '',
                stage TEXT NOT NULL DEFAULT 'parse', status TEXT NOT NULL DEFAULT 'pending',
                chunk_count INTEGER NOT NULL DEFAULT 0, error TEXT DEFAULT '',
                meta JSONB DEFAULT '{}', created_at DOUBLE PRECISION NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL REFERENCES docs(content_hash)
                    ON DELETE CASCADE,
                doc_name TEXT NOT NULL, section_path TEXT NOT NULL, page INTEGER,
                text TEXT NOT NULL, parent_text TEXT DEFAULT '', tokens TEXT DEFAULT '',
                embedding vector({self._dim})
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON chunks USING hnsw "
            f"(embedding vector_cosine_ops) WITH (m={self._settings.hnsw_m}, "
            f"ef_construction={self._settings.hnsw_ef_construction})"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin "
            "(to_tsvector('simple', tokens))"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY, path TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                stage TEXT NOT NULL DEFAULT 'queued', content_hash TEXT DEFAULT '',
                error TEXT DEFAULT '', attempts INTEGER NOT NULL DEFAULT 0,
                updated_at DOUBLE PRECISION NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY, question TEXT NOT NULL, useful INTEGER NOT NULL,
                refused INTEGER NOT NULL, citations TEXT DEFAULT '[]',
                created_at DOUBLE PRECISION NOT NULL
            )
            """
        )

    # ---- 文档版本 ----

    def get_doc(self, content_hash: str) -> DocVersion | None:
        row = self._conn.execute("SELECT * FROM docs WHERE content_hash=%s", (content_hash,)).fetchone()
        return _to_docversion(row) if row else None

    def register_version(self, doc_name: str, content_hash: str, parser: str, meta: dict) -> DocVersion:
        existing = self.get_doc(content_hash)
        if existing is not None:
            return existing
        row = self._conn.execute("SELECT MAX(version) AS v FROM docs WHERE doc_name=%s", (doc_name,)).fetchone()
        next_version = (row["v"] or 0) + 1
        self._conn.execute("UPDATE docs SET is_latest=FALSE WHERE doc_name=%s", (doc_name,))
        self._conn.execute(
            "INSERT INTO docs(content_hash, doc_name, version, is_latest, parser, stage, status, "
            "chunk_count, meta, created_at) VALUES(%s,%s,%s,TRUE,%s,'parse','pending',0,%s,%s)",
            (content_hash, doc_name, next_version, parser, json.dumps(meta, ensure_ascii=False), time.time()),
        )
        return self.get_doc(content_hash)  # type: ignore[return-value]

    def set_stage(self, content_hash: str, stage: str, status: str = "running", error: str = "") -> None:
        self._conn.execute("UPDATE docs SET stage=%s, status=%s, error=%s WHERE content_hash=%s",
                           (stage, status, error, content_hash))

    def list_docs(self) -> list[DocVersion]:
        rows = self._conn.execute("SELECT * FROM docs ORDER BY doc_name, version").fetchall()
        return [_to_docversion(r) for r in rows]

    def versions_of(self, doc_name: str) -> list[DocVersion]:
        rows = self._conn.execute("SELECT * FROM docs WHERE doc_name=%s ORDER BY version", (doc_name,)).fetchall()
        return [_to_docversion(r) for r in rows]

    def set_latest(self, doc_name: str, version: int) -> bool:
        row = self._conn.execute("SELECT content_hash FROM docs WHERE doc_name=%s AND version=%s",
                                 (doc_name, version)).fetchone()
        if row is None:
            return False
        self._conn.execute("UPDATE docs SET is_latest=FALSE WHERE doc_name=%s", (doc_name,))
        self._conn.execute("UPDATE docs SET is_latest=TRUE WHERE doc_name=%s AND version=%s",
                           (doc_name, version))
        return True

    def delete_version(self, content_hash: str) -> bool:
        doc = self.get_doc(content_hash)
        if doc is None:
            return False
        self._conn.execute("DELETE FROM docs WHERE content_hash=%s", (content_hash,))  # 级联删 chunks
        if doc.is_latest:
            row = self._conn.execute(
                "SELECT content_hash FROM docs WHERE doc_name=%s ORDER BY version DESC LIMIT 1",
                (doc.doc_name,)).fetchone()
            if row is not None:
                self._conn.execute("UPDATE docs SET is_latest=TRUE WHERE content_hash=%s",
                                   (row["content_hash"],))
        return True

    def purge_superseded(self, doc_name: str) -> int:
        rows = self._conn.execute("SELECT content_hash FROM docs WHERE doc_name=%s AND is_latest=FALSE",
                                 (doc_name,)).fetchall()
        for row in rows:
            self._conn.execute("DELETE FROM docs WHERE content_hash=%s", (row["content_hash"],))
        return len(rows)

    # ---- 块与索引 ----

    def existing_chunk_ids(self, content_hash: str) -> set[str]:
        rows = self._conn.execute("SELECT chunk_id FROM chunks WHERE content_hash=%s", (content_hash,)).fetchall()
        return {r["chunk_id"] for r in rows}

    def upsert_chunk_batch(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for chunk, vector in zip(chunks, embeddings):
            self._conn.execute(
                "INSERT INTO chunks(chunk_id, content_hash, doc_name, section_path, page, text, "
                "parent_text, tokens, embedding) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET text=EXCLUDED.text, "
                "parent_text=EXCLUDED.parent_text, tokens=EXCLUDED.tokens, embedding=EXCLUDED.embedding",
                (chunk.chunk_id, chunk.content_hash, chunk.doc_name, chunk.section_path, chunk.page,
                 chunk.text, chunk.parent_text, index_text(chunk.text),
                 np.asarray(vector, dtype=np.float32)),
            )

    def finalize_chunk_count(self, content_hash: str) -> None:
        self._conn.execute(
            "UPDATE docs SET chunk_count=(SELECT COUNT(*) FROM chunks WHERE content_hash=%s) "
            "WHERE content_hash=%s", (content_hash, content_hash))

    def clear_chunks(self, content_hash: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE content_hash=%s", (content_hash,))

    def vector_search(self, query_vec: np.ndarray, top_n: int) -> list[dict]:
        self._conn.execute(f"SET hnsw.ef_search = {self._settings.hnsw_ef_search}")
        rows = self._conn.execute(
            "SELECT c.chunk_id, c.doc_name, c.section_path, c.page, c.text, c.parent_text, "
            "1 - (c.embedding <=> %s) AS vec_cos FROM chunks c "
            "JOIN docs d ON c.content_hash=d.content_hash WHERE d.is_latest "
            "ORDER BY c.embedding <=> %s LIMIT %s",
            (np.asarray(query_vec, dtype=np.float32), np.asarray(query_vec, dtype=np.float32), top_n),
        ).fetchall()
        return [dict(r) for r in rows]

    def keyword_search(self, match_expr: str, limit: int) -> list[dict]:
        tokens = match_expr.replace('"', "").replace(" OR ", " | ")
        if not tokens.strip():
            return []
        rows = self._conn.execute(
            "SELECT c.chunk_id, c.doc_name, c.section_path, c.page, c.text, c.parent_text, "
            "ts_rank(to_tsvector('simple', c.tokens), to_tsquery('simple', %s)) AS score "
            "FROM chunks c JOIN docs d ON c.content_hash=d.content_hash WHERE d.is_latest "
            "AND to_tsvector('simple', c.tokens) @@ to_tsquery('simple', %s) "
            "ORDER BY score DESC LIMIT %s",
            (tokens, tokens, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- 任务队列 ----

    def enqueue_job(self, path: str) -> int:
        row = self._conn.execute(
            "INSERT INTO jobs(path, status, stage, updated_at) VALUES(%s,'pending','queued',%s) RETURNING id",
            (path, time.time())).fetchone()
        return int(row["id"])

    def next_pending_job(self) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN ('pending','failed') "
            "ORDER BY (status='failed'), id LIMIT 1").fetchone()
        return _to_job(row) if row else None

    def update_job(self, job_id: int, *, status: str, stage: str, content_hash: str = "",
                   error: str = "", bump_attempt: bool = False) -> None:
        bump = ", attempts=attempts+1" if bump_attempt else ""
        self._conn.execute(
            f"UPDATE jobs SET status=%s, stage=%s, content_hash=%s, error=%s, updated_at=%s{bump} WHERE id=%s",
            (status, stage, content_hash, error, time.time(), job_id))

    def list_jobs(self) -> list[Job]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        return [_to_job(r) for r in rows]

    # ---- 反馈与运维 ----

    def add_feedback(self, question: str, useful: bool, refused: bool, citations: list[str]) -> None:
        self._conn.execute(
            "INSERT INTO feedback(question, useful, refused, citations, created_at) VALUES(%s,%s,%s,%s,%s)",
            (question, int(useful), int(refused), json.dumps(citations, ensure_ascii=False), time.time()))

    def feedback_stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(useful),0) AS useful FROM feedback").fetchone()
        return {"total": row["total"], "useful": row["useful"]}

    def health(self) -> dict:
        def count(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()["c"])

        return {
            "ok": True,
            "docs": count("SELECT COUNT(*) AS c FROM docs"),
            "latest_docs": count("SELECT COUNT(*) AS c FROM docs WHERE is_latest"),
            "chunks": count("SELECT COUNT(*) AS c FROM chunks"),
            "jobs_pending": count("SELECT COUNT(*) AS c FROM jobs WHERE status IN ('pending','failed')"),
            "feedback": count("SELECT COUNT(*) AS c FROM feedback"),
            "db_bytes": count("SELECT pg_database_size(current_database()) AS c"),
        }

    def backup(self, dest_path: str) -> str:
        raise NotImplementedError("PostgreSQL 备份请用 pg_dump，例如：pg_dump <dsn> > backup.sql")


def _dict_row():
    from psycopg.rows import dict_row
    return dict_row


def _to_docversion(row: dict[str, Any]) -> DocVersion:
    return DocVersion(
        doc_name=row["doc_name"], content_hash=row["content_hash"], version=row["version"],
        is_latest=bool(row["is_latest"]), parser=row["parser"], stage=row["stage"],
        status=row["status"], chunk_count=row["chunk_count"], created_at=row["created_at"],
        error=row.get("error", ""),
    )


def _to_job(row: dict[str, Any]) -> Job:
    return Job(
        id=row["id"], path=row["path"], status=row["status"], stage=row["stage"],
        content_hash=row.get("content_hash", ""), error=row.get("error", ""),
        attempts=row["attempts"], updated_at=row["updated_at"],
    )
