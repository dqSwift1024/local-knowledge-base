"""导入管线：解析 → 清洗 → 父子切分 → 批量嵌入 → 索引。

- 分阶段写入 docs.stage；失败时记录“真实到达的阶段”，便于定位与重跑。
- 嵌入按批进行；chunk_id 确定性，已入库的子块会被跳过，实现断点续跑。
- 同一文件（内容哈希）已完成则跳过；同名新内容登记为新版本并置最新。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from kb.chunk import chunk_document, iter_chunks
from kb.clean import clean_blocks
from kb.config import Settings
from kb.embed import Embedder
from kb.model import ParsedDoc
from kb.parsers import parse_file
from kb.parsers.stream import STREAM_EXTENSIONS, hash_file, iter_blocks
from kb.store import Store


log = logging.getLogger("kb.pipeline")


@dataclass
class IngestReport:
    doc_name: str
    content_hash: str
    version: int
    status: str
    chunk_count: int
    warnings: list[str]
    skipped: bool = False


class Pipeline:
    def __init__(self, store: Store, embedder: Embedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    def ingest(self, path: str | Path) -> IngestReport:
        file_path = Path(path)
        if self._should_stream(file_path):
            return self._ingest_stream(file_path)
        doc = parse_file(file_path)

        existing = self._store.get_doc(doc.content_hash)
        if existing is not None and existing.status == "done":
            log.info("skip (already indexed): %s", doc.source_name)
            return IngestReport(existing.doc_name, existing.content_hash, existing.version,
                                "done", existing.chunk_count, [], skipped=True)

        version = self._store.register_version(doc.source_name, doc.content_hash, doc.parser, doc.meta)
        warnings = list(doc.warnings)
        stage = "clean"
        try:
            self._store.set_stage(doc.content_hash, stage)
            blocks, clean_warnings = clean_blocks(doc)
            warnings += clean_warnings

            stage = "chunk"
            self._store.set_stage(doc.content_hash, stage)
            chunks = chunk_document(
                doc, blocks,
                child_chars=self._settings.child_chars,
                overlap_chars=self._settings.overlap_chars,
                parent_max_chars=self._settings.parent_max_chars,
            )
            if not chunks:
                self._store.set_stage(doc.content_hash, "done", "empty", "切分后无内容")
                return IngestReport(doc.source_name, doc.content_hash, version.version,
                                    "empty", 0, warnings + ["切分后无内容"])

            stage = "embed"
            self._store.set_stage(doc.content_hash, stage)
            self._embed_and_index(doc.content_hash, chunks)

            stage = "index"
            self._store.set_stage(doc.content_hash, stage)
            self._store.finalize_chunk_count(doc.content_hash)

            self._store.set_stage(doc.content_hash, "done", "done")
            log.info("ingested %s (v%s, %s chunks)", doc.source_name, version.version, len(chunks))
            return IngestReport(doc.source_name, doc.content_hash, version.version,
                                "done", len(chunks), warnings)
        except Exception as exc:  # noqa: BLE001 — 记录真实失败阶段，便于重跑恢复
            self._store.set_stage(doc.content_hash, stage, "failed", str(exc))
            log.warning("ingest failed at stage %s: %s", stage, exc)
            raise

    def _embed_and_index(self, content_hash: str, chunks: list) -> None:
        """只嵌入尚未入库的子块，按批提交。中途失败后重跑会自动接着做。"""
        done = self._store.existing_chunk_ids(content_hash)
        pending = [c for c in chunks if c.chunk_id not in done]
        batch = max(1, self._settings.embed_batch)
        for start in range(0, len(pending), batch):
            group = pending[start : start + batch]
            vectors = self._embedder.embed([c.embed_text for c in group])
            self._store.upsert_chunk_batch(group, vectors)

    # ---- 流式导入（大文件）----

    def _should_stream(self, file_path: Path) -> bool:
        if file_path.suffix.lower() not in STREAM_EXTENSIONS:
            return False
        try:
            return file_path.stat().st_size > self._settings.stream_threshold_bytes
        except OSError:
            return False

    def _ingest_stream(self, file_path: Path) -> IngestReport:
        """纯文本大文件：流式哈希 + 流式解析 + 流式切分 + 批量嵌入，内存有界。

        不做跨页页眉检测（流式下无法回看全局），仅逐块轻清洗。
        """
        content_hash = hash_file(file_path)
        existing = self._store.get_doc(content_hash)
        if existing is not None and existing.status == "done":
            return IngestReport(existing.doc_name, existing.content_hash, existing.version,
                                "done", existing.chunk_count, [], skipped=True)

        meta = {"size_bytes": file_path.stat().st_size, "ext": file_path.suffix.lower()}
        version = self._store.register_version(file_path.name, content_hash, "text-stream", meta)
        doc = ParsedDoc(source_name=file_path.name, content_hash=content_hash, blocks=[],
                        parser="text-stream")
        stage = "embed"
        try:
            self._store.set_stage(content_hash, stage)
            chunk_iter = iter_chunks(
                doc, iter_blocks(file_path),
                child_chars=self._settings.child_chars,
                overlap_chars=self._settings.overlap_chars,
                parent_max_chars=self._settings.parent_max_chars,
            )
            count = self._embed_and_index_stream(content_hash, chunk_iter)
            if count == 0:
                self._store.set_stage(content_hash, "done", "empty", "切分后无内容")
                return IngestReport(doc.source_name, content_hash, version.version, "empty", 0,
                                    ["切分后无内容"])
            stage = "index"
            self._store.set_stage(content_hash, stage)
            self._store.finalize_chunk_count(content_hash)
            self._store.set_stage(content_hash, "done", "done")
            log.info("stream-ingested %s (v%s, %s chunks)", doc.source_name, version.version, count)
            return IngestReport(doc.source_name, content_hash, version.version, "done", count, [])
        except Exception as exc:  # noqa: BLE001
            self._store.set_stage(content_hash, stage, "failed", str(exc))
            log.warning("stream ingest failed at stage %s: %s", stage, exc)
            raise

    def _embed_and_index_stream(self, content_hash: str, chunk_iter) -> int:
        done = self._store.existing_chunk_ids(content_hash)
        batch = max(1, self._settings.embed_batch)
        buffer: list = []
        total = 0

        def flush() -> None:
            nonlocal total
            if not buffer:
                return
            vectors = self._embedder.embed([c.embed_text for c in buffer])
            self._store.upsert_chunk_batch(buffer, vectors)
            total += len(buffer)
            buffer.clear()

        for chunk in chunk_iter:
            if chunk.chunk_id in done:
                total += 1  # 已入库（续跑）
                continue
            buffer.append(chunk)
            if len(buffer) >= batch:
                flush()
        flush()
        return total
