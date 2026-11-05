"""KnowledgeBase 门面：把存储、嵌入、管线、检索、问答、任务串成一个对象。

CLI 和测试都通过它使用系统，避免各处重复装配。
"""

from __future__ import annotations

import time
from pathlib import Path

from kb.config import Settings
from kb.embed import build_embedder
from kb.eval import EvalReport, run_eval
from kb.gateway import build_generator
from kb.jobs import Worker
from kb.parsers import supported_extensions
from kb.pipeline import IngestReport, Pipeline
from kb.qa import Answer, QAService
from kb.rerank import build_reranker
from kb.retrieve import Retriever
from kb.storage import build_store


class KnowledgeBase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = build_store(settings)
        self.embedder = build_embedder(settings)
        self.pipeline = Pipeline(self.store, self.embedder, settings)
        self.generator = build_generator(settings)
        self.retriever = Retriever(self.store, self.embedder, settings, reranker=build_reranker(settings))
        self.qa = QAService(self.retriever, self.generator, settings)
        self.worker = Worker(self.store, self.pipeline, settings.max_job_attempts)

    def close(self) -> None:
        self.store.close()

    # ---- 导入 ----

    def ingest(self, path: str | Path) -> IngestReport:
        return self.pipeline.ingest(path)

    def ingest_dir(self, directory: str | Path, recursive: bool = True) -> list[IngestReport]:
        root = Path(directory)
        pattern = "**/*" if recursive else "*"
        exts = set(supported_extensions())
        reports: list[IngestReport] = []
        for file_path in sorted(root.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in exts:
                reports.append(self.pipeline.ingest(file_path))
        return reports

    def enqueue(self, path: str | Path) -> int:
        return self.store.enqueue_job(str(Path(path)))

    def run_worker_once(self) -> int:
        return self.worker.run_pending()

    # ---- 问答与反馈 ----

    def ask(self, question: str) -> Answer:
        return self.qa.ask(question)

    def record_feedback(self, answer: Answer, useful: bool) -> None:
        self.store.add_feedback(answer.question, useful, answer.refused,
                                [c.chunk_id for c in answer.citations])

    # ---- 版本 ----

    def versions(self, doc_name: str):
        return self.store.versions_of(doc_name)

    def rollback(self, doc_name: str, version: int) -> bool:
        return self.store.set_latest(doc_name, version)

    def delete_version(self, content_hash: str) -> bool:
        return self.store.delete_version(content_hash)

    def purge_superseded(self, doc_name: str) -> int:
        return self.store.purge_superseded(doc_name)

    # ---- 运维 ----

    def health(self) -> dict:
        return self.store.health()

    def backup(self, dest: str | None = None) -> str:
        dest = dest or str(Path(self.settings.backup_dir) / f"kb-{int(time.time())}.sqlite3")
        return self.store.backup(dest)

    def evaluate(self, cases: list[dict]) -> EvalReport:
        return run_eval(self, cases)
