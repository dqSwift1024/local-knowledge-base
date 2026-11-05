"""后台任务：导入放进 SQLite 任务队列，由 Worker 逐个处理。

单人本地不引入 Redis；任务表在 SQLite，Worker 可按需 run_pending 或常驻 run_forever。
失败会计数重试，超过上限标记 dead。因 chunk_id 确定性，重跑会跳过已完成子块（断点续跑）。
服务重启后，队列里 pending/failed 的任务仍可被重新拾起。
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from kb.pipeline import IngestReport, Pipeline
from kb.store import Job, Store


log = logging.getLogger("kb.jobs")


class Worker:
    def __init__(self, store: Store, pipeline: Pipeline, max_attempts: int = 3) -> None:
        self._store = store
        self._pipeline = pipeline
        self._max_attempts = max_attempts

    def process_one(self, job: Job) -> IngestReport | None:
        self._store.update_job(job.id, status="running", stage="ingest", bump_attempt=True)
        try:
            report = self._pipeline.ingest(job.path)
            self._store.update_job(job.id, status="done", stage=report.status,
                                   content_hash=report.content_hash)
            return report
        except Exception as exc:  # noqa: BLE001 — 任务级失败不应掀翻 Worker
            dead = job.attempts + 1 >= self._max_attempts
            self._store.update_job(job.id, status="dead" if dead else "failed",
                                   stage="ingest", error=str(exc))
            log.warning("job %s %s: %s", job.id, "dead" if dead else "failed", exc)
            return None

    def run_pending(self) -> int:
        """处理完当前所有待办任务，返回处理次数。"""
        processed = 0
        while True:
            job = self._store.next_pending_job()
            if job is None:
                break
            if job.attempts >= self._max_attempts:
                self._store.update_job(job.id, status="dead", stage="ingest", error=job.error)
                continue
            self.process_one(job)
            processed += 1
        return processed

    def run_forever(self, interval: float = 2.0, stop: Callable[[], bool] | None = None) -> None:
        log.info("worker started")
        while stop is None or not stop():
            if self.run_pending() == 0:
                time.sleep(interval)
