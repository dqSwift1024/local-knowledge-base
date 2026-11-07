"""导入任务的后台线程。使用独立数据库连接，避免和请求线程抢同一条连接。"""

from __future__ import annotations

import threading
import time

from kb.config import Settings
from kb.jobs import Worker
from kb.pipeline import Pipeline
from kb.store import Store


def start_worker(settings: Settings, embedder) -> None:
    def loop() -> None:
        store = Store(settings.db_path)
        worker = Worker(store, Pipeline(store, embedder, settings), settings.max_job_attempts)
        while True:
            job = store.next_pending_job()
            if job is None:
                time.sleep(0.4)
                continue
            if job.attempts >= settings.max_job_attempts:
                store.update_job(job.id, status="dead", stage="ingest", error=job.error or "超过重试次数")
                continue
            worker.process_one(job)

    threading.Thread(target=loop, name="kb-worker", daemon=True).start()
