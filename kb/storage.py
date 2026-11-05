"""存储工厂：按配置返回 SQLite（默认）或 PostgreSQL + pgvector 实现。

两者方法签名一致，上层（管线、检索、问答、任务）不感知差异。
pgvector 实现延迟导入，未选用时不要求安装 psycopg。
"""

from __future__ import annotations

from kb.config import Settings
from kb.store import Store


def build_store(settings: Settings):
    if settings.store == "postgres":
        from kb.pgstore import PgVectorStore  # 延迟导入，避免无谓依赖
        return PgVectorStore(settings)
    return Store(settings.db_path)
