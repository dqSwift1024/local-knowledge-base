"""集中配置。全部有默认值，可用环境变量覆盖，便于后续切云端/切 Postgres。

字段在创建 Settings 时读取环境变量。项目根目录的 .env 由 load_settings 载入，
且不覆盖已经存在的环境变量。测试直接构造 Settings，不会读取 .env。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(name: str, default: str):
    def read() -> str:
        return os.environ.get(name, default)
    return read


def _env_int(name: str, default: str):
    def read() -> int:
        return int(os.environ.get(name, default))
    return read


def _env_float(name: str, default: str):
    def read() -> float:
        return float(os.environ.get(name, default))
    return read


@dataclass
class Settings:
    db_path: str = field(default_factory=_env_str("KB_DB", "kb_data/kb.sqlite3"))

    # 切分
    child_chars: int = field(default_factory=_env_int("KB_CHILD_CHARS", "450"))
    overlap_chars: int = field(default_factory=_env_int("KB_OVERLAP_CHARS", "60"))
    parent_max_chars: int = field(default_factory=_env_int("KB_PARENT_MAX_CHARS", "2000"))
    embed_batch: int = field(default_factory=_env_int("KB_EMBED_BATCH", "64"))

    # 嵌入：offline（离线哈希，零依赖）或 openai（兼容接口）
    embedder: str = field(default_factory=_env_str("KB_EMBEDDER", "offline"))
    embed_dim: int = field(default_factory=_env_int("KB_EMBED_DIM", "256"))
    embed_model: str = field(default_factory=_env_str("KB_EMBED_MODEL", "text-embedding-3-small"))

    # 回答模型：offline（离线抽取式）或 openai（兼容接口）
    answerer: str = field(default_factory=_env_str("KB_ANSWERER", "offline"))
    chat_model: str = field(default_factory=_env_str("KB_CHAT_MODEL", "gpt-4o-mini"))
    # 回答容灾链，逗号分隔。每一项再用 KB_CHAT_<NAME>_BASE_URL / _API_KEY / _MODEL。
    # 本地 Ollama 与云端都用同一套对话补全请求。空则沿用上面的单个模型。
    chat_chain: str = field(default_factory=_env_str("KB_CHAT_CHAIN", ""))
    # 嵌入容灾链。备用只有维度与 KB_EMBED_DIM 一致才会被调用。
    embed_chain: str = field(default_factory=_env_str("KB_EMBED_CHAIN", ""))
    history_messages: int = field(default_factory=_env_int("KB_HISTORY_MESSAGES", "6"))
    failover_threshold: int = field(default_factory=_env_int("KB_FAILOVER_THRESHOLD", "2"))
    failover_cooldown_s: float = field(default_factory=_env_float("KB_FAILOVER_COOLDOWN_S", "60"))

    # 云端接口（embedder/answerer 为 openai 时使用）
    base_url: str = field(default_factory=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=_env_str("OPENAI_API_KEY", ""))

    # 检索
    vector_top_n: int = field(default_factory=_env_int("KB_VECTOR_TOP_N", "30"))
    keyword_top_n: int = field(default_factory=_env_int("KB_KEYWORD_TOP_N", "30"))
    final_top_k: int = field(default_factory=_env_int("KB_FINAL_TOP_K", "5"))
    # 置信度低于此值则拒答
    refuse_below: float = field(default_factory=_env_float("KB_REFUSE_BELOW", "0.12"))

    # 运维
    backup_dir: str = field(default_factory=_env_str("KB_BACKUP_DIR", "kb_data/backups"))
    log_level: str = field(default_factory=_env_str("KB_LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=_env_str("KB_LOG_FILE", "kb_data/kb.log"))
    max_job_attempts: int = field(default_factory=_env_int("KB_MAX_JOB_ATTEMPTS", "3"))

    # 规模化 / 下一阶段
    store: str = field(default_factory=_env_str("KB_STORE", "sqlite"))  # sqlite | postgres
    pg_dsn: str = field(default_factory=_env_str("KB_PG_DSN", ""))
    hnsw_m: int = field(default_factory=_env_int("KB_HNSW_M", "16"))
    hnsw_ef_construction: int = field(default_factory=_env_int("KB_HNSW_EF_CONSTRUCTION", "100"))
    hnsw_ef_search: int = field(default_factory=_env_int("KB_HNSW_EF_SEARCH", "80"))
    # 超过该字节数的纯文本走流式导入，内存不随文件线性增长
    stream_threshold_bytes: int = field(default_factory=_env_int("KB_STREAM_THRESHOLD", "5000000"))
    # 重排：none（默认，按融合分）| cloud（HTTP rerank 接口）| bge（本地交叉编码器）
    reranker: str = field(default_factory=_env_str("KB_RERANKER", "none"))
    rerank_url: str = field(default_factory=_env_str("KB_RERANK_URL", ""))
    rerank_model: str = field(default_factory=_env_str("KB_RERANK_MODEL", "bge-reranker-v2-m3"))
    rerank_candidates: int = field(default_factory=_env_int("KB_RERANK_CANDIDATES", "20"))
    # 召回去冗余：近重复判定阈值（token Jaccard）与单文档候选上限，抑制同一类内容刷屏
    dedup_jaccard: float = field(default_factory=_env_float("KB_DEDUP_JACCARD", "0.82"))
    per_doc_cap: int = field(default_factory=_env_int("KB_PER_DOC_CAP", "3"))
    # 本地嵌入模型（embedder=bge 时使用）
    embed_local_model: str = field(default_factory=_env_str("KB_EMBED_LOCAL_MODEL", "BAAI/bge-m3"))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_settings() -> Settings:
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return Settings()
