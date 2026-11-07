"""嵌入层。离线哈希实现零依赖可跑；云端实现走 OpenAI 兼容接口（带重试）。

两者产出同维度的 L2 归一化向量，检索代码只认接口，不关心来源。
换模型意味着整库重嵌（不同模型向量不可混用），由上层负责。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from kb.config import Settings
from kb.http import post_json
from kb.textutil import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """把 token 哈希散列到固定维度。确定性、离线、可复现，用于先跑通闭环。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                bucket = hash(("tok", token)) % self.dim
                sign = 1.0 if hash(("sgn", token)) % 2 else -1.0
                vectors[row, bucket] += sign
        return _l2_normalize(vectors)


class EndpointEmbedder:
    """一个 OpenAI 兼容嵌入端点。本地和云端只是地址不同。"""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, dim: int, post=None) -> None:
        from kb.http import post_json

        self.name = name
        self.dim = dim
        self._url = base_url.rstrip("/") + "/embeddings"
        self._key = api_key
        self._model = model
        self._post = post or post_json

    def embed(self, texts: list[str]) -> np.ndarray:
        payload = self._post(
            self._url, self._key,
            {"model": self._model, "input": texts, "keep_alive": "30m"},
        )
        vectors = np.array([item["embedding"] for item in payload["data"]], dtype=np.float32)
        if vectors.shape[1] != self.dim:
            self.dim = int(vectors.shape[1])
        return _l2_normalize(vectors)


class OpenAICompatEmbedder:
    """调用 /embeddings 的兼容接口。密钥只从环境变量读。"""

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise RuntimeError("使用 openai 嵌入需要设置 OPENAI_API_KEY")
        self._url = settings.base_url.rstrip("/") + "/embeddings"
        self._key = settings.api_key
        self._model = settings.embed_model
        self.name = settings.embed_model
        self.dim = settings.embed_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        payload = post_json(
            self._url, self._key,
            {"model": self._model, "input": texts, "keep_alive": "30m"},
        )
        vectors = np.array([item["embedding"] for item in payload["data"]], dtype=np.float32)
        self.dim = vectors.shape[1]
        return _l2_normalize(vectors)


class LocalBgeEmbedder:
    """本地 bge-m3 等句向量模型（需 sentence-transformers + 模型权重）。

    此环境未安装 torch/模型，仅在装好依赖的机器上可用；接口与其它嵌入一致。
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "本地 bge 嵌入需要 sentence-transformers：pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(settings.embed_local_model)
        self.name = settings.embed_local_model
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)


def build_embedder(settings: Settings) -> Embedder:
    from kb.failover import build_embed_chain

    chained = build_embed_chain(settings)
    if chained is not None:
        return chained
    if settings.embedder == "openai":
        return OpenAICompatEmbedder(settings)
    if settings.embedder == "bge":
        return LocalBgeEmbedder(settings)
    return HashingEmbedder(settings.embed_dim)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
