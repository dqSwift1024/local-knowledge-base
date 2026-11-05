"""重排器：默认按融合分；可切换云端 rerank 接口或本地 bge 交叉编码器。

Retriever 先用 RRF 召回候选，再交给重排器精排。换重排器不影响检索与问答。
"""

from __future__ import annotations

import logging

from kb.config import Settings
from kb.http import post_json
from kb.retrieve import FusionReranker, Reranker, Retrieved


log = logging.getLogger("kb.rerank")


class CloudReranker:
    """调用 HTTP rerank 接口（Jina / bge-reranker 服务 / Cohere 风格）。"""

    def __init__(self, settings: Settings) -> None:
        if not settings.rerank_url:
            raise RuntimeError("cloud 重排需要设置 KB_RERANK_URL")
        self._url = settings.rerank_url
        self._key = settings.api_key
        self._model = settings.rerank_model

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return candidates
        documents = [c.parent_text or c.text for c in candidates]
        body = post_json(self._url, self._key,
                         {"model": self._model, "query": query, "documents": documents})
        results = body.get("results") or body.get("data") or []
        ordered: list[Retrieved] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                candidates[idx].score = float(score)
                ordered.append(candidates[idx])
        return ordered or sorted(candidates, key=lambda c: c.score, reverse=True)


class LocalBgeReranker:
    """本地 bge-reranker 交叉编码器（需 FlagEmbedding + 模型权重）。

    此环境未安装，仅在装好依赖的机器上可用；接口与其它重排器一致。
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError("本地 bge 重排需要 FlagEmbedding：pip install FlagEmbedding") from exc
        self._model = FlagReranker(settings.rerank_model, use_fp16=True)

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return candidates
        pairs = [[query, c.parent_text or c.text] for c in candidates]
        scores = self._model.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cloud":
        return CloudReranker(settings)
    if settings.reranker == "bge":
        return LocalBgeReranker(settings)
    return FusionReranker()
