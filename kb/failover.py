"""回答与嵌入的容灾链。

本地 Ollama 和云端都是链上的一个端点，请求格式相同。
某个回答模型连续失败达到阈值后，冷却期内跳过；全部失败则退回离线摘录。
嵌入备用端点只有在维度与当前库一致时才会被调用。
"""

from __future__ import annotations

import os
import time

import numpy as np

from kb.config import Settings
from kb.embed import EndpointEmbedder
from kb.gateway import CloudAnswerGenerator, OfflineExtractive
from kb.retrieve import Retrieved


class FailoverGenerator:
    def __init__(
        self,
        models: list,
        fallback: OfflineExtractive | None = None,
        *,
        threshold: int = 2,
        cooldown: float = 60,
        clock=None,
    ) -> None:
        self.models = list(models)
        self.fallback = fallback or OfflineExtractive()
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock or time.monotonic
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def generate(self, question: str, contexts: list[Retrieved], history: list[dict] | None = None) -> dict:
        for model in self.models:
            if self._skip(model.name):
                continue
            try:
                result = dict(model.generate(question, contexts, history))
            except Exception:
                self._mark_failure(model.name)
                continue
            self._mark_success(model.name)
            result["model"] = model.name
            result["degraded"] = False
            return result
        result = dict(self.fallback.generate(question, contexts, history))
        result["model"] = "offline-extractive"
        result["degraded"] = True
        return result

    def _skip(self, name: str) -> bool:
        return self.clock() < self._open_until.get(name, 0.0)

    def _mark_failure(self, name: str) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count >= self.threshold:
            self._open_until[name] = self.clock() + self.cooldown

    def _mark_success(self, name: str) -> None:
        self._failures[name] = 0
        self._open_until[name] = 0.0


class FailoverEmbedder:
    """按顺序嵌入。维度与 expected_dim 不同的端点不会被调用。"""

    def __init__(self, embedders: list, expected_dim: int) -> None:
        self._embedders = list(embedders)
        self.dim = expected_dim
        names = [getattr(item, "name", "?") for item in self._embedders]
        self.name = "failover:" + ",".join(names)

    def embed(self, texts: list[str]) -> np.ndarray:
        last: Exception | None = None
        for emb in self._embedders:
            if getattr(emb, "dim", None) != self.dim:
                continue
            try:
                vectors = emb.embed(texts)
            except Exception as exc:
                last = exc
                continue
            if getattr(vectors, "shape", (0, 0))[1] != self.dim:
                last = RuntimeError("嵌入维度与当前库不一致")
                continue
            return vectors
        if last is not None:
            raise last
        raise RuntimeError("没有维度匹配的嵌入模型")


def build_chat_models(settings: Settings) -> list[CloudAnswerGenerator]:
    chain = [part.strip() for part in settings.chat_chain.split(",") if part.strip()]
    if chain:
        models: list[CloudAnswerGenerator] = []
        for name in chain:
            key = name.upper()
            models.append(CloudAnswerGenerator(
                settings,
                name=name,
                base_url=os.environ[f"KB_CHAT_{key}_BASE_URL"],
                api_key=os.environ.get(f"KB_CHAT_{key}_API_KEY", ""),
                model=os.environ.get(f"KB_CHAT_{key}_MODEL", name),
            ))
        return models
    if settings.answerer == "openai":
        return [CloudAnswerGenerator(settings, name=settings.chat_model)]
    return []


def build_embed_chain(settings: Settings) -> FailoverEmbedder | None:
    chain = [part.strip() for part in settings.embed_chain.split(",") if part.strip()]
    if not chain:
        return None
    embedders = []
    for name in chain:
        key = name.upper()
        embedders.append(EndpointEmbedder(
            name,
            os.environ[f"KB_EMBED_{key}_BASE_URL"],
            os.environ.get(f"KB_EMBED_{key}_API_KEY", ""),
            os.environ.get(f"KB_EMBED_{key}_MODEL", name),
            int(os.environ.get(f"KB_EMBED_{key}_DIM", str(settings.embed_dim))),
        ))
    return FailoverEmbedder(embedders, settings.embed_dim)
