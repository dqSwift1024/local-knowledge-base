"""问答编排：检索 → 置信度判断 → 生成 → 校验引用 → 组织带出处的回答或拒答。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kb.config import Settings
from kb.gateway import AnswerGenerator
from kb.retrieve import Retrieved, Retriever


REFUSAL = "当前知识库中没有足够依据回答该问题。"


@dataclass
class Citation:
    chunk_id: str
    doc_name: str
    section_path: str
    page: int | None
    text: str


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    nearest: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[dict] = field(default_factory=list)
    session_id: str = ""
    model: str = ""
    degraded: bool = False


class QAService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator, settings: Settings) -> None:
        self._retriever = retriever
        self._generator = generator
        self._settings = settings

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        trace: list[dict] = []
        started = time.perf_counter()
        candidates = self._retriever.search(question)
        trace.append({
            "step": "混合检索", "status": "done",
            "detail": f"召回 {len(candidates)} 条",
            "ms": _ms(started),
        })

        judged = time.perf_counter()
        confidence = self._retriever.confidence(candidates)
        refused = (not candidates) or confidence < self._settings.refuse_below
        trace.append({
            "step": "置信度判断",
            "status": "refused" if refused else "done",
            "detail": f"{confidence:.3f}",
            "ms": _ms(judged),
        })
        if refused:
            return Answer(
                question=question, text=REFUSAL, refused=True,
                nearest=[_to_citation(c) for c in candidates[:3]],
                confidence=confidence, trace=trace,
            )

        generated = time.perf_counter()
        result = self._generator.generate(question, candidates, history)
        allowed = {c.chunk_id: c for c in candidates}
        valid_ids = [cid for cid in result.get("citations", []) if cid in allowed]
        rejected = bool(result.get("refused") or not valid_ids)
        model = str(result.get("model") or "offline-extractive")
        degraded = bool(result.get("degraded", False))
        trace.append({
            "step": "组织回答",
            "status": "refused" if rejected else "done",
            "detail": "引用未通过校验" if rejected else f"{len(valid_ids)} 条引用",
            "ms": _ms(generated),
        })
        if rejected:
            return Answer(
                question=question, text=REFUSAL, refused=True,
                nearest=[_to_citation(c) for c in candidates[:3]],
                confidence=confidence, trace=trace,
                model=model, degraded=degraded,
            )

        return Answer(
            question=question,
            text=result.get("answer", "").strip() or REFUSAL,
            refused=False,
            citations=[_to_citation(allowed[cid]) for cid in valid_ids],
            confidence=confidence, trace=trace,
            model=model, degraded=degraded,
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _to_citation(hit: Retrieved) -> Citation:
    return Citation(
        chunk_id=hit.chunk_id,
        doc_name=hit.doc_name,
        section_path=hit.section_path,
        page=hit.page,
        text=hit.text,
    )
