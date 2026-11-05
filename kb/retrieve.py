"""混合检索：向量召回 + 关键词召回，RRF 融合，输出候选与置信度。

子块用于命中，候选同时带上父块文本供回答使用。
置信度用于拒答：取候选中“查询 token 覆盖率”和向量余弦的较优者，
二者都低说明资料里没有相关内容。重排做成接口，日后可换 bge-reranker。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kb.config import Settings
from kb.embed import Embedder
from kb.store import Store
from kb.textutil import match_query, tokenize


@dataclass
class Retrieved:
    chunk_id: str
    doc_name: str
    section_path: str
    page: int | None
    text: str
    parent_text: str
    score: float
    vec_cos: float
    token_overlap: float
    methods: tuple[str, ...]


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]: ...


class FusionReranker:
    """默认重排：直接按融合分排序。接口保留，便于换成交叉编码器（bge-reranker）。"""

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)


# qwen3-embedding 的查询侧需要任务说明，文档侧保持原文。Ollama 模板不会自动加这段。
_QWEN3_QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _embed_query(embedder: Embedder, query: str) -> str:
    name = getattr(embedder, "name", "")
    if name.startswith("qwen3-embedding"):
        return f"Instruct: {_QWEN3_QUERY_TASK}\nQuery:{query}"
    return query


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, settings: Settings,
                 reranker: Reranker | None = None) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._reranker = reranker or FusionReranker()

    def search(self, query: str) -> list[Retrieved]:
        query_tokens = set(tokenize(query))
        vec_hits = self._vector_hits(query, query_tokens)
        key_hits = self._keyword_hits(query, query_tokens)
        fused = _rrf_merge(vec_hits, key_hits)
        # 先按融合分排序，再去冗余（近重复 + 单文档限流），最后截断到重排候选数
        fused.sort(key=lambda c: c.score, reverse=True)
        diversified = _diversify(
            fused,
            jaccard=self._settings.dedup_jaccard,
            per_doc_cap=self._settings.per_doc_cap,
        )
        candidates = diversified[: self._settings.rerank_candidates]
        ranked = self._reranker.rerank(query, candidates)
        return ranked[: self._settings.final_top_k]

    def confidence(self, candidates: list[Retrieved]) -> float:
        if not candidates:
            return 0.0
        return max(max(c.vec_cos, c.token_overlap) for c in candidates)

    def _vector_hits(self, query: str, query_tokens: set[str]) -> list[Retrieved]:
        query_vec = self._embedder.embed([_embed_query(self._embedder, query)])[0]
        rows = self._store.vector_search(query_vec, self._settings.vector_top_n)
        return [_to_retrieved(r, r["vec_cos"], query_tokens, "vector") for r in rows]

    def _keyword_hits(self, query: str, query_tokens: set[str]) -> list[Retrieved]:
        rows = self._store.keyword_search(match_query(query), self._settings.keyword_top_n)
        return [_to_retrieved(row, 0.0, query_tokens, "keyword") for row in rows]


def _to_retrieved(row, vec_cos: float, query_tokens: set[str], method: str) -> Retrieved:
    chunk_tokens = set(tokenize(row["text"]))
    overlap = len(query_tokens & chunk_tokens) / len(query_tokens) if query_tokens else 0.0
    return Retrieved(
        chunk_id=row["chunk_id"],
        doc_name=row["doc_name"],
        section_path=row["section_path"],
        page=row["page"],
        text=row["text"],
        parent_text=row["parent_text"] or row["text"],
        score=0.0,
        vec_cos=max(vec_cos, 0.0),
        token_overlap=overlap,
        methods=(method,),
    )


def _diversify(candidates: list[Retrieved], *, jaccard: float, per_doc_cap: int) -> list[Retrieved]:
    """去冗余：丢弃与已入选片段高度重复的候选，并限制单个文档的候选数量。

    主流 RAG 在重排前减少冗余，避免同一类内容（如反复刷屏的日志行）挤占
    有限的上下文窗口，导致"lost in the middle"。按已排好的融合分顺序贪心选取。
    """
    # 单文档库里没有“别的来源”需要保护，禁用单文档限流，避免把候选饿死
    distinct_docs = {c.doc_name for c in candidates}
    cap = per_doc_cap if (per_doc_cap > 0 and len(distinct_docs) > 1) else 0
    kept: list[Retrieved] = []
    kept_tokensets: list[set[str]] = []
    per_doc: dict[str, int] = {}
    for cand in candidates:
        if cap and per_doc.get(cand.doc_name, 0) >= cap:
            continue
        tokens = set(tokenize(cand.text))
        if any(_jaccard(tokens, seen) >= jaccard for seen in kept_tokensets):
            continue
        kept.append(cand)
        kept_tokensets.append(tokens)
        per_doc[cand.doc_name] = per_doc.get(cand.doc_name, 0) + 1
    return kept


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _rrf_merge(vec_hits: list[Retrieved], key_hits: list[Retrieved], k: int = 60) -> list[Retrieved]:
    """Reciprocal Rank Fusion：对两路排名取 1/(k+rank) 求和，避免分数量纲不一致。"""
    merged: dict[str, Retrieved] = {}
    for ranking in (vec_hits, key_hits):
        for rank, hit in enumerate(ranking):
            current = merged.get(hit.chunk_id)
            contribution = 1.0 / (k + rank)
            if current is None:
                hit.score = contribution
                merged[hit.chunk_id] = hit
            else:
                current.score += contribution
                current.vec_cos = max(current.vec_cos, hit.vec_cos)
                current.methods = tuple(dict.fromkeys(current.methods + hit.methods))
    return list(merged.values())
