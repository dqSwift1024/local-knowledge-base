"""评估集回归：命中率、引用有效性、拒答准确率。

用例字段：
- question：问题
- expected_pages：期望命中的页码列表（任一出现在候选页码中即算命中），可选
- expected_doc：期望命中的文档名，可选
- should_refuse：该问题是否应当拒答，可选（默认 False）
每次改动切分/嵌入/检索后重跑，分数下降即说明该次改动有害。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb.service import KnowledgeBase


@dataclass
class EvalReport:
    total: int
    recall_hits: int
    citation_valid: int
    citation_checked: int
    refusal_correct: int
    refusal_total: int
    cases: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float:
        base = self.total - self.refusal_total
        return self.recall_hits / base if base else 1.0

    @property
    def citation_validity(self) -> float:
        return self.citation_valid / self.citation_checked if self.citation_checked else 1.0

    @property
    def refusal_accuracy(self) -> float:
        return self.refusal_correct / self.refusal_total if self.refusal_total else 1.0


def run_eval(kb: "KnowledgeBase", cases: list[dict]) -> EvalReport:
    report = EvalReport(total=len(cases), recall_hits=0, citation_valid=0, citation_checked=0,
                        refusal_correct=0, refusal_total=0)
    for case in cases:
        question = case["question"]
        should_refuse = bool(case.get("should_refuse", False))
        candidates = kb.retriever.search(question)
        answer = kb.qa.ask(question)
        row = {"question": question, "refused": answer.refused}

        if should_refuse:
            report.refusal_total += 1
            correct = answer.refused
            report.refusal_correct += int(correct)
            row["refusal_correct"] = correct
        else:
            pages = {c.page for c in candidates}
            docs = {c.doc_name for c in candidates}
            hit = True
            if "expected_pages" in case:
                hit = hit and bool(set(case["expected_pages"]) & pages)
            if "expected_doc" in case:
                hit = hit and case["expected_doc"] in docs
            report.recall_hits += int(hit)
            row["recall_hit"] = hit

            # 引用必须全部来自本次检索候选
            allowed = {c.chunk_id for c in candidates}
            for citation in answer.citations:
                report.citation_checked += 1
                valid = citation.chunk_id in allowed
                report.citation_valid += int(valid)
            row["citations"] = len(answer.citations)

        report.cases.append(row)
    return report
