"""第一版知识库的端到端测试，全部走离线实现，无需外部服务。

运行：python -m unittest discover -s tests
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kb.config import Settings
from kb.service import KnowledgeBase


SAMPLE = (
    "# 产品说明\n星云知识库支持文档检索和引用溯源。\n\n"
    "## 使用限制\n单文件上限为一GB，超出需要拆分处理。\n\n"
    "## 计费说明\n基础版本按月付费，支持随时取消。\n"
)


def _settings(tmp: Path, **over) -> Settings:
    return Settings(
        db_path=str(tmp / "kb.sqlite3"),
        backup_dir=str(tmp / "backups"),
        log_file=str(tmp / "kb.log"),
        **over,
    )


class KnowledgeBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.sample = self.tmp / "说明.md"
        self.sample.write_text(SAMPLE, encoding="utf-8")
        self.kb = KnowledgeBase(_settings(self.tmp))

    def tearDown(self) -> None:
        self.kb.close()
        self._tmpdir.cleanup()

    def test_ingest_and_cited_answer(self) -> None:
        report = self.kb.ingest(self.sample)
        self.assertEqual(report.status, "done")
        self.assertGreater(report.chunk_count, 0)

        answer = self.kb.ask("星云知识库支持什么？")
        self.assertFalse(answer.refused)
        self.assertTrue(answer.citations)
        # 引用必须来自真实入库的块
        known = self.kb.store.existing_chunk_ids(report.content_hash)
        for citation in answer.citations:
            self.assertIn(citation.chunk_id, known)

    def test_refusal_for_unknown(self) -> None:
        self.kb.ingest(self.sample)
        answer = self.kb.ask("火星的天气怎么样？")
        self.assertTrue(answer.refused)
        self.assertEqual(answer.citations, [])

    def test_dedupe_same_file(self) -> None:
        first = self.kb.ingest(self.sample)
        second = self.kb.ingest(self.sample)
        self.assertFalse(first.skipped)
        self.assertTrue(second.skipped)

    def test_parent_text_present(self) -> None:
        self.kb.ingest(self.sample)
        hits = self.kb.retriever.search("使用限制")
        self.assertTrue(hits)
        self.assertTrue(any(h.parent_text for h in hits))

    def test_versioning_and_rollback(self) -> None:
        self.kb.ingest(self.sample)
        self.sample.write_text(SAMPLE + "\n## 新增\n支持导出为 PDF。\n", encoding="utf-8")
        self.kb.ingest(self.sample)

        versions = self.kb.versions("说明.md")
        self.assertEqual(len(versions), 2)
        self.assertTrue(versions[1].is_latest)

        self.assertTrue(self.kb.rollback("说明.md", 1))
        versions = self.kb.versions("说明.md")
        self.assertTrue(versions[0].is_latest)
        self.assertFalse(versions[1].is_latest)

    def test_resume_after_interrupted_embedding(self) -> None:
        """模拟嵌入中途失败：第二次导入应跳过已完成子块并最终完成。"""
        from kb import pipeline as pipeline_module

        report_holder = {}
        original = self.kb.embedder.embed
        calls = {"n": 0}

        def flaky(texts):
            calls["n"] += 1
            if calls["n"] == 1:
                # 先成功嵌入一批并落库，再在下一批抛错
                return original(texts)
            raise RuntimeError("模拟嵌入失败")

        # 用极小批次，确保第一批落库后仍有剩余
        kb2 = KnowledgeBase(_settings(self.tmp / "r", embed_batch=1))
        kb2.embedder.embed = flaky  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            kb2.ingest(self.sample)
        partial = kb2.store.existing_chunk_ids(_hash(self.sample))
        self.assertGreaterEqual(len(partial), 1)

        # 恢复正常嵌入后重跑，应补齐并完成
        kb2.embedder.embed = original  # type: ignore[assignment]
        # 重新装配 pipeline 以使用恢复后的 embedder
        kb2.pipeline = pipeline_module.Pipeline(kb2.store, kb2.embedder, kb2.settings)
        report = kb2.ingest(self.sample)
        self.assertEqual(report.status, "done")
        report_holder["ok"] = True
        kb2.close()
        self.assertTrue(report_holder["ok"])

    def test_job_queue_worker(self) -> None:
        job_id = self.kb.enqueue(self.sample)
        self.assertIsInstance(job_id, int)
        processed = self.kb.run_worker_once()
        self.assertEqual(processed, 1)
        jobs = self.kb.store.list_jobs()
        self.assertEqual(jobs[0].status, "done")

    def test_eval_report(self) -> None:
        self.kb.ingest(self.sample)
        cases = [
            {"question": "单文件上限是多少？", "expected_doc": "说明.md"},
            {"question": "火星的天气怎么样？", "should_refuse": True},
        ]
        report = self.kb.evaluate(cases)
        self.assertEqual(report.total, 2)
        self.assertEqual(report.refusal_accuracy, 1.0)
        self.assertEqual(report.citation_validity, 1.0)

    def test_feedback_and_health_and_backup(self) -> None:
        self.kb.ingest(self.sample)
        answer = self.kb.ask("计费是怎样的？")
        self.kb.record_feedback(answer, useful=True)
        self.assertEqual(self.kb.store.feedback_stats()["total"], 1)

        health = self.kb.health()
        self.assertTrue(health["ok"])
        self.assertGreaterEqual(health["chunks"], 1)

        dest = self.kb.backup(str(self.tmp / "b.sqlite3"))
        self.assertTrue(Path(dest).exists())


def _hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
