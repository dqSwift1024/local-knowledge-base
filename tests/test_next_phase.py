"""下一阶段能力的测试：流式大文件、可插拔重排/存储、FastAPI 界面。

走离线实现，无需外部服务。需要外部服务/模型的实现（pgvector、本地 bge）
只测“未装依赖时给出明确错误”的路由分支。
运行：python -m unittest discover -s tests
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from kb.config import Settings
from kb.service import KnowledgeBase


def _settings(tmp: Path, **over) -> Settings:
    return Settings(
        db_path=str(tmp / "kb.sqlite3"),
        backup_dir=str(tmp / "backups"),
        log_file=str(tmp / "kb.log"),
        **over,
    )


class StreamingIngestTest(unittest.TestCase):
    def test_large_text_uses_streaming_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            big = tmp / "日志.txt"
            # 构造超过阈值的纯文本（多段落）
            paragraphs = [f"第{i}段：系统在时间点{i}完成了一次检索与回答。" for i in range(400)]
            big.write_text("\n\n".join(paragraphs), encoding="utf-8")

            kb = KnowledgeBase(_settings(tmp, stream_threshold_bytes=200))
            try:
                report = kb.ingest(big)
                self.assertEqual(report.status, "done")
                self.assertGreater(report.chunk_count, 0)
                # 走的是流式解析器
                doc = kb.store.get_doc(report.content_hash)
                self.assertEqual(doc.parser, "text-stream")
                # 仍可检索问答
                answer = kb.ask("系统完成了什么？")
                self.assertFalse(answer.refused)
            finally:
                kb.close()

    def test_small_text_uses_normal_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            small = tmp / "小.txt"
            small.write_text("一句话说明。", encoding="utf-8")
            kb = KnowledgeBase(_settings(tmp, stream_threshold_bytes=10_000_000))
            try:
                report = kb.ingest(small)
                doc = kb.store.get_doc(report.content_hash)
                self.assertEqual(doc.parser, "text")  # 非流式
            finally:
                kb.close()


class PluggableRoutingTest(unittest.TestCase):
    def test_default_reranker_is_fusion(self) -> None:
        from kb.rerank import FusionReranker, build_reranker

        self.assertIsInstance(build_reranker(Settings()), FusionReranker)

    def test_cloud_reranker_requires_url(self) -> None:
        from kb.rerank import build_reranker

        with self.assertRaises(RuntimeError):
            build_reranker(Settings(reranker="cloud"))

    def test_bge_reranker_missing_dependency(self) -> None:
        from kb.rerank import build_reranker

        with self.assertRaises(RuntimeError):
            build_reranker(Settings(reranker="bge"))

    def test_postgres_store_missing_dependency(self) -> None:
        from kb.storage import build_store

        with self.assertRaises(RuntimeError):
            build_store(Settings(store="postgres", pg_dsn="postgresql://x/y"))


class Qwen3QueryPrefixTest(unittest.TestCase):
    def test_query_gets_instruct_document_path_does_not(self) -> None:
        import numpy as np

        from kb.retrieve import Retriever

        class _Store:
            def vector_search(self, query_vec, top_n):
                return []

            def keyword_search(self, match_expr, limit):
                return []

        class _Embedder:
            name = "qwen3-embedding:0.6b"
            dim = 4

            def embed(self, texts):
                self.texts = texts
                return np.zeros((len(texts), self.dim), dtype=np.float32)

        embedder = _Embedder()
        Retriever(_Store(), embedder, Settings()).search("怎么检索")
        self.assertEqual(
            embedder.texts,
            ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:怎么检索"],
        )


class DiversityTest(unittest.TestCase):
    def test_near_duplicates_and_per_doc_cap(self) -> None:
        from kb.retrieve import Retrieved, _diversify

        def make(cid, doc, text, score):
            return Retrieved(cid, doc, "", None, text, text, score, 0.0, 0.0, ("vector",))

        cands = [
            make("a1", "log.txt", "address 80:F1:F1:62:6E:A2 write permit progress 61", 0.9),
            make("a2", "log.txt", "address 80:F1:F1:62:6E:A2 write permit progress 61", 0.8),  # 近重复
            make("a3", "log.txt", "device R3AV DT on 54:77:87:4b:9d:5f hid bluetooth", 0.7),
            make("a4", "log.txt", "magenta remote connected state changed host", 0.6),
            make("b1", "guide.md", "蓝牙 设备 名称 与 地址 对应 关系 说明", 0.5),
        ]
        kept = _diversify(cands, jaccard=0.82, per_doc_cap=3)
        ids = [c.chunk_id for c in kept]
        self.assertNotIn("a2", ids)          # 近重复被丢弃
        self.assertLessEqual(sum(1 for c in kept if c.doc_name == "log.txt"), 3)  # 单文档限流
        self.assertIn("b1", ids)             # 其它来源仍保留，提升多样性


class LogChunkingTest(unittest.TestCase):
    def test_log_lines_kept_as_records(self) -> None:
        import tempfile, os
        from kb.parsers.stream import iter_blocks, _looks_like_log

        text = (
            "01-01 01:00:00.000 0 0 I boot start address=80:F1:F1:62:6E:A2\n"
            "01-01 01:00:01.000 0 0 I device R3AV DT on 54:77:87:4B:9D:5F\n"
            "    at com.example.Foo.bar(Foo.java:10)\n"  # 续行并入上一条记录
            "01-01 01:00:02.000 0 0 I done\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
            path = fh.name
        try:
            self.assertTrue(_looks_like_log(path))
            blocks = list(iter_blocks(path))
            joined = "\n".join(b.text for b in blocks)
            # 行边界收口：标识符整体保留，未被从中间截断
            self.assertIn("80:F1:F1:62:6E:A2", joined)
            self.assertIn("54:77:87:4B:9D:5F", joined)
            self.assertIn("Foo.java:10", joined)                   # 续行并入所属记录
            # 每个 block 内部按行组织，续行跟随其时间戳行
            for b in blocks:
                for ln in b.text.split("\n"):
                    self.assertTrue(ln.strip())                    # 无空行碎片
        finally:
            os.unlink(path)


class EmbedTextTest(unittest.TestCase):
    def test_heading_path_prepended_only_when_meaningful(self) -> None:
        from kb.model import Chunk

        with_heading = Chunk("d", "h", "安装 / 环境要求", "需 8GB 内存", None)
        self.assertTrue(with_heading.embed_text.startswith("安装 / 环境要求\n"))
        self.assertEqual(with_heading.text, "需 8GB 内存")  # 展示文本不变
        plain = Chunk("d", "h", "(正文)", "一行日志", None)
        self.assertEqual(plain.embed_text, "一行日志")       # 无标题不拼接


class CitationCoerceTest(unittest.TestCase):
    def test_bracketed_citations_normalized(self) -> None:
        from kb.gateway import _coerce
        import json
        out = _coerce(json.dumps({"answer": "x", "citations": ["[abc123]", " def456 ", ""]}))
        self.assertEqual(out["citations"], ["abc123", "def456"])


class WebInterfaceTest(unittest.TestCase):
    def test_ingest_ask_health_via_testclient(self) -> None:
        from fastapi.testclient import TestClient

        from kb.web import create_app

        with tempfile.TemporaryDirectory() as td:
            app = create_app(_settings(Path(td)))
            client = TestClient(app)

            files = {"file": ("说明.md", "# 标题\n星云知识库支持检索与引用。\n".encode(), "text/markdown")}
            r = client.post("/ingest", files=files)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "queued")
            job_id = r.json()["job_id"]
            job = {}
            for _ in range(50):
                st = client.get("/status").json()
                self.assertIn("busy", st)
                self.assertIn("runtime", st)
                job = next(item for item in st["jobs"] if item["id"] == job_id)
                if job["status"] in ("done", "dead", "failed"):
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "done")

            r = client.get("/health")
            self.assertTrue(r.json()["ok"])

            r = client.post("/ask", data={"question": "星云知识库支持什么？"})
            body = r.json()
            self.assertFalse(body["refused"])
            self.assertTrue(body["citations"])
            self.assertTrue(body["trace"])
            self.assertEqual(body["trace"][-1]["status"], "done")

            r = client.post("/ask", data={"question": "火星天气如何？"})
            self.assertTrue(r.json()["refused"])

            status = client.get("/status").json()
            doc = status["docs"][0]
            self.assertIn("content_hash", doc)
            r = client.delete(f"/documents/{doc['content_hash']}")
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()["ok"])
            self.assertEqual(client.get("/health").json()["docs"], 0)
            self.assertEqual(
                client.delete(f"/documents/{doc['content_hash']}").status_code,
                404,
            )


if __name__ == "__main__":
    unittest.main()
