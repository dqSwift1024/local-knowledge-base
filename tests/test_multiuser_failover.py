"""冻结用例：多轮对话、多用户隔离、本地/云端模型容灾。

用例编号对应计划书。断言在编码开始后不再改写。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kb.config import Settings
from kb.failover import FailoverEmbedder, FailoverGenerator
from kb.gateway import CloudAnswerGenerator
from kb.qa import REFUSAL
from kb.service import KnowledgeBase, SessionError


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


class RecordingChat:
    def __init__(self, name: str = "recorder") -> None:
        self.name = name
        self.calls: list[dict] = []

    def generate(self, question, contexts, history=None):
        self.calls.append({"question": question, "history": list(history or [])})
        cid = contexts[0].chunk_id if contexts else ""
        return {
            "answer": f"答:{question}",
            "citations": [cid] if cid else [],
            "refused": not bool(cid),
        }


class BoomChat:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.fail = True

    def generate(self, question, contexts, history=None):
        self.calls += 1
        if self.fail:
            raise TimeoutError("timed out")
        cid = contexts[0].chunk_id
        return {"answer": f"{self.name}回答", "citations": [cid], "refused": False}


class OkChat:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        self.calls = 0

    def generate(self, question, contexts, history=None):
        self.calls += 1
        cid = contexts[0].chunk_id
        return {"answer": self.text, "citations": [cid], "refused": False}


class FakeEmb:
    def __init__(self, name: str, dim: int, fail: bool = False) -> None:
        self.name = name
        self.dim = dim
        self.fail = fail
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise TimeoutError("down")
        vec = np.ones((len(texts), self.dim), dtype=np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return vec


class FrozenPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.sample = self.tmp / "说明.md"
        self.sample.write_text(SAMPLE, encoding="utf-8")
        self.kb = KnowledgeBase(_settings(self.tmp))
        self.kb.ingest(self.sample)
        self.recorder = RecordingChat()
        self.kb.qa._generator = self.recorder

    def tearDown(self) -> None:
        self.kb.close()
        self._tmpdir.cleanup()

    def _sid(self, user: str) -> str:
        return self.kb.open_session(user)

    def test_mt_01_second_turn_sees_first(self) -> None:
        sid = self._sid("alice")
        first = "星云知识库支持什么？"
        self.kb.ask(first, "alice", sid)
        self.kb.ask("星云知识库计费说明是什么？", "alice", sid)
        history = self.recorder.calls[1]["history"]
        blob = " ".join(item["content"] for item in history)
        self.assertIn(first, blob)
        self.assertIn(f"答:{first}", blob)

    def test_mt_02_only_last_six_messages_sent(self) -> None:
        sid = self._sid("alice")
        questions = [
            "轮次甲 星云知识库支持什么",
            "轮次乙 星云知识库引用溯源",
            "轮次丙 星云知识库单文件上限",
            "轮次丁 星云知识库计费说明",
            "轮次戊 星云知识库能否取消",
        ]
        for question in questions:
            self.kb.ask(question, "alice", sid)
        history = self.recorder.calls[-1]["history"]
        blob = " ".join(item["content"] for item in history)
        self.assertEqual(len(history), 6)
        self.assertNotIn("轮次甲", blob)
        self.assertIn("轮次丁", blob)
        stored = " ".join(item["content"] for item in self.kb.get_session("alice", sid)["turns"])
        self.assertIn("轮次甲", stored)

    def test_mt_03_stateless_ask_ignores_sessions(self) -> None:
        sid = self._sid("alice")
        self.kb.ask("星云知识库支持什么？", "alice", sid)
        before = self.kb.store.count_turns(sid)
        sessions = self.kb.store.count_sessions()
        self.recorder.calls.clear()
        self.kb.ask("星云知识库计费说明")
        self.assertEqual(self.recorder.calls[-1]["history"], [])
        self.assertEqual(self.kb.store.count_turns(sid), before)
        self.assertEqual(self.kb.store.count_sessions(), sessions)

    def test_mt_04_refusal_is_visible_next_turn(self) -> None:
        sid = self._sid("alice")
        first = self.kb.ask("火星的天气怎么样？", "alice", sid)
        self.assertTrue(first.refused)
        self.kb.ask("星云知识库支持什么？", "alice", sid)
        blob = " ".join(item["content"] for item in self.recorder.calls[-1]["history"])
        self.assertIn(REFUSAL, blob)

    def test_mu_01_sessions_are_private(self) -> None:
        alice = self._sid("alice")
        bob = self._sid("bob")
        self.assertEqual(self.kb.list_sessions("alice"), [alice])
        self.assertEqual(self.kb.list_sessions("bob"), [bob])
        self.assertNotIn(alice, self.kb.list_sessions("bob"))

    def test_mu_02_other_user_cannot_continue(self) -> None:
        sid = self._sid("alice")
        self.kb.ask("星云知识库支持什么？", "alice", sid)
        before = self.kb.store.count_turns(sid)
        with self.assertRaises(SessionError):
            self.kb.ask("星云知识库计费说明", "bob", sid)
        self.assertEqual(self.kb.store.count_turns(sid), before)

    def test_mu_03_other_user_cannot_read(self) -> None:
        sid = self._sid("alice")
        self.assertIsNone(self.kb.get_session("bob", sid))

    def test_mu_04_two_sessions_do_not_mix(self) -> None:
        one = self._sid("alice")
        two = self._sid("alice")
        self.kb.ask("问题甲 星云知识库支持什么", "alice", one)
        self.kb.ask("问题乙 星云知识库计费说明", "alice", two)
        first = " ".join(item["content"] for item in self.kb.get_session("alice", one)["turns"])
        second = " ".join(item["content"] for item in self.kb.get_session("alice", two)["turns"])
        self.assertIn("问题甲", first)
        self.assertNotIn("问题甲", second)
        self.assertIn("问题乙", second)
        self.assertNotIn("问题乙", first)

    def test_dr_01_backup_cloud_style_model_answers(self) -> None:
        primary, backup = BoomChat("local"), OkChat("cloud", "备用回答")
        self.kb.qa._generator = FailoverGenerator([primary, backup], threshold=2, cooldown=60)
        answer = self.kb.ask("星云知识库支持什么？")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(answer.model, "cloud")
        self.assertFalse(answer.degraded)
        self.assertIn("备用回答", answer.text)

    def test_dr_02_extractive_when_all_models_fail(self) -> None:
        self.kb.qa._generator = FailoverGenerator(
            [BoomChat("local"), BoomChat("cloud")], threshold=2, cooldown=60,
        )
        answer = self.kb.ask("星云知识库支持什么？")
        self.assertTrue(answer.degraded)
        self.assertEqual(answer.model, "offline-extractive")
        self.assertIn("根据资料", answer.text)

    def test_dr_03_local_and_cloud_share_chat_payload(self) -> None:
        seen: list[dict] = []

        def posting(fail: bool):
            def post(url, _key, payload, timeout=180, max_attempts=1):
                seen.append({"url": url, "payload": payload, "timeout": timeout, "attempts": max_attempts})
                if fail:
                    raise TimeoutError("timed out")
                return {"choices": [{"message": {"content": json.dumps(
                    {"answer": "云端回答", "citations": [], "refused": True}
                )}}]}
            return post

        settings = _settings(self.tmp / "unused")
        local = CloudAnswerGenerator(
            settings, name="local", base_url="http://127.0.0.1:11434/v1",
            api_key="ollama", model="qwen2.5:7b", post=posting(True),
        )
        cloud = CloudAnswerGenerator(
            settings, name="cloud", base_url="https://api.example.com/v1",
            api_key="sk-cloud", model="gpt-4o-mini", post=posting(False),
        )
        self.kb.qa._generator = FailoverGenerator([local, cloud], threshold=2, cooldown=60)
        self.kb.ask("星云知识库支持什么？")
        self.assertEqual(len(seen), 2)
        self.assertTrue(seen[0]["url"].endswith("/chat/completions"))
        self.assertTrue(seen[1]["url"].startswith("https://api.example.com/"))
        for item in seen:
            self.assertIn("model", item["payload"])
            self.assertIsInstance(item["payload"]["messages"], list)
            self.assertEqual(item["attempts"], 1)

    def test_dr_04_breaker_skips_primary(self) -> None:
        now = {"t": 0.0}
        primary, backup = BoomChat("local"), OkChat("cloud", "备用回答")
        self.kb.qa._generator = FailoverGenerator(
            [primary, backup], threshold=2, cooldown=60, clock=lambda: now["t"],
        )
        self.kb.ask("星云知识库支持什么？")
        self.kb.ask("星云知识库计费说明")
        self.kb.ask("星云知识库使用限制")
        self.assertEqual(primary.calls, 2)
        self.assertEqual(backup.calls, 3)

    def test_dr_05_primary_returns_after_cooldown(self) -> None:
        now = {"t": 0.0}
        primary, backup = BoomChat("local"), OkChat("cloud", "备用回答")
        self.kb.qa._generator = FailoverGenerator(
            [primary, backup], threshold=2, cooldown=60, clock=lambda: now["t"],
        )
        self.kb.ask("星云知识库支持什么？")
        self.kb.ask("星云知识库计费说明")
        primary.fail = False
        now["t"] = 60.0
        answer = self.kb.ask("星云知识库使用限制")
        self.assertEqual(primary.calls, 3)
        self.assertEqual(answer.model, "local")

    def test_dr_06_same_dim_embed_backup_is_used(self) -> None:
        kb = KnowledgeBase(_settings(self.tmp / "dim4", embed_dim=4))
        try:
            kb.ingest(self.sample)
            primary, backup = FakeEmb("local", 4, fail=True), FakeEmb("cloud", 4, fail=False)
            kb.retriever._embedder = FailoverEmbedder([primary, backup], 4)
            answer = kb.ask("星云知识库支持什么？")
            self.assertGreater(primary.calls, 0)
            self.assertGreater(backup.calls, 0)
            self.assertFalse(answer.refused)
        finally:
            kb.close()

    def test_dr_07_different_dim_backup_is_not_called(self) -> None:
        fresh = KnowledgeBase(_settings(self.tmp / "fresh"))
        try:
            primary, backup = FakeEmb("local", 4, fail=True), FakeEmb("cloud", 8, fail=False)
            fresh.pipeline._embedder = FailoverEmbedder([primary, backup], 4)
            with self.assertRaises(Exception):
                fresh.ingest(self.sample)
            self.assertEqual(backup.calls, 0)
            self.assertGreater(primary.calls, 0)
            self.assertEqual(fresh.store.health()["chunks"], 0)
        finally:
            fresh.close()

    def test_dr_08_timed_out_model_called_once(self) -> None:
        primary, backup = BoomChat("local"), OkChat("cloud", "备用回答")
        self.kb.qa._generator = FailoverGenerator([primary, backup], threshold=2, cooldown=60)
        self.kb.ask("星云知识库支持什么？")
        self.assertEqual(primary.calls, 1)


class FrozenWebTest(unittest.TestCase):
    def test_web_01_and_web_02(self) -> None:
        from fastapi.testclient import TestClient

        from kb.web import create_app

        with tempfile.TemporaryDirectory() as td:
            app = create_app(_settings(Path(td)))
            client = TestClient(app)
            opened = client.post("/sessions", data={"user_id": "alice"})
            self.assertEqual(opened.status_code, 200)
            sid = opened.json()["session_id"]
            asked = client.post("/ask", data={
                "question": "星云知识库支持什么？",
                "user_id": "alice",
                "session_id": sid,
            })
            body = asked.json()
            self.assertEqual(asked.status_code, 200)
            self.assertEqual(body["session_id"], sid)
            self.assertIn("model", body)
            self.assertIn("degraded", body)
            missing = client.get(f"/sessions/{sid}", params={"user_id": "bob"})
            self.assertEqual(missing.status_code, 404)
