"""回答生成层。离线抽取式零依赖可跑；云端走 OpenAI 兼容 /chat/completions（带重试）。

两种实现都返回 {answer, citations}，citations 为引用的 chunk_id 列表，
由上层 QA 校验这些 id 确实出现在本次检索结果中，杜绝编造引用。
回答上下文使用父块文本（更完整），引用仍指向子块（更精准）。
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from kb.config import Settings
from kb.http import post_json
from kb.retrieve import Retrieved


SYSTEM_PROMPT = """你是知识库问答助手。只能依据下面提供的资料片段回答。
每个片段以 [编号] 开头。回答中每个结论后用 [编号] 标注依据，编号必须来自给定片段。
资料不足以回答时，refused 置为 true。
只输出 JSON 对象：{"answer": "...", "citations": ["片段编号"], "refused": false}。"""


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: list[Retrieved], history: list[dict] | None = None) -> dict: ...


class OfflineExtractive:
    """从检索片段里抽取前几条组织成答案，引用即这些片段。用于离线闭环。"""

    def __init__(self, max_snippets: int = 3) -> None:
        self._max = max_snippets

    def generate(self, question: str, contexts: list[Retrieved], history: list[dict] | None = None) -> dict:
        if not contexts:
            return {"answer": "", "citations": [], "refused": True}
        chosen = contexts[: self._max]
        lines = [f"根据资料，与「{question}」相关的内容如下："]
        for i, ctx in enumerate(chosen, start=1):
            lines.append(f"{i}. {_lead_sentence(ctx.text)} [{ctx.chunk_id}]")
        return {
            "answer": "\n".join(lines),
            "citations": [ctx.chunk_id for ctx in chosen],
            "refused": False,
        }


class CloudAnswerGenerator:
    """调用兼容接口，要求模型只输出带引用的 JSON；上下文用父块文本。"""

    def __init__(
        self,
        settings: Settings,
        *,
        name: str = "primary",
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        post=None,
    ) -> None:
        key = settings.api_key if api_key is None else api_key
        if not key:
            raise RuntimeError("使用 openai 回答需要设置 OPENAI_API_KEY")
        root = (base_url or settings.base_url).rstrip("/")
        self.name = name
        self._url = root + "/chat/completions"
        self._key = key
        self._model = model or settings.chat_model
        self._post = post or post_json

    def generate(self, question: str, contexts: list[Retrieved], history: list[dict] | None = None) -> dict:
        context_text = "\n\n".join(f"[{c.chunk_id}] {c.parent_text}" for c in contexts)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history or []:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": f"资料片段：\n{context_text}\n\n问题：{question}"})
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 512,
            # 让本地模型在空闲后仍留在内存里，避免每次提问都重新加载并超时
            "keep_alive": "30m",
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        # 每个模型只请求一次。超时后立刻重试会在本地模型里叠加上一次还没结束的生成。
        body = self._post(self._url, self._key, payload, timeout=180, max_attempts=1)
        return _coerce(body["choices"][0]["message"]["content"])


def build_generator(settings: Settings) -> AnswerGenerator:
    from kb.failover import FailoverGenerator, build_chat_models

    models = build_chat_models(settings)
    if not models:
        return OfflineExtractive()
    return FailoverGenerator(
        models,
        OfflineExtractive(),
        threshold=settings.failover_threshold,
        cooldown=settings.failover_cooldown_s,
    )


def _lead_sentence(text: str) -> str:
    parts = re.split(r"(?<=[。！？.!?])", text, maxsplit=1)
    lead = parts[0].strip() if parts else text
    return lead[:160]


def _coerce(content: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {"answer": content.strip(), "citations": [], "refused": False}
    data.setdefault("answer", "")
    data.setdefault("citations", [])
    data.setdefault("refused", False)
    # 模型常把引用写成 [编号]，去掉包裹的方括号/空白，保证与真实 chunk_id 校验一致
    data["citations"] = [_clean_cite(c) for c in data.get("citations", []) if _clean_cite(c)]
    return data


def _clean_cite(raw: object) -> str:
    return str(raw).strip().strip("[]").strip()
