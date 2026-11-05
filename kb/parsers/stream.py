"""大文件流式解析：常量内存地算哈希、逐块产出文本/Markdown 内容。

仅覆盖纯文本/Markdown 这类“体积大但结构简单”的资料（日志、导出、长文稿），
它们最容易达到 GB 级。PDF/Office 需各自库支持，暂走整文件路径。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

from kb.model import Block


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = "```"


def hash_file(path: str | Path, buf_size: int = 1 << 20) -> str:
    """分块读取算 SHA-256，内存占用与文件大小无关。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(buf_size), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_markdown_blocks(path: str | Path) -> Iterator[Block]:
    """逐行读取 Markdown，边读边产出标题/段落/代码块，不缓存整篇。"""
    para: list[str] = []
    in_code = False
    code: list[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.strip().startswith(_FENCE):
                if in_code:
                    if code:
                        yield Block(kind="code", text="\n".join(code))
                    code = []
                    in_code = False
                else:
                    yield from _flush_para(para)
                    in_code = True
                continue
            if in_code:
                code.append(line)
                continue

            heading = _HEADING.match(line)
            if heading:
                yield from _flush_para(para)
                yield Block(kind="heading", text=heading.group(2).strip(), level=len(heading.group(1)))
                continue
            if not line.strip():
                yield from _flush_para(para)
                continue
            para.append(line)

    if in_code and code:
        yield Block(kind="code", text="\n".join(code))
    yield from _flush_para(para)


# 日志行时间戳：MM-DD HH:MM:SS(.mmm) 或 YYYY-MM-DD HH:MM:SS，覆盖 Android/syslog 等常见格式
_LOG_LINE = re.compile(r"^\s*(?:\d{4}-)?\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}")
_LOG_PROBE_LINES = 30
_LOG_PROBE_RATIO = 0.6


def looks_like_log_lines(lines) -> bool:
    """抽样前若干非空行，多数带时间戳则判为日志，走按记录切分。供流式与整文件两条路径复用。"""
    hits = total = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        total += 1
        if _LOG_LINE.match(s):
            hits += 1
        if total >= _LOG_PROBE_LINES:
            break
    return total > 0 and hits / total >= _LOG_PROBE_RATIO


def group_log_lines(lines, group_chars: int = 450):
    """把日志行按“记录”聚合再打包：带时间戳的行开启新记录，无时间戳行并入上一条；
    记录打包进不超过 group_chars 的文本块，始终在行边界收口。产出若干块文本（str）。"""
    group: list[str] = []
    group_len = 0
    record: list[str] = []

    def _push_record():
        nonlocal group, group_len, record
        if not record:
            return None
        rec = "\n".join(record)
        record = []
        out = None
        if group and group_len + len(rec) + 1 > group_chars:
            out = "\n".join(group)
            group[:] = []
            group_len = 0
        group.append(rec)
        group_len += len(rec) + 1
        return out

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if _LOG_LINE.match(line) and record:
            emitted = _push_record()
            if emitted is not None:
                yield emitted
        record.append(line)
    emitted = _push_record()
    if emitted is not None:
        yield emitted
    if group:
        yield "\n".join(group)


def _looks_like_log(path: str | Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return looks_like_log_lines(handle)


_LOG_GROUP_CHARS = 450


def iter_log_blocks(path: str | Path, group_chars: int = _LOG_GROUP_CHARS) -> Iterator[Block]:
    """日志流式切分：带时间戳行开启新记录、续行并入，记录打包进 ≤group_chars 的 Block，
    始终在行边界收口。既避免按字数硬切断行，也避免整篇拼成一段，并控制分块数量。"""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for text in group_log_lines(handle, group_chars):
            yield Block(kind="text", text=text)


def iter_text_blocks(path: str | Path) -> Iterator[Block]:
    """逐行读取纯文本，空行分段，边读边产出；日志类文本改为按记录切分。"""
    if _looks_like_log(path):
        yield from iter_log_blocks(path)
        return
    para: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                yield from _flush_para(para)
            else:
                para.append(line)
    yield from _flush_para(para)


def _flush_para(para: list[str]) -> Iterator[Block]:
    if para:
        text = " ".join(s.strip() for s in para).strip()
        para.clear()
        if text:
            yield Block(kind="text", text=text)


STREAM_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}


def iter_blocks(path: str | Path) -> Iterator[Block]:
    ext = Path(path).suffix.lower()
    if ext in {".md", ".markdown"}:
        return iter_markdown_blocks(path)
    return iter_text_blocks(path)
