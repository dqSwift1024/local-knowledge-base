"""纯文本与 Markdown 解析，仅用标准库，便于最小闭环先跑通。"""

from __future__ import annotations

import re

from kb.model import Block


_FENCE = "```"
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*([-*+]|\d+\.)\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def parse_markdown(data: bytes, _name: str) -> list[Block]:
    text = data.decode("utf-8", errors="replace")
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        if para:
            joined = " ".join(s.strip() for s in para).strip()
            if joined:
                blocks.append(Block(kind="text", text=joined))
            para.clear()

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith(_FENCE):
            flush_para()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(_FENCE):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束围栏
            if code_lines:
                blocks.append(Block(kind="code", text="\n".join(code_lines)))
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_para()
            blocks.append(
                Block(kind="heading", text=heading.group(2).strip(), level=len(heading.group(1)))
            )
            i += 1
            continue

        if _TABLE_ROW.match(line):
            flush_para()
            table_lines: list[str] = []
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                table_lines.append(lines[i].strip())
                i += 1
            table_lines = [r for r in table_lines if not re.fullmatch(r"[\s|:\-]+", r)]
            if table_lines:
                blocks.append(Block(kind="table", text="\n".join(table_lines)))
            continue

        if _BULLET.match(line):
            flush_para()
            item_lines: list[str] = []
            while i < len(lines) and _BULLET.match(lines[i]):
                item_lines.append(lines[i].strip())
                i += 1
            blocks.append(Block(kind="list", text="\n".join(item_lines)))
            continue

        if not line.strip():
            flush_para()
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para()
    return blocks


def parse_text(data: bytes, _name: str) -> list[Block]:
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # 日志类文本：按记录成块（不从行中间截断），与流式路径一致
    from kb.parsers.stream import looks_like_log_lines, group_log_lines
    if looks_like_log_lines(lines):
        return [Block(kind="text", text=t) for t in group_log_lines(lines)]
    blocks: list[Block] = []
    for chunk in re.split(r"\n\s*\n", text):
        cleaned = " ".join(s.strip() for s in chunk.splitlines()).strip()
        if cleaned:
            blocks.append(Block(kind="text", text=cleaned))
    return blocks
