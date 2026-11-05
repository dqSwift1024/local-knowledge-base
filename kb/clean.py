"""内容清洗：去掉跨页重复的页眉页脚、独立页码，压缩空白。

清洗产出新的 Block 列表，不改原始解析结果，保证可重新生成。
"""

from __future__ import annotations

import re

from kb.model import Block, ParsedDoc


_PAGE_NUMBER = re.compile(r"^\s*(?:第?\s*\d+\s*页?|-\s*\d+\s*-|\d+\s*/\s*\d+)\s*$")
_WS = re.compile(r"[ \t\u3000]+")


def clean_blocks(doc: ParsedDoc) -> tuple[list[Block], list[str]]:
    warnings: list[str] = []
    pages = {b.page for b in doc.blocks if b.page is not None}
    repeated = _repeated_short_lines(doc.blocks, page_count=len(pages)) if len(pages) >= 4 else set()
    if repeated:
        warnings.append(f"识别并移除了 {len(repeated)} 条疑似页眉/页脚")

    cleaned: list[Block] = []
    for block in doc.blocks:
        text = _WS.sub(" ", block.text).strip()
        if not text:
            continue
        if block.kind != "table" and _PAGE_NUMBER.match(text):
            continue
        if block.kind == "text" and _norm(text) in repeated:
            continue
        cleaned.append(
            Block(kind=block.kind, text=text, page=block.page, level=block.level, locator=block.locator)
        )
    if not cleaned:
        warnings.append("清洗后内容为空")
    return cleaned, warnings


def _repeated_short_lines(blocks: list[Block], page_count: int) -> set[str]:
    """同一条短文本在很多页重复出现，判定为页眉页脚。"""
    seen_pages: dict[str, set[int]] = {}
    for block in blocks:
        if block.kind != "text" or block.page is None or len(block.text) > 40:
            continue
        seen_pages.setdefault(_norm(block.text), set()).add(block.page)
    threshold = max(3, int(page_count * 0.5))
    return {text for text, pages in seen_pages.items() if len(pages) >= threshold}


def _norm(text: str) -> str:
    return _WS.sub("", text).strip().lower()
