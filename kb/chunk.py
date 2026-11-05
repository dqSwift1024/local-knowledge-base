"""父子切分：按标题归属划出章节（父块），再把章节正文切成检索用子块。

- 父块：同一章节下的正文合并，作为回答时交给模型的完整上下文。
- 子块（小）：用于精准检索，引用时也定位到子块。
- 表格整块保留，不从行列中间截断；过大的父块按上限截断，避免撑爆上下文。
- 子块 ID 由内容哈希+章节+页码+子块文本决定，内容不变则 ID 不变，支持增量与续跑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from kb.model import Block, Chunk, ParsedDoc


@dataclass
class _Section:
    path: str
    blocks: list[Block] = field(default_factory=list)


def chunk_document(
    doc: ParsedDoc,
    blocks: list[Block],
    *,
    child_chars: int,
    overlap_chars: int,
    parent_max_chars: int = 2000,
) -> list[Chunk]:
    """便捷接口：一次性返回全部子块（供小文件与测试使用）。"""
    return list(iter_chunks(
        doc, blocks, child_chars=child_chars, overlap_chars=overlap_chars,
        parent_max_chars=parent_max_chars,
    ))


def iter_chunks(
    doc: ParsedDoc,
    blocks: Iterable[Block],
    *,
    child_chars: int,
    overlap_chars: int,
    parent_max_chars: int = 2000,
) -> Iterator[Chunk]:
    """流式切分：按章节边界逐段产出子块，内存只保留当前章节。

    blocks 可以是列表，也可以是来自流式解析的生成器；配合大文件导入时
    整篇文档不会一次性驻留内存。
    """
    for section in _iter_sections(blocks):
        section_blocks = section.blocks
        for idx, block in enumerate(section_blocks):
            # small2big：父块用“邻居窗口”（前一块+本块+后一块）而非整段开头，
            # 保证单章节大文档（如日志）每条拿到的是本条附近的局部上下文，而非共用的篇首。
            parent_text = _neighbor_parent(section_blocks, idx, parent_max_chars)
            pieces = [block.text] if block.kind == "table" else _split_text(
                block.text, child_chars, overlap_chars
            )
            for piece in pieces:
                if piece.strip():
                    yield Chunk(
                        doc_name=doc.source_name,
                        content_hash=doc.content_hash,
                        section_path=section.path,
                        text=piece.strip(),
                        page=block.page,
                        parent_text=parent_text,
                    )


def _iter_sections(blocks: Iterable[Block]) -> Iterator[_Section]:
    """每遇到一个标题就结束当前章节；连续正文归入同一章节。流式产出。"""
    heading_stack: list[tuple[int, str]] = []
    current: _Section | None = None

    for block in blocks:
        if block.is_heading():
            if current is not None:
                yield current
                current = None
            level = block.level or 2
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text))
            continue
        if current is None:
            path = " / ".join(text for _level, text in heading_stack) or "(正文)"
            current = _Section(path=path)
        current.blocks.append(block)
    if current is not None:
        yield current


def _parent_text(blocks: list[Block], max_chars: int) -> str:
    joined = "\n".join(b.text for b in blocks if b.text.strip())
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rstrip() + "…"


def _neighbor_parent(blocks: list[Block], idx: int, max_chars: int) -> str:
    """以第 idx 块为中心，向两侧扩展邻居块，拼成不超过 max_chars 的局部上下文。

    小章节会把整段纳入（与旧行为一致）；大章节（日志等）只取本块附近，
    避免所有子块共用篇首，保证回答时模型看到的是命中内容周围的上下文。
    """
    texts = [b.text for b in blocks]
    center = texts[idx] if texts[idx].strip() else ""
    acc = center
    lo = hi = idx
    while True:
        grew = False
        if lo - 1 >= 0 and len(acc) + len(texts[lo - 1]) + 1 <= max_chars:
            lo -= 1
            acc = texts[lo] + "\n" + acc
            grew = True
        if hi + 1 < len(texts) and len(acc) + len(texts[hi + 1]) + 1 <= max_chars:
            hi += 1
            acc = acc + "\n" + texts[hi]
            grew = True
        if not grew:
            break
    if len(acc) > max_chars:
        acc = acc[:max_chars].rstrip() + "…"
    return acc


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        cut = _soft_cut(text, start, end)
        pieces.append(text[start:cut])
        if cut >= len(text):
            break
        next_start = cut - overlap
        # 保证前进，避免重叠过大时原地打转
        start = next_start if next_start > start else cut
    return pieces


def _soft_cut(text: str, start: int, end: int) -> int:
    """尽量在标点处收口，避免从句子中间断开。"""
    if end >= len(text):
        return len(text)
    window = text[start:end]
    for punct in ("。", "！", "？", "\n", "；", ".", "!", "?"):
        pos = window.rfind(punct)
        if pos >= int((end - start) * 0.6):
            return start + pos + 1
    return end
