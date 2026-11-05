"""贯穿整条管线的统一数据结构。

所有解析器都把各自格式归一化成同一种 Block，后面的切分、索引、
引用都只认这套结构，不关心原始文件是 PDF 还是 Excel。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal


BlockKind = Literal["heading", "text", "table", "list", "caption", "code"]


@dataclass
class Block:
    """一段带位置信息的内容。位置信息是后面能做引用的前提。"""

    kind: BlockKind
    text: str
    # 页码对 PDF/PPT 有意义；Word/Markdown 没有分页时为 None。
    page: int | None = None
    # 标题层级，1 为最高；仅 kind == "heading" 时有意义。
    level: int | None = None
    # 原始定位线索，例如工作表名、幻灯片序号，排查错位时用。
    locator: str | None = None

    def is_heading(self) -> bool:
        return self.kind == "heading"


@dataclass
class ParsedDoc:
    """一份文件解析后的结果。"""

    source_name: str
    content_hash: str
    blocks: list[Block]
    # 解析器名字与遇到的问题，便于抽查解析质量。
    parser: str = ""
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class Chunk:
    """送去嵌入和检索的最小单位（子块），自带可引用的出处。

    parent_text 是该子块所属章节的更完整文本：检索靠子块精准命中，
    回答时把 parent_text 交给模型，兼顾“找得准”和“读得全”。
    """

    doc_name: str
    content_hash: str
    section_path: str
    text: str
    page: int | None
    parent_text: str = ""
    # 同一文档内稳定、可复现的编号，用作去重键与引用锚点。
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = self.compute_id()

    def compute_id(self) -> str:
        # 仅由子块定位信息决定，不含 parent_text，便于内容不变时 ID 稳定。
        raw = f"{self.content_hash}|{self.section_path}|{self.page}|{self.text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def embed_text(self) -> str:
        """送去嵌入的文本：有真实标题路径时拼上标题（对齐主流“Title+Headings”），
        让片段携带章节语境；纯正文/日志（无标题）则保持原文，不污染。
        展示、FTS、引用与 chunk_id 仍只用 text。"""
        path = (self.section_path or "").strip()
        if path and path != "(正文)":
            return f"{path}\n{self.text}"
        return self.text


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
