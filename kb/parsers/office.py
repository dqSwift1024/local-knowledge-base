"""Word / Excel / PowerPoint 解析，统一输出带位置信息的 Block。"""

from __future__ import annotations

import io

from kb.model import Block
from kb.parsers._deps import require


def parse_docx(data: bytes, _name: str) -> list[Block]:
    docx = require("docx")
    document = docx.Document(io.BytesIO(data))
    blocks: list[Block] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            level = _heading_level(style)
            blocks.append(Block(kind="heading", text=text, level=level))
        elif style.startswith("title"):
            blocks.append(Block(kind="heading", text=text, level=1))
        else:
            blocks.append(Block(kind="text", text=text))
    for index, table in enumerate(document.tables, start=1):
        rows = ["\t".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        rows = [r for r in rows if r.strip()]
        if rows:
            blocks.append(Block(kind="table", text="\n".join(rows), locator=f"表{index}"))
    return blocks


def parse_xlsx(data: bytes, _name: str) -> list[Block]:
    openpyxl = require("openpyxl")
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    blocks: list[Block] = []
    for sheet in workbook.worksheets:
        blocks.append(Block(kind="heading", text=sheet.title, level=1, locator=f"工作表 {sheet.title}"))
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if v is None else str(v).strip() for v in row]
            if any(cells):
                rows.append("\t".join(cells))
            if len(rows) >= 2000:  # 单表过大时分块，避免单个 Block 过长
                blocks.append(Block(kind="table", text="\n".join(rows), locator=f"工作表 {sheet.title}"))
                rows = []
        if rows:
            blocks.append(Block(kind="table", text="\n".join(rows), locator=f"工作表 {sheet.title}"))
    workbook.close()
    return blocks


def parse_pptx(data: bytes, _name: str) -> list[Block]:
    pptx = require("pptx")
    presentation = pptx.Presentation(io.BytesIO(data))
    blocks: list[Block] = []
    for index, slide in enumerate(presentation.slides, start=1):
        blocks.append(Block(kind="heading", text=f"第 {index} 页幻灯片", level=1, page=index))
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if text:
                    blocks.append(Block(kind="text", text=text, page=index))
    return blocks


def _heading_level(style: str) -> int:
    digits = "".join(ch for ch in style if ch.isdigit())
    if digits:
        return min(int(digits), 6)
    return 2
