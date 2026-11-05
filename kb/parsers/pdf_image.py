"""PDF 与图片解析。PDF 文字版直接取文字，扫描页和图片走 OCR（有则用）。"""

from __future__ import annotations

import io

from kb.model import Block
from kb.parsers._deps import optional, require


def parse_pdf(data: bytes, _name: str) -> list[Block]:
    fitz = require("fitz")
    doc = fitz.open(stream=data, filetype="pdf")
    blocks: list[Block] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_no = page_index + 1
        page_blocks = _blocks_from_page_dict(page, page_no)
        if not page_blocks:
            ocr_text = _ocr_pdf_page(page)
            if ocr_text:
                blocks.append(Block(kind="text", text=ocr_text, page=page_no, locator="OCR"))
        else:
            blocks.extend(page_blocks)
    doc.close()
    return blocks


def parse_image(data: bytes, name: str) -> list[Block]:
    text = _ocr_image_bytes(data)
    if not text:
        raise RuntimeError(
            f"图片 {name} 需要 OCR 才能提取文字，但未安装 pytesseract/tesseract。"
            "请安装后重试：pip install pytesseract pillow，brew install tesseract tesseract-lang"
        )
    return [Block(kind="text", text=text, locator="OCR")]


def _blocks_from_page_dict(page, page_no: int) -> list[Block]:
    """用字号把明显更大的行识别为标题，其余按行聚成段落。"""
    data = page.get_text("dict")
    sizes: list[float] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    sizes.append(span.get("size", 0.0))
    if not sizes:
        return []
    body_size = sorted(sizes)[len(sizes) // 2]  # 中位字号当正文基准

    blocks: list[Block] = []
    for block in data.get("blocks", []):
        lines = block.get("lines", [])
        if not lines:
            continue
        line_texts: list[str] = []
        max_size = 0.0
        for line in lines:
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if line_text:
                line_texts.append(line_text)
                max_size = max(max_size, *(span.get("size", 0.0) for span in spans))
        text = _join_lines(line_texts)
        if not text:
            continue
        if max_size >= body_size * 1.3 and len(text) <= 40:
            blocks.append(Block(kind="heading", text=text, level=2, page=page_no))
        else:
            blocks.append(Block(kind="text", text=text, page=page_no))
    return blocks


def _join_lines(lines: list[str]) -> str:
    """拼接 PDF 断行：英文连字符接续，其余用空格，中文之间不留空格。"""
    out = ""
    for line in lines:
        piece = line.strip()
        if not out:
            out = piece
            continue
        if out.endswith("-"):
            out = out[:-1] + piece
        elif _is_cjk(out[-1]) and piece and _is_cjk(piece[0]):
            out += piece
        else:
            out += " " + piece
    return out.strip()


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _ocr_pdf_page(page) -> str:
    pytesseract = optional("pytesseract")
    pil = optional("PIL.Image")
    if pytesseract is None or pil is None:
        return ""
    pix = page.get_pixmap(dpi=200)
    image = pil.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang=_ocr_lang()).strip()


def _ocr_image_bytes(data: bytes) -> str:
    pytesseract = optional("pytesseract")
    pil = optional("PIL.Image")
    if pytesseract is None or pil is None:
        return ""
    image = pil.open(io.BytesIO(data))
    return pytesseract.image_to_string(image, lang=_ocr_lang()).strip()


def _ocr_lang() -> str:
    return "chi_sim+eng"
