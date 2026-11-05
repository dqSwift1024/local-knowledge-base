"""按扩展名把文件分派给对应解析器。

新增一种格式 = 写一个返回 list[Block] 的函数并在 _REGISTRY 登记，
管线其余部分不动。重的依赖（PDF/OCR 等）在各自解析器里懒加载，
缺依赖时给出明确的安装提示，而不是在导入本模块时就报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from kb.model import Block, ParsedDoc, hash_bytes
from kb.parsers import office, pdf_image, text_md


# 扩展名（小写，含点） -> (解析器名, 解析函数)
_REGISTRY: dict[str, tuple[str, Callable[[bytes, str], list[Block]]]] = {
    ".md": ("markdown", text_md.parse_markdown),
    ".markdown": ("markdown", text_md.parse_markdown),
    ".txt": ("text", text_md.parse_text),
    ".text": ("text", text_md.parse_text),
    ".pdf": ("pdf", pdf_image.parse_pdf),
    ".docx": ("docx", office.parse_docx),
    ".xlsx": ("xlsx", office.parse_xlsx),
    ".pptx": ("pptx", office.parse_pptx),
    ".png": ("image-ocr", pdf_image.parse_image),
    ".jpg": ("image-ocr", pdf_image.parse_image),
    ".jpeg": ("image-ocr", pdf_image.parse_image),
    ".webp": ("image-ocr", pdf_image.parse_image),
}


class UnsupportedFormat(Exception):
    """没有登记对应扩展名的解析器。"""


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY)


def parse_file(path: str | Path) -> ParsedDoc:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    entry = _REGISTRY.get(ext)
    if entry is None:
        raise UnsupportedFormat(
            f"暂不支持的格式 {ext or '(无扩展名)'}；已支持：{', '.join(supported_extensions())}"
        )
    parser_name, parse = entry
    data = file_path.read_bytes()
    warnings: list[str] = []
    blocks = parse(data, file_path.name)
    if not blocks:
        warnings.append("解析结果为空，请确认文件内容或是否需要 OCR")
    return ParsedDoc(
        source_name=file_path.name,
        content_hash=hash_bytes(data),
        blocks=blocks,
        parser=parser_name,
        warnings=warnings,
        meta={"size_bytes": len(data), "ext": ext},
    )
