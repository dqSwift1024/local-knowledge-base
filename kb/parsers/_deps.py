"""可选依赖的懒加载：缺库时给出可直接执行的安装命令。

重解析库只在真正用到某种格式时才导入，因此 import 本模块、乃至
整个 kb 包，都不要求预先装好 PDF/Office/OCR 相关依赖。
"""

from __future__ import annotations

import importlib
from typing import Any


_INSTALL_HINTS = {
    "fitz": "pip install pymupdf",
    "docx": "pip install python-docx",
    "openpyxl": "pip install openpyxl",
    "pptx": "pip install python-pptx",
    "PIL": "pip install pillow",
    "pytesseract": "pip install pytesseract，并安装系统 tesseract：brew install tesseract tesseract-lang",
}


class MissingDependency(Exception):
    """运行到某格式解析时才发现缺少对应依赖。"""


def require(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # noqa: PERF203 — 仅在首次解析该格式时触发
        hint = _INSTALL_HINTS.get(module, f"pip install {module}")
        raise MissingDependency(f"缺少依赖 {module}，请先安装：{hint}") from exc


def optional(module: str) -> Any | None:
    """用于 OCR 这类“有就用、没有就降级”的依赖。"""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None
