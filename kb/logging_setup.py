"""统一日志：同时输出到控制台和文件，供 CLI 与 Worker 使用。"""

from __future__ import annotations

import logging
from pathlib import Path

from kb.config import Settings


_CONFIGURED = False


def configure(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("kb")
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True
