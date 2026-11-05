"""中英文混合分词，供关键词索引与 token 重合度计算共用。

中文没有空格，FTS5 默认分词器切不开，这里把中文切成字二元组、
英文数字按词切，索引和查询用同一套，保证检索召回一致。
"""

from __future__ import annotations

import re


# 标识符：字母数字用 : . _ - 连接的整体，如 MAC 地址、UUID、版本号、包名路径。
# 整体保留为一个 token，避免被切成 54 77 87 这类上万行通用的碎片，稀释精确检索。
_IDENT = re.compile(r"[a-z0-9]+(?:[:._-][a-z0-9]+)+")
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    # 先抽出标识符整体，占位后再做常规分词，兼顾“整体精确命中”和“碎片召回”
    idents: list[str] = []
    def _stash(m: re.Match) -> str:
        idents.append(m.group(0))
        return " "
    masked = _IDENT.sub(_stash, lowered)
    tokens.extend(idents)
    for match in _TOKEN.finditer(masked):
        piece = match.group(0)
        if piece[0].isascii():
            tokens.append(piece)
        elif len(piece) == 1:
            tokens.append(piece)
        else:
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    # 标识符再补上其字母数字碎片，保证部分匹配仍可召回
    for ident in idents:
        for sub in re.findall(r"[a-z0-9]+", ident):
            tokens.append(sub)
    return tokens


def index_text(text: str) -> str:
    """落库时存成空格分隔的 token 串，交给 FTS5 索引。"""
    return " ".join(tokenize(text))


def match_query(text: str) -> str:
    """FTS5 MATCH 查询：token 用引号包裹并以 OR 连接，提高召回。"""
    tokens = tokenize(text)
    if not tokens:
        return ""
    unique = list(dict.fromkeys(tokens))
    return " OR ".join(f'"{tok}"' for tok in unique)
