"""带超时与重试的最小 HTTP 客户端，供云端嵌入与回答共用。

把限流（429）和服务端临时故障（5xx、超时）做成自动重试，
业务代码（embed / gateway）只管组织请求与解析结果。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class HttpError(Exception):
    """重试用尽或遇到不可重试状态码后抛出。"""


def post_json(
    url: str,
    api_key: str,
    payload: dict,
    *,
    timeout: float = 60,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in RETRYABLE_STATUS and attempt < max_attempts:
                last_error = HttpError(f"HTTP {exc.code}: {detail}")
                sleep(min(2 ** (attempt - 1), 4))
                continue
            raise HttpError(f"HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < max_attempts:
                sleep(min(2 ** (attempt - 1), 4))
                continue
            raise HttpError(str(exc)) from exc

    raise HttpError(str(last_error))
