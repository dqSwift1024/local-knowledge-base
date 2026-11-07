"""HTTP 接口。只负责请求和响应，业务在 kb 的领域模块里。"""

from kb.api.app import create_app

__all__ = ["create_app"]
