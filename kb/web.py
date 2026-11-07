"""兼容入口。新代码请使用 kb.api.create_app，页面在 frontend/。"""

from kb.api.app import create_app

__all__ = ["create_app"]
