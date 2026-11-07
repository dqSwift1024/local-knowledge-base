"""按资源拆开的接口。"""

from kb.api.routes.documents import register_documents
from kb.api.routes.qa import register_qa
from kb.api.routes.system import register_system

__all__ = ["register_documents", "register_qa", "register_system"]
