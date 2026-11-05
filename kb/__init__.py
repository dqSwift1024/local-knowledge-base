"""本地知识库：解析 → 清洗 → 父子切分 → 索引 → 混合检索 → 带引用问答。

单人本地运行，存储用 SQLite + FTS5 + 向量，嵌入与回答走可替换网关
（离线实现零依赖可跑，或切换 OpenAI 兼容云端）。后台任务表支持断点续跑，
命令行入口见 `python -m kb`。规模化（PostgreSQL + pgvector、HNSW、
bge-reranker、真实嵌入、Web 界面）为下一阶段，接口已就位。
"""
