# 本地知识库（第一版）

单人本地运行的文档知识库：导入多格式资料 → 解析/清洗/父子切分 → 建立向量与关键词索引 → 混合检索 + 重排 → 生成**带引用**的回答，资料里没有的内容会**拒答**。

默认全部走离线实现，**无需联网、无需安装数据库**即可跑通；需要更高质量或更大规模时，把嵌入/回答切到云端，把存储切到 PostgreSQL + pgvector，上层不动。

## 快速开始

```bash
pip install -r requirements.txt   # 仅 numpy 为必需
python -m kb demo                 # 内置样例：导入 + 带引用问答 + 拒答演示
```

常用命令：

```bash
python -m kb ingest 路径/文件.pdf 路径/目录    # 导入文件或整目录
python -m kb ingest 路径/大文件 --async        # 放入后台队列
python -m kb worker --once                     # 处理后台队列（可重复、可恢复）
python -m kb ask "报销流程是什么？"            # 提问，返回答案与引用
python -m kb status                            # 文档与任务状态
python -m kb versions 文件.pdf                 # 查看版本
python -m kb rollback 文件.pdf 1               # 回滚到某版本
python -m kb purge 文件.pdf                    # 清理历史版本
python -m kb eval cases.json                   # 评估集回归
python -m kb backup                            # 备份数据库
python -m kb health                            # 健康检查
```

## 运行测试

```bash
python -m unittest discover -s tests
```

## 切换到云端模型

```bash
export OPENAI_API_KEY=你的密钥
export OPENAI_BASE_URL=https://api.openai.com/v1   # 可换成任意兼容接口
export KB_EMBEDDER=openai KB_EMBED_MODEL=text-embedding-3-small
export KB_ANSWERER=openai KB_CHAT_MODEL=gpt-4o-mini
```

云端调用带超时与重试（见 `kb/http.py`）。换嵌入模型需整库重嵌（不同模型向量不可混用）。

## 格式支持

Markdown、TXT 无需额外依赖；PDF、Word、Excel、PPT、图片 OCR 需安装对应可选库（见 `requirements.txt`），缺失时仅该格式给出安装提示，不影响其它格式。

## 模块一览

| 模块 | 职责 |
|------|------|
| `kb/parsers/` | 按扩展名分派的多格式解析 |
| `kb/clean.py` | 去页眉页脚/页码、合并空白 |
| `kb/chunk.py` | 父子切分（子块检索、父块上下文） |
| `kb/embed.py` | 嵌入（离线哈希 / 云端兼容） |
| `kb/store.py` | SQLite：文档版本、块、向量、FTS5、任务、反馈 |
| `kb/retrieve.py` | 向量 + 关键词混合检索与 RRF 融合 |
| `kb/gateway.py` | 回答生成（离线抽取 / 云端，引用校验） |
| `kb/qa.py` | 问答编排与拒答 |
| `kb/jobs.py` | 后台任务队列与断点续跑 |
| `kb/eval.py` | 评估集回归 |
| `kb/service.py` | 统一门面 |
| `kb/cli.py` | 命令行入口 |

## Web 管理界面

```bash
pip install fastapi uvicorn python-multipart
python -m kb serve            # 默认 http://127.0.0.1:8000
```

页面支持上传并索引、提问看引用。接口：`POST /ingest`、`POST /ask`、`GET /status`、`GET /health`、`POST /feedback`。

## 已实现并验证

- 多格式解析分派、内容清洗、父子切分（子块检索 + 父块上下文）
- 向量索引 + FTS5 关键词索引 + RRF 混合检索，向量检索下沉到存储层
- 带引用问答、引用程序校验、低置信度拒答
- 内容哈希去重、文档版本管理（切换/回滚/删除/清理）
- 后台任务队列、批量嵌入、断点续跑（chunk_id 确定性）
- **大文件流式导入**（纯文本/日志类，内存有界）
- 可插拔**重排**（默认融合分 / 云端 rerank 接口）
- **FastAPI Web 界面**、反馈、评估集回归、日志、健康检查、数据库备份
- 命令行入口与自动化测试（`python -m unittest discover -s tests`，16 项通过）

## 已实现、需外部服务或模型（本机未运行）

通过环境变量启用，接口与默认实现一致：

- `KB_STORE=postgres` + `KB_PG_DSN=...`：PostgreSQL + pgvector 存储，向量检索用 HNSW（`kb/pgstore.py`）
- `KB_EMBEDDER=bge`：本地 bge-m3 嵌入（需 sentence-transformers + 模型权重）
- `KB_RERANKER=bge`：本地 bge-reranker 重排（需 FlagEmbedding + 模型权重）
- `KB_RERANKER=cloud` + `KB_RERANK_URL=...`：云端 rerank 接口

## 仍待后续

- PDF/Office 的真正流式解析（当前大文件流式仅覆盖纯文本）
- 分片上传与断点续传的前端
- 中文 OCR 换 PaddleOCR
