"""命令行入口：python -m kb <命令>。

子命令：ingest / status / ask / versions / rollback / delete / purge /
worker / eval / backup / health / demo。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from kb.config import load_settings
from kb.logging_setup import configure
from kb.service import KnowledgeBase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb", description="本地知识库")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="导入文件或目录")
    p_ingest.add_argument("paths", nargs="+")
    p_ingest.add_argument("--async", dest="async_", action="store_true", help="放入后台队列")
    p_ingest.add_argument("--no-recursive", dest="recursive", action="store_false")

    sub.add_parser("status", help="查看文档与任务状态")

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")

    p_versions = sub.add_parser("versions", help="查看某文档的版本")
    p_versions.add_argument("doc_name")

    p_rollback = sub.add_parser("rollback", help="把某版本切为最新")
    p_rollback.add_argument("doc_name")
    p_rollback.add_argument("version", type=int)

    p_delete = sub.add_parser("delete", help="删除某版本（按内容哈希）")
    p_delete.add_argument("content_hash")

    p_purge = sub.add_parser("purge", help="清理某文档的非最新版本")
    p_purge.add_argument("doc_name")

    p_worker = sub.add_parser("worker", help="处理后台队列")
    p_worker.add_argument("--once", action="store_true", help="处理完当前队列即退出")

    p_eval = sub.add_parser("eval", help="运行评估集（JSON 文件）")
    p_eval.add_argument("cases_file")

    p_backup = sub.add_parser("backup", help="备份数据库")
    p_backup.add_argument("dest", nargs="?")

    p_serve = sub.add_parser("serve", help="启动 Web 管理界面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("health", help="健康检查")
    sub.add_parser("demo", help="内置样例演示导入与问答")

    args = parser.parse_args(argv)
    settings = load_settings()
    configure(settings)
    kb = KnowledgeBase(settings)
    try:
        return _dispatch(kb, args)
    finally:
        kb.close()


def _dispatch(kb: KnowledgeBase, args: argparse.Namespace) -> int:
    if args.command == "ingest":
        for raw in args.paths:
            path = Path(raw)
            if args.async_:
                job_id = kb.enqueue(path)
                print(f"已入队 job={job_id}: {path}")
            elif path.is_dir():
                for report in kb.ingest_dir(path, recursive=args.recursive):
                    print(f"{report.status}\t{report.chunk_count}\t{report.doc_name}")
            else:
                report = kb.ingest(path)
                print(f"{report.status}\t{report.chunk_count}\t{report.doc_name}")
        return 0

    if args.command == "status":
        for doc in kb.store.list_docs():
            latest = "*" if doc.is_latest else " "
            print(f"{latest} v{doc.version}\t{doc.status}/{doc.stage}\t{doc.chunk_count}块\t{doc.doc_name}")
        jobs = kb.store.list_jobs()
        if jobs:
            print("--- 任务 ---")
            for job in jobs:
                print(f"#{job.id}\t{job.status}\t尝试{job.attempts}\t{job.path}")
        return 0

    if args.command == "ask":
        answer = kb.ask(args.question)
        print(answer.text)
        if answer.refused:
            if answer.nearest:
                print("\n最接近的内容：")
                for c in answer.nearest:
                    print(f"  - {c.doc_name} / {c.section_path} / 第{c.page}页")
        else:
            print("\n引用：")
            for c in answer.citations:
                page = f"第{c.page}页" if c.page is not None else "无页码"
                print(f"  [{c.chunk_id}] {c.doc_name} / {c.section_path} / {page}")
        print(f"\n置信度：{answer.confidence:.3f}")
        return 0

    if args.command == "versions":
        for doc in kb.versions(args.doc_name):
            latest = "*" if doc.is_latest else " "
            print(f"{latest} v{doc.version}\t{doc.content_hash}\t{doc.chunk_count}块")
        return 0

    if args.command == "rollback":
        ok = kb.rollback(args.doc_name, args.version)
        print("已切换" if ok else "未找到该版本")
        return 0 if ok else 1

    if args.command == "delete":
        ok = kb.delete_version(args.content_hash)
        print("已删除" if ok else "未找到该版本")
        return 0 if ok else 1

    if args.command == "purge":
        n = kb.purge_superseded(args.doc_name)
        print(f"已清理 {n} 个历史版本")
        return 0

    if args.command == "worker":
        processed = kb.run_worker_once() if args.once else _run_forever(kb)
        print(f"处理任务 {processed} 个")
        return 0

    if args.command == "eval":
        cases = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
        report = kb.evaluate(cases)
        print(f"命中率(前{kb.settings.final_top_k})：{report.recall:.0%}  "
              f"引用有效率：{report.citation_validity:.0%}  "
              f"拒答准确率：{report.refusal_accuracy:.0%}")
        return 0

    if args.command == "backup":
        dest = kb.backup(args.dest)
        print(f"已备份到 {dest}")
        return 0

    if args.command == "serve":
        import uvicorn

        from kb.api.app import create_app

        print(f"Web 界面：http://{args.host}:{args.port}")
        uvicorn.run(create_app(kb.settings), host=args.host, port=args.port)
        return 0

    if args.command == "health":
        print(json.dumps(kb.health(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "demo":
        return _demo(kb)

    return 1


def _run_forever(kb: KnowledgeBase) -> int:
    try:
        kb.worker.run_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _demo(kb: KnowledgeBase) -> int:
    with tempfile.TemporaryDirectory() as td:
        sample = Path(td) / "示例.md"
        sample.write_text(
            "# 产品说明\n星云知识库支持文档检索和引用溯源。\n\n"
            "## 使用限制\n单文件上限为一GB，超出需拆分。\n",
            encoding="utf-8",
        )
        report = kb.ingest(sample)
        print(f"导入：{report.status}，{report.chunk_count} 块")
        for question in ["星云知识库能做什么？", "单文件上限是多少？", "它支持语音通话吗？"]:
            answer = kb.ask(question)
            tag = "拒答" if answer.refused else f"{len(answer.citations)}条引用"
            print(f"\nQ: {question}\nA: {answer.text}\n[{tag}, 置信度 {answer.confidence:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
