"""FastAPI 管理界面：上传、提问、运行状态。

导入写入收件目录并进入任务队列，后台线程执行。页面轮询 /status
就能看到排队、阶段和完成。提问仍同步返回，并带上分步耗时。
"""

import threading
import time
from pathlib import Path
from typing import Optional

from kb.config import Settings, load_settings
from kb.http import HttpError
from kb.service import KnowledgeBase


def create_app(settings: Optional[Settings] = None):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    settings = settings or load_settings()
    kb = KnowledgeBase(settings)
    app = FastAPI(title="本地知识库")
    # 请求线程共用 kb 的连接，必须串行。后台导入另开一条连接。
    db_lock = threading.Lock()
    _start_worker(settings, kb.embedder)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)) -> JSONResponse:
        name = Path(file.filename or "upload").name
        inbox = Path(settings.db_path).parent / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        target = inbox / f"{int(time.time() * 1000)}_{name}"
        target.write_bytes(await file.read())
        with db_lock:
            job_id = kb.store.enqueue_job(str(target))
        return JSONResponse({"job_id": job_id, "doc_name": name, "status": "queued"})

    @app.post("/ask")
    def ask(question: str = Form(...)) -> JSONResponse:
        try:
            with db_lock:
                answer = kb.ask(question)
        except HttpError:
            return JSONResponse(
                {"detail": "回答模型响应超时。模型可能正在加载，请过几秒再问一次。"},
                status_code=503,
            )
        return JSONResponse({
            "question": answer.question, "text": answer.text, "refused": answer.refused,
            "confidence": round(answer.confidence, 3),
            "trace": answer.trace,
            "citations": [
                {"chunk_id": c.chunk_id, "doc_name": c.doc_name,
                 "section_path": c.section_path, "page": c.page, "text": c.text}
                for c in answer.citations
            ],
        })

    @app.get("/status")
    def status() -> JSONResponse:
        with db_lock:
            docs = kb.store.list_docs()
            jobs = kb.store.list_jobs()
            health = kb.health()
        busy = any(j.status in ("pending", "running", "failed") for j in jobs)
        running_doc = next((d for d in docs if d.status == "running"), None)
        return JSONResponse({
            "busy": busy,
            "health": health,
            "runtime": {
                "embedder": kb.embedder.name,
                "answerer": "摘录" if settings.answerer == "offline" else settings.chat_model,
            },
            "active_stage": running_doc.stage if running_doc else "",
            "docs": [
                {"doc_name": d.doc_name, "version": d.version, "is_latest": d.is_latest,
                 "content_hash": d.content_hash,
                 "status": d.status, "stage": d.stage, "chunks": d.chunk_count,
                 "error": d.error}
                for d in docs if d.is_latest
            ],
            "jobs": [
                {"id": j.id, "status": j.status, "stage": j.stage, "attempts": j.attempts,
                 "path": Path(j.path).name, "error": j.error, "updated_at": j.updated_at}
                for j in jobs[-20:]
            ],
        })

    @app.delete("/documents/{content_hash}")
    def delete_document(content_hash: str) -> JSONResponse:
        with db_lock:
            doc = kb.store.get_doc(content_hash)
            if doc is None:
                return JSONResponse({"ok": False, "error": "文档不存在"}, status_code=404)
            deleted = kb.delete_version(content_hash)
        return JSONResponse({
            "ok": deleted,
            "doc_name": doc.doc_name,
            "content_hash": content_hash,
        })

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(kb.health())

    @app.post("/feedback")
    def feedback(question: str = Form(...), useful: bool = Form(...)) -> JSONResponse:
        with db_lock:
            kb.store.add_feedback(question, useful, False, [])
        return JSONResponse({"ok": True})

    app.state.kb = kb
    return app


def _start_worker(settings: Settings, embedder) -> None:
    """后台线程使用自己的数据库连接，避免和页面轮询抢同一条连接。"""
    from kb.jobs import Worker
    from kb.pipeline import Pipeline
    from kb.store import Store

    def loop() -> None:
        store = Store(settings.db_path)
        worker = Worker(store, Pipeline(store, embedder, settings), settings.max_job_attempts)
        while True:
            job = store.next_pending_job()
            if job is None:
                time.sleep(0.4)
                continue
            if job.attempts >= settings.max_job_attempts:
                store.update_job(job.id, status="dead", stage="ingest", error=job.error or "超过重试次数")
                continue
            worker.process_one(job)

    threading.Thread(target=loop, name="kb-worker", daemon=True).start()


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>知屿 · 本地知识库</title>
<style>
 :root {
   color-scheme: light;
   --bg:#f4f6f9; --surface:#ffffff; --surface-2:#f8fafc; --text:#18212f;
   --muted:#667085; --subtle:#98a2b3; --line:#e4e7ec; --line-strong:#d0d5dd;
   --brand:#4f46e5; --brand-hover:#4338ca; --brand-soft:#eef2ff;
   --success:#067647; --success-soft:#ecfdf3; --warning:#b54708; --warning-soft:#fffaeb;
   --danger:#b42318; --danger-soft:#fef3f2;
   --shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.05);
 }
 * { box-sizing:border-box; }
 body {
   margin:0; min-height:100vh; background:var(--bg); color:var(--text);
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
   font-size:14px; line-height:1.55; -webkit-font-smoothing:antialiased;
 }
 button,input,textarea { font:inherit; }
 button { border:0; }
 .shell { max-width:1180px; margin:0 auto; padding:28px 24px 56px; }
 .topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:24px; margin-bottom:24px; }
 .brand { display:flex; align-items:center; gap:13px; }
 .brand-mark {
   width:42px; height:42px; display:grid; place-items:center; border-radius:12px;
   color:#fff; background:var(--brand); font-size:17px; font-weight:700; letter-spacing:-.04em;
 }
 h1,h2,h3,p { margin:0; }
 h1 { font-size:22px; line-height:1.25; letter-spacing:-.02em; }
 .brand-copy p { color:var(--muted); margin-top:3px; }
 .live { display:flex; align-items:center; gap:8px; color:var(--muted); padding-top:9px; white-space:nowrap; }
 .live-dot { width:8px; height:8px; border-radius:50%; background:var(--success); }
 .live-dot.busy { background:#f79009; animation:pulse 1.4s infinite; }
 @keyframes pulse { 50% { opacity:.35; } }
 .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }
 .metric { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:15px 17px; }
 .metric-label { color:var(--muted); font-size:12px; margin-bottom:5px; }
 .metric-value { font-size:20px; font-weight:650; letter-spacing:-.02em; }
 .metric-detail { color:var(--subtle); font-size:12px; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
 .layout { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(300px,1fr); gap:18px; align-items:start; }
 .stack { display:grid; gap:18px; }
 .card { background:var(--surface); border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow); overflow:hidden; }
 .card-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding:20px 22px 0; }
 .card-title { font-size:16px; font-weight:650; letter-spacing:-.01em; }
 .card-subtitle { color:var(--muted); font-size:13px; margin-top:3px; }
 .card-body { padding:18px 22px 22px; }
 .composer { display:flex; gap:10px; align-items:flex-end; }
 textarea {
   width:100%; min-height:92px; resize:vertical; border:1px solid var(--line-strong); border-radius:10px;
   padding:12px 14px; color:var(--text); background:var(--surface); outline:none; transition:.15s;
 }
 textarea:focus,input:focus { border-color:var(--brand); box-shadow:0 0 0 3px var(--brand-soft); }
 .btn {
   min-height:42px; display:inline-flex; align-items:center; justify-content:center; gap:7px;
   border-radius:9px; padding:0 16px; cursor:pointer; font-weight:600; transition:.15s; white-space:nowrap;
 }
 .btn-primary { color:#fff; background:var(--brand); }
 .btn-primary:hover { background:var(--brand-hover); }
 .btn-secondary { color:var(--text); background:var(--surface); border:1px solid var(--line-strong); }
 .btn:disabled { opacity:.55; cursor:wait; }
 .shortcut { display:flex; justify-content:space-between; gap:12px; margin-top:8px; color:var(--subtle); font-size:12px; }
 .answer { margin-top:18px; border-top:1px solid var(--line); padding-top:18px; }
 .answer[hidden],.trace[hidden],.notice[hidden] { display:none; }
 .answer-meta { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
 .answer-text { white-space:pre-wrap; font-size:15px; line-height:1.75; }
 .trace { display:flex; flex-wrap:wrap; gap:7px; margin-top:16px; }
 .trace-item { display:flex; align-items:center; gap:6px; border-radius:7px; padding:5px 8px; background:var(--surface-2); color:var(--muted); font-size:12px; }
 .trace-item::before { content:""; width:6px; height:6px; border-radius:50%; background:var(--success); }
 .trace-item.running::before { background:#f79009; animation:pulse 1.4s infinite; }
 .trace-item.bad::before { background:var(--danger); }
 .sources { display:grid; gap:8px; margin-top:14px; }
 details.source { border:1px solid var(--line); border-radius:9px; background:var(--surface-2); }
 details.source summary { cursor:pointer; padding:10px 12px; color:var(--muted); list-style:none; }
 details.source summary::-webkit-details-marker { display:none; }
 .source-text { padding:0 12px 12px; color:var(--muted); border-top:1px solid var(--line); padding-top:10px; }
 .badge { display:inline-flex; align-items:center; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:600; }
 .badge.ok { color:var(--success); background:var(--success-soft); }
 .badge.run { color:var(--warning); background:var(--warning-soft); }
 .badge.bad { color:var(--danger); background:var(--danger-soft); }
 .badge.neutral { color:var(--muted); background:var(--surface-2); border:1px solid var(--line); }
 .dropzone {
   display:grid; place-items:center; min-height:180px; padding:24px; text-align:center;
   border:1.5px dashed var(--line-strong); border-radius:11px; background:var(--surface-2);
   cursor:pointer; transition:.15s;
 }
 .dropzone:hover,.dropzone.drag { border-color:var(--brand); background:var(--brand-soft); }
 .upload-icon { width:42px; height:42px; display:grid; place-items:center; margin:0 auto 10px; border-radius:10px; color:var(--brand); background:var(--brand-soft); }
 .drop-title { font-weight:600; }
 .drop-help { color:var(--muted); font-size:12px; margin-top:4px; }
 #f { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
 .file-row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:12px; padding:10px 12px; border:1px solid var(--line); border-radius:9px; }
 .file-name { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:550; }
 .file-size { color:var(--muted); font-size:12px; white-space:nowrap; }
 .upload-actions { display:flex; align-items:center; justify-content:flex-end; gap:10px; margin-top:12px; }
 .notice { border-radius:9px; padding:9px 11px; margin-top:12px; color:var(--muted); background:var(--surface-2); }
 .pipeline { display:grid; grid-template-columns:repeat(6,1fr); margin:18px 0 12px; }
 .pipe-step { position:relative; text-align:center; color:var(--subtle); font-size:11px; padding-top:24px; }
 .pipe-step::before {
   content:""; position:absolute; width:12px; height:12px; left:50%; top:0; transform:translateX(-50%);
   border-radius:50%; border:2px solid var(--line-strong); background:var(--surface);
 }
 .pipe-step:not(:last-child)::after { content:""; position:absolute; left:calc(50% + 8px); right:calc(-50% + 8px); top:6px; height:2px; background:var(--line); }
 .pipe-step.done { color:var(--success); }
 .pipe-step.done::before { border-color:var(--success); background:var(--success); }
 .pipe-step.done::after { background:var(--success); }
 .pipe-step.active { color:var(--warning); font-weight:600; }
 .pipe-step.active::before { border-color:#f79009; background:#f79009; box-shadow:0 0 0 4px var(--warning-soft); }
 .pipe-step.failed { color:var(--danger); }
 .pipe-step.failed::before { border-color:var(--danger); background:var(--danger); }
 .task-summary { color:var(--muted); font-size:13px; min-height:20px; }
 .data-card { margin-top:18px; }
 .tabs { display:flex; gap:2px; padding:0 22px; border-bottom:1px solid var(--line); }
 .tab { color:var(--muted); background:transparent; border-bottom:2px solid transparent; padding:14px 12px 11px; cursor:pointer; }
 .tab.active { color:var(--brand); border-color:var(--brand); font-weight:600; }
 .table-wrap { overflow-x:auto; }
 table { width:100%; border-collapse:collapse; font-size:13px; }
 th { color:var(--muted); font-weight:550; background:var(--surface-2); }
 th,td { padding:11px 14px; text-align:left; border-bottom:1px solid var(--line); vertical-align:middle; }
 tbody tr:last-child td { border-bottom:0; }
 .doc-name { max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:550; }
 .action-col { width:72px; min-width:72px; text-align:right; white-space:nowrap; }
 .delete-btn {
   color:var(--danger); background:transparent; border-radius:7px; padding:5px 8px;
   cursor:pointer; font-size:12px; font-weight:600; white-space:nowrap; min-width:40px;
 }
 .delete-btn:hover { background:var(--danger-soft); }
 .delete-btn:disabled { opacity:.5; cursor:wait; }
 .empty { padding:32px 14px; text-align:center; color:var(--subtle); }
 .panel[hidden] { display:none; }
 .mobile-only { display:none; }
 @media (prefers-color-scheme:dark) {
   :root {
     color-scheme:dark; --bg:#0f1115; --surface:#181b21; --surface-2:#20242c; --text:#f2f4f7;
     --muted:#a7afbd; --subtle:#7f8898; --line:#2b3039; --line-strong:#3a414d;
     --brand:#818cf8; --brand-hover:#6366f1; --brand-soft:#252849;
     --success:#6ce9a6; --success-soft:#173b2a; --warning:#fdb022; --warning-soft:#3b2d16;
     --danger:#f97066; --danger-soft:#431d1b; --shadow:none;
   }
 }
 @media (max-width:820px) {
   .shell { padding:20px 14px 40px; }
   .metrics { grid-template-columns:repeat(2,1fr); }
   .layout { grid-template-columns:1fr; }
   .composer { align-items:stretch; flex-direction:column; }
   .composer .btn { width:100%; }
 }
 @media (max-width:520px) {
   .topbar { align-items:center; }
   .brand-copy p,.live span { display:none; }
   .metrics { gap:8px; }
   .metric { padding:12px; }
   .metric-value { font-size:18px; }
   .card-head,.card-body { padding-left:16px; padding-right:16px; }
   .tabs { padding:0 8px; }
   th,td { padding:9px 10px; }
   .shortcut { display:none; }
 }
</style></head><body>
<main class="shell">
 <header class="topbar">
  <div class="brand">
   <div class="brand-mark">知</div>
   <div class="brand-copy"><h1>知屿</h1><p>本地知识库工作台</p></div>
  </div>
  <div class="live"><i id="live-dot" class="live-dot"></i><span id="live-text">正在连接服务</span></div>
 </header>

 <section class="metrics" aria-label="知识库概览">
  <div class="metric"><div class="metric-label">运行状态</div><div class="metric-value" id="m-state">连接中</div><div class="metric-detail" id="m-sync">等待首次同步</div></div>
  <div class="metric"><div class="metric-label">知识文档</div><div class="metric-value" id="m-docs">—</div><div class="metric-detail">当前生效版本</div></div>
  <div class="metric"><div class="metric-label">检索分块</div><div class="metric-value" id="m-chunks">—</div><div class="metric-detail" id="m-embed">读取模型中</div></div>
  <div class="metric"><div class="metric-label">待处理任务</div><div class="metric-value" id="m-jobs">—</div><div class="metric-detail" id="m-answer">读取回答模式中</div></div>
 </section>

 <div class="layout">
  <div class="stack">
   <section class="card" aria-labelledby="ask-title">
    <div class="card-head">
     <div><h2 class="card-title" id="ask-title">向知识库提问</h2><p class="card-subtitle">答案仅依据已索引文档，并附带可核验来源</p></div>
     <span class="badge neutral" id="ask-mode">本地检索</span>
    </div>
    <div class="card-body">
     <div class="composer">
      <label style="flex:1"><span style="position:absolute;width:1px;height:1px;overflow:hidden">问题</span>
       <textarea id="q" placeholder="例如：这份文档的核心结论是什么？"></textarea>
      </label>
      <button class="btn btn-primary" id="askbtn" onclick="ask()">开始检索</button>
     </div>
     <div class="shortcut"><span>Enter 换行，⌘/Ctrl + Enter 提交</span><span id="question-count">0 字</span></div>
     <div class="trace" id="atrace" hidden></div>
     <div class="answer" id="answer" hidden>
      <div class="answer-meta"><span class="badge ok" id="answer-status">已完成</span><span class="card-subtitle" id="confidence"></span></div>
      <div class="answer-text" id="aout"></div>
      <div class="sources" id="cites"></div>
     </div>
    </div>
   </section>

   <section class="card data-card" aria-labelledby="library-title">
    <div class="card-head"><div><h2 class="card-title" id="library-title">知识库内容</h2><p class="card-subtitle">查看文档索引状态与最近任务</p></div></div>
    <div class="tabs" role="tablist">
     <button class="tab active" id="tab-docs" onclick="showPanel('docs')" role="tab">文档</button>
     <button class="tab" id="tab-jobs" onclick="showPanel('jobs')" role="tab">任务记录</button>
    </div>
    <div class="panel table-wrap" id="panel-docs">
     <table><thead><tr><th>文档名称</th><th>阶段</th><th>状态</th><th>分块</th><th class="action-col">操作</th></tr></thead><tbody id="docs"></tbody></table>
    </div>
    <div class="panel table-wrap" id="panel-jobs" hidden>
     <table><thead><tr><th>任务</th><th>状态</th><th>说明</th></tr></thead><tbody id="jobs"></tbody></table>
    </div>
   </section>
  </div>

  <aside class="stack">
   <section class="card" aria-labelledby="upload-title">
    <div class="card-head"><div><h2 class="card-title" id="upload-title">导入资料</h2><p class="card-subtitle">上传后自动解析、切分并建立索引</p></div></div>
    <div class="card-body">
     <label class="dropzone" id="dropzone" for="f">
      <span>
       <span class="upload-icon" aria-hidden="true">
        <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/><path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/></svg>
       </span>
       <span class="drop-title">拖入文件，或点击选择</span>
       <span class="drop-help">支持项目已配置的文档格式</span>
      </span>
     </label>
     <input type="file" id="f" onchange="selectFile(this.files[0])">
     <div class="file-row" id="file-row" hidden><span class="file-name" id="file-name"></span><span class="file-size" id="file-size"></span></div>
     <div class="upload-actions"><button class="btn btn-primary" id="upbtn" onclick="up()" disabled>上传并索引</button></div>
     <div class="notice" id="uout" hidden></div>
    </div>
   </section>

   <section class="card" aria-labelledby="process-title">
    <div class="card-head">
     <div><h2 class="card-title" id="process-title">执行过程</h2><p class="card-subtitle">页面会自动同步后台任务</p></div>
     <span class="badge neutral" id="process-badge">空闲</span>
    </div>
    <div class="card-body">
     <div class="pipeline" id="steps"></div>
     <p class="task-summary" id="taskline">当前没有进行中的任务</p>
    </div>
   </section>
  </aside>
 </div>
</main>
<script>
const STAGES = [
 ["parse","解析"],["clean","清洗"],["chunk","切分"],
 ["embed","嵌入"],["index","索引"],["done","完成"]
];
const STATUS = {pending:"排队",running:"进行中",done:"已完成",failed:"等待重试",dead:"失败",empty:"无内容",queued:"排队"};
let selectedFile = null;
let lastStatus = null;
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function badge(status,label){
 const cls=status==="done"?"ok":(status==="pending"||status==="running"||status==="queued"||status==="failed"?"run":(status==="dead"?"bad":"neutral"));
 return '<span class="badge '+cls+'">'+esc(label||STATUS[status]||status)+'</span>';
}
function cleanTaskName(name){return String(name||"").replace(/^\\d+_/,"");}
function stageLabel(stage){const row=STAGES.find(s=>s[0]===stage);return row?row[1]:(stage||"—");}
function formatSize(bytes){
 if(bytes<1024)return bytes+" B";
 if(bytes<1024*1024)return(bytes/1024).toFixed(1)+" KB";
 return(bytes/1024/1024).toFixed(1)+" MB";
}
function selectFile(file){
 selectedFile=file||null;
 document.getElementById("file-row").hidden=!selectedFile;
 document.getElementById("upbtn").disabled=!selectedFile;
 if(selectedFile){
  document.getElementById("file-name").textContent=selectedFile.name;
  document.getElementById("file-size").textContent=formatSize(selectedFile.size);
 }
}
function renderPipeline(active,failed){
 const idx=STAGES.findIndex(s=>s[0]===active);
 document.getElementById("steps").innerHTML=STAGES.map((s,i)=>{
  let cls="pipe-step";
  if(failed&&s[0]===active)cls+=" failed";
  else if(active==="done"||(idx>=0&&i<idx))cls+=" done";
  else if(i===idx)cls+=" active";
  return '<span class="'+cls+'">'+s[1]+'</span>';
 }).join("");
}
function showPanel(name){
 for(const p of ["docs","jobs"]){
  document.getElementById("panel-"+p).hidden=p!==name;
  document.getElementById("tab-"+p).classList.toggle("active",p===name);
 }
}
async function refresh(){
 try{
  const r=await fetch("/status",{cache:"no-store"}); if(!r.ok)throw new Error(String(r.status));
  const d=await r.json(); lastStatus=d; const h=d.health;
  document.getElementById("live-dot").classList.toggle("busy",d.busy);
  document.getElementById("live-text").textContent=d.busy?"后台任务执行中":"服务在线";
  document.getElementById("m-state").textContent=d.busy?"处理中":"运行正常";
  document.getElementById("m-sync").textContent="更新于 "+new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"});
  document.getElementById("m-docs").textContent=Number(h.latest_docs).toLocaleString();
  document.getElementById("m-chunks").textContent=Number(h.chunks).toLocaleString();
  document.getElementById("m-jobs").textContent=Number(h.jobs_pending).toLocaleString();
  document.getElementById("m-embed").textContent=d.runtime.embedder;
  document.getElementById("m-answer").textContent="回答："+d.runtime.answerer;
  document.getElementById("ask-mode").textContent=d.runtime.answerer==="摘录"?"检索摘录":"模型回答";
  const jobs=(d.jobs||[]).slice().reverse();
  document.getElementById("jobs").innerHTML=jobs.map(j=>
   '<tr><td><div class="doc-name">'+esc(cleanTaskName(j.path))+'</div></td><td>'+badge(j.status)+'</td><td>'+esc(j.error||"—")+'</td></tr>'
  ).join("")||'<tr><td colspan="3" class="empty">还没有任务记录</td></tr>';
  document.getElementById("docs").innerHTML=(d.docs||[]).map(doc=>
   '<tr><td><div class="doc-name" title="'+esc(doc.doc_name)+'">'+esc(cleanTaskName(doc.doc_name))+'</div></td><td>'+esc(stageLabel(doc.stage))+'</td><td>'+badge(doc.status)+'</td><td>'+Number(doc.chunks).toLocaleString()+'</td><td class="action-col"><button class="delete-btn" data-hash="'+esc(doc.content_hash)+'" data-name="'+esc(cleanTaskName(doc.doc_name))+'" onclick="deleteDocument(this)">删除</button></td></tr>'
  ).join("")||'<tr><td colspan="5" class="empty">还没有文档，先从右侧导入资料</td></tr>';
  const latest=jobs[0];
  const failed=latest&&(latest.status==="failed"||latest.status==="dead");
  const processBadge=document.getElementById("process-badge");
  if(d.busy){
   renderPipeline(d.active_stage||"parse",false);
   processBadge.className="badge run"; processBadge.textContent="执行中";
   document.getElementById("taskline").textContent="正在处理「"+cleanTaskName(latest?.path||"文档")+"」 · "+stageLabel(d.active_stage||"parse");
  }else if(latest&&latest.status==="done"){
   renderPipeline("done",false);
   processBadge.className="badge ok"; processBadge.textContent="已完成";
   document.getElementById("taskline").textContent="最近完成「"+cleanTaskName(latest.path)+"」";
  }else if(failed){
   renderPipeline(d.active_stage||"parse",true);
   processBadge.className="badge bad"; processBadge.textContent="失败";
   document.getElementById("taskline").textContent=latest.error||"任务执行失败";
  }else{
   renderPipeline("",false);
   processBadge.className="badge neutral"; processBadge.textContent="空闲";
   document.getElementById("taskline").textContent="当前没有进行中的任务";
  }
 }catch(e){
  document.getElementById("live-dot").classList.add("busy");
  document.getElementById("live-text").textContent="服务连接失败";
  document.getElementById("m-state").textContent="连接中断";
  document.getElementById("m-sync").textContent="正在自动重试";
 }
}
async function deleteDocument(button){
 const contentHash=button.dataset.hash,name=button.dataset.name;
 if(!confirm("确定删除「"+name+"」？\\n\\n该文档的分块、向量和全文索引会一起删除，此操作不可撤销。"))return;
 const original=button.textContent;button.disabled=true;button.textContent="删除中";
 try{
  const r=await fetch("/documents/"+encodeURIComponent(contentHash),{method:"DELETE"});
  const d=await r.json();
  if(!r.ok||!d.ok)throw new Error(d.error||("HTTP "+r.status));
  await refresh();
 }catch(e){
  alert("删除失败："+e.message);
  button.disabled=false;button.textContent=original;
 }
}
async function up(){
 if(!selectedFile)return;
 const btn=document.getElementById("upbtn"),notice=document.getElementById("uout");
 btn.disabled=true; btn.textContent="正在上传…";
 notice.hidden=false; notice.textContent="正在提交「"+selectedFile.name+"」";
 try{
  const fd=new FormData();fd.append("file",selectedFile);
  const r=await fetch("/ingest",{method:"POST",body:fd});
  if(!r.ok)throw new Error("HTTP "+r.status);
  const d=await r.json();
  notice.textContent="任务 #"+d.job_id+" 已进入队列，可在执行过程查看进度。";
  showPanel("jobs"); await refresh();
 }catch(e){notice.textContent="上传失败："+e.message;}
 finally{btn.disabled=false;btn.textContent="上传并索引";}
}
async function ask(){
 const input=document.getElementById("q"),q=input.value.trim();if(!q)return;
 const btn=document.getElementById("askbtn"),trace=document.getElementById("atrace");
 const answer=document.getElementById("answer");
 btn.disabled=true;btn.textContent="检索中…";answer.hidden=true;trace.hidden=false;
 trace.innerHTML='<span class="trace-item running">向量与关键词检索</span><span class="trace-item">置信度判断</span><span class="trace-item">组织回答</span>';
 try{
  const fd=new FormData();fd.append("question",q);
  const r=await fetch("/ask",{method:"POST",body:fd});
  if(!r.ok){
    let msg="HTTP "+r.status;
    try{const e=await r.json(); if(typeof e.detail==="string") msg=e.detail;}catch(_){}
    throw new Error(msg);
  }
  const d=await r.json();
  trace.innerHTML=(d.trace||[]).map(t=>'<span class="trace-item '+(t.status==="done"?"":"bad")+'">'+esc(t.step)+' · '+t.ms+'ms · '+esc(t.detail||"")+'</span>').join("");
  document.getElementById("aout").textContent=d.text;
  document.getElementById("confidence").textContent="置信度 "+d.confidence;
  const status=document.getElementById("answer-status");
  status.className="badge "+(d.refused?"bad":"ok");status.textContent=d.refused?"依据不足":"回答完成";
  document.getElementById("cites").innerHTML=(d.citations||[]).map((c,i)=>
   '<details class="source"><summary>来源 '+(i+1)+' · '+esc(cleanTaskName(c.doc_name))+' / '+esc(c.section_path)+(c.page!=null?' / 第'+c.page+'页':'')+'</summary><div class="source-text">'+esc(c.text)+'</div></details>'
  ).join("");
  answer.hidden=false;
 }catch(e){
  trace.innerHTML='<span class="trace-item bad">请求失败</span>';
  document.getElementById("aout").textContent="提问失败："+e.message;
  document.getElementById("answer-status").className="badge bad";
  document.getElementById("answer-status").textContent="错误";
  document.getElementById("confidence").textContent="";
  answer.hidden=false;
 }finally{btn.disabled=false;btn.textContent="开始检索";}
}
const q=document.getElementById("q");
q.addEventListener("input",()=>document.getElementById("question-count").textContent=q.value.length+" 字");
q.addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)){e.preventDefault();ask();}});
const drop=document.getElementById("dropzone");
for(const event of ["dragenter","dragover"])drop.addEventListener(event,e=>{e.preventDefault();drop.classList.add("drag");});
for(const event of ["dragleave","drop"])drop.addEventListener(event,e=>{e.preventDefault();drop.classList.remove("drag");});
drop.addEventListener("drop",e=>selectFile(e.dataTransfer.files[0]));
renderPipeline("",false);refresh();setInterval(refresh,1500);
</script>
</body></html>"""
