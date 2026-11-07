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
function cleanTaskName(name){return String(name||"").replace(/^\d+_/,"");}
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
 if(!confirm("确定删除「"+name+"」？\n\n该文档的分块、向量和全文索引会一起删除，此操作不可撤销。"))return;
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
