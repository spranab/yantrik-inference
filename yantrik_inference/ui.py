"""The page served at /. Two tabs over one model: typed decisions, and chat."""

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>yantrik-inference</title>
<style>
:root{--bg:#fbfaf9;--fg:#1c1b19;--mut:#6b6864;--line:#e2ded8;--card:#fff;--ok:#1a7f5a;--warn:#b4690e;--bad:#b3261e;--accent:#2f5d8a}
@media(prefers-color-scheme:dark){:root{--bg:#17161a;--fg:#eceae6;--mut:#9b978f;--line:#302e33;--card:#1f1e23;--ok:#5fd0a0;--warn:#e0a34a;--bad:#f2857c;--accent:#7fb2e0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:26px 20px 70px}
h1{font-size:21px;margin:0 0 2px;font-weight:620}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:20px}
.tab{padding:9px 16px;cursor:pointer;font-size:14px;font-weight:550;color:var(--mut);border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.on{color:var(--fg);border-bottom-color:var(--accent)}
.pane{display:none}.pane.on{display:block}
label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 6px}
textarea{width:100%;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:11px;font:13px/1.55 ui-monospace,Consolas,monospace;resize:vertical}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
#rec,#qs{height:180px}#msg{height:78px}
.row{display:flex;gap:10px;align-items:center;margin:16px 0 4px;flex-wrap:wrap}
button{background:var(--accent);color:#fff;border:0;border-radius:7px;padding:9px 16px;font-size:14px;font-weight:550;cursor:pointer}
button.alt{background:transparent;color:var(--fg);border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:default}
.note{color:var(--mut);font-size:12px}
.timing{display:flex;gap:26px;margin:18px 0 4px;flex-wrap:wrap}
.t{border-left:3px solid var(--line);padding-left:12px}
.t b{display:block;font-size:23px;font-weight:620;font-variant-numeric:tabular-nums}
.t.win{border-left-color:var(--ok)}.t span{font-size:12px;color:var(--mut)}
table{width:100%;border-collapse:collapse;margin-top:6px;font-size:14px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);font-weight:600;padding:7px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--line)}
td.a{font-weight:620;white-space:nowrap}
.bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden;min-width:70px}
.bar i{display:block;height:100%}
.c{font-variant-numeric:tabular-nums;color:var(--mut);font-size:12px}
.diff{color:var(--bad);font-weight:600}
.err{color:var(--bad);white-space:pre-wrap;font:12px ui-monospace,monospace;margin-top:10px}
#log{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;min-height:150px;white-space:pre-wrap;font:14px/1.6 system-ui;margin-bottom:12px;overflow-x:auto}
.you{color:var(--mut);margin-top:12px}
.api{margin-top:34px;padding-top:16px;border-top:1px solid var(--line);color:var(--mut);font-size:13px}
.api code{font:12px ui-monospace,Consolas,monospace;background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 5px}
</style></head><body>
<div class="wrap">
<h1>yantrik-inference</h1>
<div class="sub" id="sub">loading…</div>
<div class="tabs">
  <div class="tab on" data-p="d">Decide — typed fields</div>
  <div class="tab" data-p="c">Chat — generation</div>
</div>

<div class="pane on" id="p-d">
  <div class="grid">
    <div><label for="rec">The record</label><textarea id="rec"></textarea></div>
    <div><label for="qs">Questions — <code>question | opt/opt</code> per line</label><textarea id="qs"></textarea></div>
  </div>
  <div class="row">
    <button id="go">Read all fields at once</button>
    <button id="gojson" class="alt">Compare: let it write JSON</button>
    <span class="note" id="note"></span>
  </div>
  <div class="timing" id="timing"></div>
  <div id="out"></div>
</div>

<div class="pane" id="p-c">
  <div id="log"></div>
  <label for="msg">Message — ctrl+enter to send</label>
  <textarea id="msg" placeholder="Ask it to write, explain, or code…"></textarea>
  <div class="row">
    <button id="send">Send</button>
    <button id="clear" class="alt">Clear</button>
    <span class="note" id="cnote"></span>
  </div>
  <div class="timing" id="ctiming"></div>
</div>

<div class="api">
  OpenAI-compatible: <code>POST /v1/chat/completions</code> (<code>stream:true</code> supported),
  <code>GET /v1/models</code>. Typed decisions:
  <code>POST /v1/decide</code> with <code>{"record":…,"questions":["q | a/b", …]}</code>.
</div>
</div>
<script>
const $=s=>document.querySelector(s);
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+t.dataset.p));
});
let MODEL='';
fetch('/api/info').then(r=>r.json()).then(d=>{
  MODEL=d.model; $('#rec').value=d.sample_record;
  $('#qs').value=d.sample_questions.map(q=>q.q+' | '+q.opts.join('/')).join('\n');
  $('#sub').textContent=`${d.model} — ${d.where}, loaded in ${d.load_s}s · decide: up to `
    +`${d.max_fields} fields at ${d.per_seq} tokens · chat: ${d.chat_ctx} tokens · cache ${d.kv}`;
});
function parseQs(){
  return $('#qs').value.split('\n').map(l=>l.trim()).filter(Boolean).map(l=>{
    const i=l.lastIndexOf('|');
    if(i<0) return {q:l,opts:['yes','no']};
    return {q:l.slice(0,i).trim(), opts:l.slice(i+1).split('/').map(s=>s.trim()).filter(Boolean)};
  }).filter(x=>x.opts.length>1);
}
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let last={},rows=[];
function timing(){
  const t=[];
  if(last.parallel!==undefined) t.push(['One pass, all fields',last.parallel.toFixed(2)+'s',true]);
  if(last.json!==undefined) t.push(['Writing it as JSON',last.json.toFixed(2)+'s',false]);
  let h=t.map(([k,v,w])=>`<div class="t ${w?'win':''}"><b>${v}</b><span>${k}</span></div>`).join('');
  if(t.length===2) h+=`<div class="t"><b>${(last.json/last.parallel).toFixed(1)}×</b><span>cheaper</span></div>`;
  $('#timing').innerHTML=h;
}
function render(){
  $('#out').innerHTML='<table><tr><th>Question</th><th>Answer</th><th>Confidence</th>'
    +(rows.some(r=>r.json!==undefined)?'<th>JSON said</th>':'')+'</tr>'
    +rows.map(r=>{const c=r.conf,col=c>=.85?'var(--ok)':c>=.6?'var(--warn)':'var(--bad)';
      const j=r.json===undefined?'':`<td class="${r.json!==r.answer?'diff':''}">${esc(r.json??'—')}</td>`;
      return `<tr><td>${esc(r.q)}</td><td class="a">${esc(r.answer)}</td><td><div class="bar">`
        +`<i style="width:${(c*100).toFixed(0)}%;background:${col}"></i></div>`
        +`<span class="c">${(c*100).toFixed(0)}%</span></td>${j}</tr>`;}).join('')+'</table>';
}
async function run(mode){
  const b=[$('#go'),$('#gojson')]; b.forEach(x=>x.disabled=true); $('#note').textContent='running…';
  try{
    const url=mode==='read'?'/v1/decide':'/api/decide_json';
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({record:$('#rec').value,questions:parseQs()})});
    const d=await r.json();
    if(d.error&&mode!=='json'){$('#out').innerHTML='<div class="err">'+esc(d.error)+'</div>';$('#note').textContent='';}
    else if(d.error){$('#note').textContent=d.error;last.json=undefined;timing();}
    else{
      if(mode==='read'){last.parallel=d.seconds;rows=d.answers.map(a=>({q:a.question,answer:a.answer,conf:a.confidence}));}
      else{last.json=d.seconds;
        rows=rows.length?rows.map((x,i)=>({...x,json:d.answers[i]}))
                        :d.answers.map((a,i)=>({q:parseQs()[i].q,answer:a,conf:1}));}
      render(); timing(); $('#note').textContent='';
    }
  }catch(e){$('#note').textContent='';$('#out').innerHTML='<div class="err">'+esc(e)+'</div>';}
  b.forEach(x=>x.disabled=false);
}
$('#go').onclick=()=>run('read'); $('#gojson').onclick=()=>run('json');

let hist=[];
$('#clear').onclick=()=>{hist=[];$('#log').textContent='';$('#ctiming').innerHTML='';};
$('#send').onclick=async()=>{
  const m=$('#msg').value.trim(); if(!m) return;
  $('#msg').value=''; $('#send').disabled=true; $('#cnote').textContent='generating…';
  hist.push({role:'user',content:m});
  const d=document.createElement('div'); d.className='you'; d.textContent='you: '+m;
  $('#log').appendChild(d);
  const cur=document.createElement('div'); $('#log').appendChild(cur);
  const t0=performance.now(); let first=null,n=0,txt='';
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:MODEL,messages:hist,stream:true,max_tokens:512})});
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){ const {value,done}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let i; while((i=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,i).trim(); buf=buf.slice(i+1);
        if(!line.startsWith('data:')) continue;
        const p=line.slice(5).trim(); if(p==='[DONE]') continue;
        const c=JSON.parse(p).choices[0].delta.content||''; if(!c) continue;
        if(first===null) first=(performance.now()-t0)/1000;
        n++; txt+=c; cur.textContent=txt; $('#log').scrollTop=$('#log').scrollHeight;
      }
    }
    hist.push({role:'assistant',content:txt});
    const tot=(performance.now()-t0)/1000;
    if(first!==null) $('#ctiming').innerHTML=
       `<div class="t"><b>${first.toFixed(2)}s</b><span>first token</span></div>`
      +`<div class="t"><b>${(n/Math.max(tot-first,.001)).toFixed(1)}</b><span>tokens/s</span></div>`
      +`<div class="t"><b>${tot.toFixed(1)}s</b><span>total, ${n} tokens</span></div>`;
  }catch(e){ cur.innerHTML='<div class="err">'+esc(e)+'</div>'; }
  $('#cnote').textContent=''; $('#send').disabled=false;
};
$('#msg').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))$('#send').click();});
</script>
</body></html>
"""
