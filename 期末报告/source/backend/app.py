"""
FastAPI 后端服务
提供 REST API 接口 + Web 前端页面
"""
import sys
import os
import json

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import yaml

from utils.path_utils import get_config_path
from qa_engine import get_engine


app = FastAPI(
    title="PyTorch 智能报错诊断问答系统",
    description="基于知识图谱的 PyTorch 错误诊断 API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求/响应模型 ----
class AskRequest(BaseModel):
    query: str
    use_llm: bool = False


class AskResponse(BaseModel):
    answer: str
    intent: str
    matched_entity: str
    source: str
    confidence: float
    debug: dict = {}


class BuildKGResponse(BaseModel):
    status: str
    message: str


# ---- 前端 HTML 页面 ----
FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyTorch 报错诊断问答系统</title>
<script>
// vis.js loader with fallback CDNs
(function(){
  var urls = [
    'https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js',
    'https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js',
    'https://cdn.jsdelivr.net/npm/vis-network@9.1.2/standalone/umd/vis-network.min.js'
  ];
  var i=0;
  function tryLoad(){
    if(i>=urls.length){ console.warn('vis.js failed to load from all CDNs'); return; }
    var s=document.createElement('script');
    s.src=urls[i]; s.onerror=function(){ i++; tryLoad(); };
    document.head.appendChild(s);
  }
  tryLoad();
})();
</script>
<style>
:root {
  --bg: #f8f9fa; --card: #fff; --text: #2c3e50; --sub: #6b7280;
  --border: #e5e7eb; --primary: #ee4c2c; --accent: #f97316;
  --radius: 12px; --shadow: 0 1px 3px rgba(0,0,0,.08);
  --green: #10b981; --blue: #3b82f6; --purple: #8b5cf6; --red: #ef4444;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }

/* Navbar */
.navbar { background: #fff; border-bottom: 1px solid var(--border); padding: 0 24px; height: 56px; display: flex; align-items: center; position: sticky; top: 0; z-index: 100; }
.navbar .logo { display: flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 700; color: var(--primary); }
.navbar .logo .icon { width: 32px; height: 32px; background: linear-gradient(135deg, var(--primary), var(--accent)); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 16px; }
.navbar .stats { margin-left: auto; display: flex; gap: 16px; font-size: 13px; color: var(--sub); }
.navbar .stats .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot-on { background: var(--green); } .dot-off { background: #d1d5db; }

/* Nav links (tab replacement) */
.nav-links { display: flex; gap: 4px; margin-left: 32px; }
.nav-link { padding: 8px 20px; border-radius: 6px; font-size: 14px; font-weight: 600; color: #4b5563; text-decoration: none; cursor: pointer; transition: all .15s; }
.nav-link:hover { background: #f3f4f6; color: #1f2937; }
.nav-link.active { background: var(--primary); color: #fff; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.tab-icon { margin-right: 6px; }

/* Main layout */
.main { max-width: 1200px; margin: 0 auto; padding: 20px; }

/* Hero */
.hero { text-align: center; padding: 10px 0 20px; }
.hero h2 { font-size: 20px; font-weight: 700; }
.hero p { font-size: 13px; color: var(--sub); margin-top: 4px; }

/* Cards */
.card { background: var(--card); border-radius: var(--radius); border: 1px solid var(--border); padding: 20px; margin-bottom: 14px; box-shadow: var(--shadow); }
.card-title { font-size: 12px; font-weight: 600; color: var(--sub); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; }
textarea#query { width: 100%; min-height: 72px; border: 2px solid var(--border); border-radius: 10px; padding: 14px 16px; font-size: 15px; font-family: inherit; resize: vertical; outline: none; transition: border-color .2s; background: #fafbfc; }
textarea#query:focus { border-color: var(--primary); background: #fff; }

/* Quick tags */
.tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.tag { display: inline-flex; align-items: center; gap: 4px; padding: 5px 14px; border-radius: 20px; font-size: 13px; cursor: pointer; border: 1px solid var(--border); background: #f9fafb; color: #4b5563; transition: all .15s; user-select: none; }
.tag:hover { border-color: var(--primary); color: var(--primary); background: #fef2f2; }

/* Buttons */
.actions { display: flex; align-items: center; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 10px 22px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; font-family: inherit; transition: all .15s; }
.btn-go { background: var(--primary); color: #fff; }
.btn-go:hover { background: #d43d1f; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(238,76,44,.35); }
.btn-go:disabled { opacity: .6; pointer-events: none; }
.btn-outline { background: #fff; color: #4b5563; border: 1.5px solid var(--border); }
.btn-outline:hover { border-color: #9ca3af; background: #f9fafb; }
.toggle-label { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #6b7280; margin-left: auto; cursor: pointer; user-select: none; }
.toggle-label input[type=checkbox] { width: 38px; height: 22px; appearance: none; background: #d1d5db; border-radius: 11px; position: relative; cursor: pointer; }
.toggle-label input[type=checkbox]:checked { background: var(--primary); }
.toggle-label input[type=checkbox]::after { content: ''; position: absolute; width: 18px; height: 18px; background: #fff; border-radius: 50%; top: 2px; left: 2px; transition: transform .2s; }
.toggle-label input[type=checkbox]:checked::after { transform: translateX(16px); }

/* Result */
.result-card { background: var(--card); border-radius: var(--radius); border: 1px solid var(--border); box-shadow: var(--shadow); overflow: hidden; }
.result-header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.badge-kg { background: #d1fae5; color: #065f46; }
.badge-faq { background: #dbeafe; color: #1e40af; }
.badge-llm { background: #ede9fe; color: #5b21b6; }
.badge-fallback { background: #fee2e2; color: #991b1b; }
.meta { font-size: 12px; color: var(--sub); display: flex; gap: 12px; flex-wrap: wrap; }
.result-body { padding: 24px; }
.result-body pre { white-space: pre-wrap; word-wrap: break-word; font-family: 'SF Mono', 'Consolas', monospace; font-size: 13.5px; line-height: 1.7; color: #374151; margin: 0; }

/* Debug */
.debug-bar { margin-top: 10px; padding: 10px 16px; background: #f3f4f6; border-radius: 8px; font-size: 11.5px; color: #9ca3af; font-family: 'SF Mono', 'Consolas', monospace; display: flex; gap: 16px; flex-wrap: wrap; }

/* Spinner */
.spinner-wrap { display: none; justify-content: center; padding: 40px; }
.spinner-wrap.active { display: flex; }
.spinner { width: 32px; height: 32px; border: 3px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Toast */
.toast { position: fixed; top: 20px; right: 20px; padding: 12px 20px; border-radius: 8px; color: #fff; font-size: 14px; z-index: 999; opacity: 0; transform: translateY(-10px); transition: all .3s; pointer-events: none; }
.toast.show { opacity: 1; transform: translateY(0); }
.toast-ok { background: var(--green); }
.toast-err { background: var(--red); }

/* Graph panel */
#graphContainer { width: 100%; height: 650px; border: 1px solid var(--border); border-radius: var(--radius); background: #fff; overflow: hidden; }
.graph-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 10px 0; font-size: 12px; }
.graph-legend span { display: flex; align-items: center; gap: 4px; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.graph-info { padding: 8px 16px; background: #f0fdf4; border-radius: 8px; font-size: 13px; color: #065f46; margin-top: 8px; display: none; }
.graph-info.show { display: block; }

/* Footer */
.footer { text-align: center; padding: 20px; font-size: 12px; color: #d1d5db; }
</style>
</head>
<body>

<nav class="navbar">
  <div class="logo">
    <div class="icon">&lt;/&gt;</div>
    <span>PyTorch 报错诊断</span>
  </div>
  <div class="nav-links">
    <span class="nav-link active" onclick="switchTab('diagnosis',this)" id="nav-diag">智能诊断</span>
    <span class="nav-link" onclick="switchTab('graph',this)" id="nav-graph">知识图谱</span>
  </div>
  <div class="stats" id="statsBar">
    <span><span class="dot dot-on" id="llmDot"></span> <span id="llmStatus">...</span></span>
    <span id="kgInfo">...</span>
  </div>
</nav>

<!-- Diagnosis Tab -->
<div class="tab-content active" id="tab-diagnosis">
<div class="main">
  <div class="hero">
    <h2>深度学习报错诊断助手</h2>
    <p>输入 PyTorch 报错或问题，智能匹配知识图谱，给出结构化诊断方案</p>
  </div>
  <div class="card">
    <div class="card-title">请输入你的问题</div>
    <textarea id="query" placeholder="例如：torch.cuda.is_available() 返回 False 怎么办？&#10;或者直接粘贴完整的报错日志..."></textarea>
    <div class="tags">
      <span class="tag" data-q="torch.cuda.is_available() 返回 False 怎么办？">CUDA 不可用</span>
      <span class="tag" data-q="CUDA out of memory 怎么解决？">CUDA OOM</span>
      <span class="tag" data-q="DataLoader 的 num_workers 是什么意思？">DataLoader 参数</span>
      <span class="tag" data-q="PyTorch 怎么保存和加载模型？">模型保存加载</span>
      <span class="tag" data-q="state_dict 不匹配怎么办？">state_dict 不匹配</span>
      <span class="tag" data-q="CrossEntropyLoss 输入维度应该是什么？">CrossEntropyLoss</span>
      <span class="tag" data-q="torch.load 的 weights_only=True 是什么意思？">torch.load 安全</span>
    </div>
    <div class="actions">
      <button class="btn btn-go" id="askBtn" onclick="doAsk()">开始诊断</button>
      <button class="btn btn-outline" onclick="doBuildKG()">重建知识图谱</button>
      <label class="toggle-label">
        <input type="checkbox" id="useLLM" checked>
        <span>DeepSeek V4 润色</span>
      </label>
    </div>
  </div>
  <div class="result-card">
    <div class="result-header" id="resultHeader" style="display:none"></div>
    <div class="result-body">
      <div class="spinner-wrap" id="spinner"><div class="spinner"></div></div>
      <pre id="result"></pre>
    </div>
  </div>
  <div class="debug-bar" id="debug"></div>
</div>
</div>

<!-- Knowledge Graph Tab -->
<div class="tab-content" id="tab-graph">
<div class="main">
  <div class="card" style="padding:14px">
    <div class="card-title">PyTorch 错误诊断知识图谱</div>
    <div class="graph-legend" id="legend">
      <span><span class="legend-dot" style="background:#ee4c2c"></span> Problem(问题)</span>
      <span><span class="legend-dot" style="background:#ef4444"></span> Error(错误)</span>
      <span><span class="legend-dot" style="background:#3b82f6"></span> API</span>
      <span><span class="legend-dot" style="background:#10b981"></span> Concept(概念)</span>
      <span><span class="legend-dot" style="background:#8b5cf6"></span> Solution(解决)</span>
      <span><span class="legend-dot" style="background:#f97316"></span> Cause(原因)</span>
      <span><span class="legend-dot" style="background:#06b6d4"></span> Command(命令)</span>
      <span><span class="legend-dot" style="background:#6366f1"></span> DocPage</span>
      <span><span class="legend-dot" style="background:#ec4899"></span> CodeExample</span>
    </div>
    <div id="graphContainer"></div>
    <div class="graph-info" id="graphInfo"></div>
  </div>
</div>
</div>

<div class="footer">Powered by PyTorch KG + DeepSeek V4</div>

<div class="toast" id="toast"></div>

<script>
// ===== Tab Switching =====
function switchTab(name, el) {
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (el) el.classList.add('active');
  else document.getElementById('nav-' + name).classList.add('active');
  if (name === 'graph') initGraph();
}

// ===== Diagnosis =====
const API = ''; let isLoading = false;

document.querySelectorAll('.tag[data-q]').forEach(el => {
  el.addEventListener('click', () => { document.getElementById('query').value = el.dataset.q; doAsk(); });
});
document.getElementById('query').addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doAsk(); });

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'toast toast-' + type + ' show';
  setTimeout(() => t.classList.remove('show'), 2500);
}

function badgeClass(src) {
  return {'RAG':'badge-kg','RAG+LLM':'badge-llm',KG:'badge-kg','FAQ->KG':'badge-faq','FAQ':'badge-faq','LLM':'badge-llm','FALLBACK':'badge-fallback'}[src] || 'badge-kg';
}

async function doAsk() {
  const q = document.getElementById('query').value.trim();
  if (!q) { showToast('Please enter a question', 'err'); return; }
  if (isLoading) return;
  isLoading = true;
  document.getElementById('askBtn').disabled = true;
  document.getElementById('spinner').classList.add('active');
  document.getElementById('result').textContent = '';
  document.getElementById('resultHeader').style.display = 'none';
  try {
    const resp = await fetch(API + '/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q, use_llm: document.getElementById('useLLM').checked})
    });
    const d = await resp.json();
    document.getElementById('resultHeader').style.display = 'flex';
    document.getElementById('resultHeader').innerHTML =
      '<span class="badge ' + badgeClass(d.source) + '">' + d.source + '</span>' +
      '<div class="meta"><span>Intent: ' + d.intent + '</span><span>Entity: ' + (d.matched_entity || '-') + '</span><span>Confidence: ' + (d.confidence * 100).toFixed(0) + '%</span></div>';
    document.getElementById('result').textContent = d.answer;
    document.getElementById('debug').textContent = 'source=' + d.source + ' | intent=' + d.intent + ' | entity=' + d.matched_entity + ' | confidence=' + d.confidence.toFixed(4) + (d.debug && d.debug.llm_polished ? ' | LLM polished' : '');
  } catch(e) {
    document.getElementById('result').textContent = 'Request failed: ' + e.message;
    showToast('Request failed: ' + e.message, 'err');
  }
  isLoading = false;
  document.getElementById('askBtn').disabled = false;
  document.getElementById('spinner').classList.remove('active');
}

async function doBuildKG() {
  document.getElementById('result').textContent = 'Building knowledge graph...';
  try {
    const resp = await fetch(API + '/build_kg', {method: 'POST'});
    const d = await resp.json();
    document.getElementById('result').textContent = d.status + ': ' + d.message;
    showToast(d.status === 'success' ? 'Build OK' : 'Build failed', d.status === 'success' ? 'ok' : 'err');
    loadStats();
  } catch(e) { document.getElementById('result').textContent = 'Build failed: ' + e.message; }
}

async function loadStats() {
  try {
    const r = await fetch(API + '/stats'); const d = await r.json();
    const kg = d.kg_stats || {};
    document.getElementById('kgInfo').textContent = kg.total_nodes + ' nodes / ' + kg.total_edges + ' edges';
    document.getElementById('llmStatus').textContent = d.llm_available ? d.model_name : 'offline';
    document.getElementById('llmDot').className = 'dot ' + (d.llm_available ? 'dot-on' : 'dot-off');
  } catch(e) {}
}

// ===== Knowledge Graph Visualization =====
let graphNetwork = null;
let graphData = null;

const nodeColors = {
  'Problem': {bg: '#ee4c2c', border: '#c0392b', shape: 'dot', size: 28},
  'Error':   {bg: '#ef4444', border: '#dc2626', shape: 'triangle', size: 22},
  'API':     {bg: '#3b82f6', border: '#2563eb', shape: 'box', size: 18},
  'Concept': {bg: '#10b981', border: '#059669', shape: 'diamond', size: 20},
  'Solution':{bg: '#8b5cf6', border: '#7c3aed', shape: 'star', size: 22},
  'Cause':   {bg: '#f97316', border: '#ea580c', shape: 'dot', size: 20},
  'Command': {bg: '#06b6d4', border: '#0891b2', shape: 'box', size: 16},
  'DocPage': {bg: '#6366f1', border: '#4f46e5', shape: 'square', size: 16},
  'CodeExample': {bg: '#ec4899', border: '#db2777', shape: 'square', size: 16}
};

const edgeColors = {
  'HAS_API': '#3b82f6', 'HAS_CAUSE': '#f97316', 'HAS_SOLUTION': '#8b5cf6',
  'CHECK_BY': '#06b6d4', 'MENTIONED_IN': '#6366f1', 'HAS_PARAMETER': '#10b981',
  'RELATED_TO': '#9ca3af', 'HAS_EXAMPLE': '#ec4899', 'SIMILAR_TO': '#f59e0b',
  'HAS_ERROR': '#ef4444'
};

async function initGraph() {
  if (graphNetwork) return;
  try { const r = await fetch(API + '/graph'); graphData = await r.json(); } catch(e) {
    document.getElementById('graphContainer').innerHTML = '<p style=\"padding:40px;text-align:center;color:#999\">Failed to load graph data</p>'; return;
  }

  if (typeof vis === 'undefined') {
    // Fallback: show nodes/edges as table
    var html = '<div style=\"padding:16px;max-height:600px;overflow:auto\"><table style=\"width:100%;font-size:13px;border-collapse:collapse\">';
    html += '<tr style=\"background:#f3f4f6\"><th style=\"padding:8px;text-align:left;border-bottom:1px solid #e5e7eb\">Node</th><th>Type</th><th>Description</th></tr>';
    graphData.nodes.forEach(function(n){
      html += '<tr><td style=\"padding:6px 8px;border-bottom:1px solid #f3f4f6\"><b>'+n.name+'</b></td><td>'+n.label+'</td><td style=\"color:#6b7280;font-size:12px\">'+(n.description||'')+'</td></tr>';
    });
    html += '</table><p style=\"padding:12px;color:#9ca3af;font-size:12px\">'+graphData.nodes.length+' nodes, '+graphData.edges.length+' edges (vis.js CDN unavailable - showing table view)</p></div>';
    document.getElementById('graphContainer').innerHTML = html;
    document.getElementById('graphInfo').innerHTML = 'Loaded '+graphData.nodes.length+' nodes, '+graphData.edges.length+' edges (table view)';
    document.getElementById('graphInfo').classList.add('show');
    return;
  }

  try {
    const nodes = graphData.nodes.map(n => ({
      id: n.id,
      label: n.name.length > 20 ? n.name.substring(0, 18) + '...' : n.name,
      title: '<b>' + n.name + '</b><br>' + (n.label ? '[' + n.label + '] ' : '') + (n.description || ''),
      color: nodeColors[n.label] || {bg: '#9ca3af', border: '#6b7280'},
      shape: (nodeColors[n.label] || {}).shape || 'dot',
      size: (nodeColors[n.label] || {}).size || 16,
      font: {size: 11, color: '#374151', face: 'Microsoft YaHei'},
      borderWidth: 1.5
    }));

    const edges = graphData.edges.map((e, i) => ({
      id: i,
      from: e.source,
      to: e.target,
      label: e.relation.replace('_', ' '),
      title: '<b>' + e.relation + '</b><br>' + (e.evidence || ''),
      color: {color: edgeColors[e.relation] || '#9ca3af', opacity: 0.6},
      arrows: {to: {enabled: true, scaleFactor: 0.6}},
      font: {size: 8, color: '#6b7280', strokeWidth: 2, strokeColor: '#fff'},
      width: 1.2,
      smooth: {type: 'continuous'}
    }));

    const container = document.getElementById('graphContainer');
    graphNetwork = new vis.Network(container, {nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges)}, {
      physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -40, centralGravity: 0.005, springLength: 120, springConstant: 0.08 }, stabilization: { iterations: 200 } },
      interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true, navigationButtons: false },
      nodes: { scaling: { min: 12, max: 40 } }
    });

    // Click node to show info
    graphNetwork.on('click', function(params) {
      const info = document.getElementById('graphInfo');
      if (params.nodes.length > 0) {
        const nid = params.nodes[0];
        const node = graphData.nodes.find(n => n.id === nid);
        if (node) {
          // Count connected edges
          const outEdges = graphData.edges.filter(e => e.source === nid).length;
          const inEdges = graphData.edges.filter(e => e.target === nid).length;
          info.innerHTML = '<b>' + node.name + '</b> [' + node.label + '] &mdash; ' +
            (node.description || '') + ' &nbsp;|&nbsp; Out: ' + outEdges + ' edges, In: ' + inEdges + ' edges' +
            (node.url ? ' <a href="' + node.url + '" target="_blank">Source</a>' : '');
          info.classList.add('show');
        }
      } else {
        info.classList.remove('show');
      }
    });

    document.getElementById('graphInfo').innerHTML = 'Loaded: ' + nodes.length + ' nodes, ' + edges.length + ' edges. Click a node to see details. Drag to pan, scroll to zoom.';
    document.getElementById('graphInfo').classList.add('show');

  } catch(e) {
    document.getElementById('graphContainer').innerHTML = '<p style="padding:40px;text-align:center;color:#999">Failed to load graph: ' + e.message + '</p>';
  }
}

// Auto-load
loadStats();
</script>
</body>
</html>
"""


# ---- API 路由 ----
@app.get("/", response_class=HTMLResponse)
async def index():
    """Web 前端页面"""
    return FRONTEND_HTML


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "PyTorchDiagQA"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """问答接口"""
    engine = get_engine()
    result = engine.ask(req.query, use_llm=req.use_llm)
    return AskResponse(**result)


@app.get("/stats")
async def stats():
    """系统统计信息"""
    engine = get_engine()
    return engine.stats()


@app.get("/graph")
async def graph():
    """获取知识图谱数据"""
    engine = get_engine()
    return {
        "nodes": engine.kg.get_all_nodes(),
        "edges": engine.kg.get_all_edges(),
    }


@app.post("/build_kg", response_model=BuildKGResponse)
async def build_kg():
    """触发知识图谱构建"""
    try:
        from kg_builder.graph_builder import build_graph
        from kg_builder.export_csv import export_to_csv

        graph = build_graph()
        export_to_csv()

        return BuildKGResponse(
            status="success",
            message=f"知识图谱构建完成: {len(graph['nodes'])} 节点, {len(graph['edges'])} 条边"
        )
    except Exception as e:
        return BuildKGResponse(
            status="error",
            message=f"知识图谱构建失败: {e}"
        )


def run_server():
    """启动服务器（自动处理端口冲突）"""
    config_path = get_config_path()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    server_config = config.get("server", {})
    host = server_config.get("host", "127.0.0.1")
    base_port = server_config.get("port", 18888)

    # 尝试多个端口，避免冲突
    import socket
    for offset in range(10):
        port = base_port + offset
        # 检测端口是否可用
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((host, port))
        sock.close()
        if result != 0:
            # 端口空闲
            print(f"[OK] PyTorchDiagQA backend started: http://{host}:{port}")
            print(f"    Web UI: http://{host}:{port}")
            print(f"    Swagger: http://{host}:{port}/docs")
            import webbrowser
            webbrowser.open(f"http://{host}:{port}")
            uvicorn.run(app, host=host, port=port, log_level="info")
            return

    print(f"[ERROR] Ports {base_port}-{base_port+9} are all in use.")
    print(f"        Close other instances and try again.")
    input("Press Enter to exit...")


if __name__ == "__main__":
    run_server()
