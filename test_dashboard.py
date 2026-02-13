#!/usr/bin/env python3
"""Crypto Trader — Test Runner + Live Dashboard

Runs all trading models end-to-end and serves a live web dashboard on port 8888.
"""

import sys, os, json, threading, time, traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from flask import Flask, jsonify, Response

# ── Global state shared between test thread and Flask ────────────────────────
state = {
    "status": "initializing",
    "started_at": datetime.now().isoformat(),
    "finished_at": None,
    "current_step": "",
    "steps": [],           # list of {name, status, detail, ts}
    "data_info": {},       # pair → row count
    "model_status": {},    # model_name → {status, detail}
    "signals": {},         # model_name → list of signal dicts
    "ensemble_signals": [],
    "summary": {},
    "errors": [],
}

def add_step(name, status="running", detail=""):
    entry = {"name": name, "status": status, "detail": detail, "ts": datetime.now().isoformat()}
    # Update existing or append
    for s in state["steps"]:
        if s["name"] == name:
            s.update(entry)
            return
    state["steps"].append(entry)
    state["current_step"] = name

def signal_to_dict(sig):
    return {
        "direction": sig.direction.name,
        "confidence": round(sig.confidence, 4),
        "pair": sig.pair,
        "model": sig.model_name,
        "timestamp": str(sig.timestamp) if sig.timestamp else None,
        "stop_loss": round(sig.stop_loss, 2) if sig.stop_loss else None,
        "take_profit": round(sig.take_profit, 2) if sig.take_profit else None,
        "metadata": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in (sig.metadata or {}).items()
                     if k != "model_details"},
    }


# ── Test runner ──────────────────────────────────────────────────────────────
def run_tests():
    try:
        # Load config
        add_step("Load config")
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        add_step("Load config", "done", f"Loaded {config_path}")

        # Import modules
        add_step("Import modules")
        from src.data.fetcher import DataFetcher
        from src.models.mean_reversion import MeanReversionModel
        from src.models.momentum import MomentumGARCHModel
        from src.models.ml_model import MLModel
        from src.models.ensemble import EnsembleModel
        add_step("Import modules", "done")

        # ── Fetch data ───────────────────────────────────────────────────
        add_step("Fetch data")
        fetcher = DataFetcher(config)
        pairs = config["trading"]["pairs"]  # BTC/USDT, ETH/USDT, SOL/USDT
        pair_data = {}
        for pair in pairs:
            try:
                df = fetcher.fetch(pair, timeframe="1h", days=90)
                pair_data[pair] = df
                state["data_info"][pair] = len(df)
            except Exception as e:
                state["errors"].append(f"Fetch {pair}: {e}")
                state["data_info"][pair] = f"ERROR: {e}"
        add_step("Fetch data", "done", f"{sum(v for v in state['data_info'].values() if isinstance(v, int))} total rows")

        primary_pair = pairs[0]  # BTC/USDT
        primary_data = pair_data[primary_pair]

        # ── Fit Mean Reversion ───────────────────────────────────────────
        add_step("Fit mean_reversion")
        state["model_status"]["mean_reversion"] = {"status": "fitting", "detail": ""}
        try:
            mr = MeanReversionModel(config)
            mr.fit(primary_data, pair_data=pair_data)
            state["model_status"]["mean_reversion"] = {"status": "fitted", "detail": f"half_lives={mr._half_lives}"}
            mr_signals = mr.generate_signals(primary_data, pair_data=pair_data)
            state["signals"]["mean_reversion"] = [signal_to_dict(s) for s in mr_signals]
            add_step("Fit mean_reversion", "done", f"{len(mr_signals)} signals")
        except Exception as e:
            state["model_status"]["mean_reversion"] = {"status": "error", "detail": str(e)}
            state["errors"].append(f"mean_reversion: {traceback.format_exc()}")
            add_step("Fit mean_reversion", "error", str(e))

        # ── Fit Momentum / GARCH ─────────────────────────────────────────
        add_step("Fit momentum")
        state["model_status"]["momentum"] = {"status": "fitting", "detail": ""}
        try:
            mom = MomentumGARCHModel(config)
            mom.fit(primary_data)
            gp = mom._garch_params or {}
            state["model_status"]["momentum"] = {
                "status": "fitted",
                "detail": f"GARCH params: ω={gp.get('omega',0):.6f} α={gp.get('alpha',0):.4f} β={gp.get('beta',0):.4f}",
            }
            mom_signals = mom.generate_signals(primary_data)
            state["signals"]["momentum"] = [signal_to_dict(s) for s in mom_signals]
            add_step("Fit momentum", "done", f"{len(mom_signals)} signals")
        except Exception as e:
            state["model_status"]["momentum"] = {"status": "error", "detail": str(e)}
            state["errors"].append(f"momentum: {traceback.format_exc()}")
            add_step("Fit momentum", "error", str(e))

        # ── Fit ML Model ─────────────────────────────────────────────────
        add_step("Fit ml_model")
        state["model_status"]["ml_model"] = {"status": "fitting", "detail": ""}
        try:
            ml = MLModel(config)
            ml.fit(primary_data)
            top_feats = dict(sorted(ml._feature_importance.items(), key=lambda x: -x[1])[:3])
            state["model_status"]["ml_model"] = {
                "status": "fitted",
                "detail": f"Top features: {top_feats}",
            }
            ml_signals = ml.generate_signals(primary_data)
            state["signals"]["ml_model"] = [signal_to_dict(s) for s in ml_signals]
            add_step("Fit ml_model", "done", f"{len(ml_signals)} signals")
        except Exception as e:
            state["model_status"]["ml_model"] = {"status": "error", "detail": str(e)}
            state["errors"].append(f"ml_model: {traceback.format_exc()}")
            add_step("Fit ml_model", "error", str(e))

        # ── Ensemble ─────────────────────────────────────────────────────
        add_step("Run ensemble")
        try:
            ensemble = EnsembleModel(config)
            ensemble.fit(primary_data, pair_data=pair_data)
            ens_signals = ensemble.generate_signals(primary_data, pair_data=pair_data)
            state["ensemble_signals"] = [signal_to_dict(s) for s in ens_signals]
            add_step("Run ensemble", "done", f"{len(ens_signals)} ensemble signals")
        except Exception as e:
            state["errors"].append(f"ensemble: {traceback.format_exc()}")
            add_step("Run ensemble", "error", str(e))

        # ── Summary ──────────────────────────────────────────────────────
        all_sigs = []
        for sigs in state["signals"].values():
            all_sigs.extend(sigs)
        all_sigs.extend(state["ensemble_signals"])

        directions = {}
        confidences = []
        for s in all_sigs:
            d = s["direction"]
            directions[d] = directions.get(d, 0) + 1
            confidences.append(s["confidence"])

        state["summary"] = {
            "total_signals": len(all_sigs),
            "directions": directions,
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
            "max_confidence": round(max(confidences), 4) if confidences else 0,
            "min_confidence": round(min(confidences), 4) if confidences else 0,
            "models_fitted": sum(1 for m in state["model_status"].values() if m["status"] == "fitted"),
            "models_errored": sum(1 for m in state["model_status"].values() if m["status"] == "error"),
        }

        state["status"] = "complete"
        state["finished_at"] = datetime.now().isoformat()
        state["current_step"] = "All done ✓"

    except Exception as e:
        state["status"] = "error"
        state["errors"].append(traceback.format_exc())
        state["current_step"] = f"FATAL: {e}"


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Crypto Trader</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0a0a;--card:#111111;--border:#1a1a1a;
  --text:#fafafa;--text-secondary:#71717a;
  --green:#22c55e;--green-bg:rgba(34,197,94,0.08);
  --amber:#f59e0b;--amber-bg:rgba(245,158,11,0.08);
  --red:#ef4444;--red-bg:rgba(239,68,68,0.08);
  --blue:#3b82f6;--blue-bg:rgba(59,130,246,0.08);
}
body{background:var(--bg);color:var(--text);font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}
.container{max-width:1120px;margin:0 auto;padding:40px 24px}

/* Header */
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px}
.header-left{display:flex;align-items:center;gap:12px}
.header-title{font-size:20px;font-weight:600;color:var(--text);letter-spacing:-0.02em}
.pill{display:inline-flex;align-items:center;padding:3px 10px;border-radius:9999px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em}
.pill-green{background:var(--green-bg);color:var(--green)}
.pill-amber{background:var(--amber-bg);color:var(--amber)}
.pill-red{background:var(--red-bg);color:var(--red)}
.pill-blue{background:var(--blue-bg);color:var(--blue)}
.pill-gray{background:rgba(113,113,122,0.1);color:var(--text-secondary)}
.header-sync{font-size:13px;color:var(--text-secondary)}

/* Progress */
.progress-section{margin-bottom:32px}
.progress-header{display:flex;justify-content:space-between;margin-bottom:8px}
.progress-label{font-size:12px;color:var(--text-secondary)}
.progress-track{height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.progress-fill{height:100%;background:var(--green);border-radius:2px;transition:width 0.4s ease}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:24px}
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:32px}
.stat-label{font-size:11px;font-weight:500;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px}
.stat-value{font-size:28px;font-weight:600;font-family:'JetBrains Mono','SF Mono',monospace;color:var(--text);letter-spacing:-0.02em}
.stat-sub{font-size:12px;margin-top:4px}
.stat-sub.green{color:var(--green)}
.stat-sub.amber{color:var(--amber)}
.stat-sub.red{color:var(--red)}

/* Section headers */
.section-header{font-size:11px;font-weight:500;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:16px}

/* Steps */
.steps-section{margin-bottom:32px}
.step-row{display:flex;align-items:center;justify-content:space-between;padding:12px 0;border-bottom:1px solid var(--border)}
.step-row:last-child{border-bottom:none}
.step-left{display:flex;align-items:center;gap:10px}
.step-check{width:20px;height:20px;flex-shrink:0}
.step-name{font-size:14px;color:var(--text);font-weight:400}
.step-right{display:flex;align-items:center;gap:12px}
.step-time{font-size:12px;color:var(--text-secondary);font-family:'JetBrains Mono',monospace}

/* Model grid */
.model-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:32px}
@media(max-width:768px){.model-grid{grid-template-columns:1fr}.stats-grid{grid-template-columns:1fr}}
.model-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.model-name{font-size:14px;font-weight:500;color:var(--text)}
.model-metric{margin-bottom:12px}
.model-metric-label{font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px}
.model-metric-value{font-size:16px;font-weight:600;color:var(--text);font-family:'JetBrains Mono',monospace}
.model-bar{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:auto}
.model-bar-fill{height:100%;border-radius:2px;transition:width 0.4s ease}

/* Signals table */
.signals-section{margin-bottom:32px}
.signals-table{width:100%}
.signals-table th{text-align:left;font-size:11px;font-weight:500;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.04em;padding:0 0 12px 0}
.signals-table td{padding:10px 0;border-top:1px solid var(--border);font-size:13px;color:var(--text)}
.signals-table .mono{font-family:'JetBrains Mono',monospace;font-size:12px}
.conf-bar{width:64px;height:4px;background:var(--border);border-radius:2px;overflow:hidden;display:inline-block;vertical-align:middle;margin-right:8px}
.conf-fill{height:100%;border-radius:2px}
.empty-state{text-align:center;padding:48px 24px}
.empty-title{font-size:14px;color:var(--text);margin-bottom:6px}
.empty-sub{font-size:13px;color:var(--text-secondary);max-width:360px;margin:0 auto;line-height:1.5}

/* Errors */
.errors-section{margin-bottom:32px}
.error-box{background:var(--red-bg);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:16px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--red);white-space:pre-wrap;max-height:200px;overflow-y:auto;margin-bottom:8px}
.error-box:last-child{margin-bottom:0}

/* SVGs */
.check-icon{color:var(--green)}
.spinner-icon{color:var(--amber);animation:spin 1s linear infinite}
.error-icon{color:var(--red)}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
</style>
<script>
let refreshTimer;
const SVG_CHECK='<svg class="step-check check-icon" viewBox="0 0 20 20" fill="currentColor"><circle cx="10" cy="10" r="10" fill="rgba(34,197,94,0.12)"/><path d="M6 10.5l2.5 2.5 5.5-5.5" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const SVG_SPINNER='<svg class="step-check spinner-icon" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="2" opacity="0.2"/><path d="M10 2a8 8 0 0 1 8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
const SVG_ERROR='<svg class="step-check error-icon" viewBox="0 0 20 20" fill="currentColor"><circle cx="10" cy="10" r="10" fill="rgba(239,68,68,0.12)"/><path d="M7 7l6 6M13 7l-6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
const MODEL_COLORS={mean_reversion:'var(--green)',momentum:'var(--amber)',ml_model:'var(--blue)',ensemble:'var(--green)'};

function load(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    render(d);
    if(d.status==='complete'||d.status==='error')clearInterval(refreshTimer);
  }).catch(()=>{});
}

function fmtElapsed(d){
  if(!d.started_at)return'--';
  let s=new Date(d.started_at),e=d.finished_at?new Date(d.finished_at):new Date();
  let sec=((e-s)/1000);
  return sec<60?sec.toFixed(1)+'s':Math.floor(sec/60)+'m '+Math.round(sec%60)+'s';
}

function render(d){
  // Header
  let isComplete=d.status==='complete',isError=d.status==='error';
  let pillCls=isComplete?'pill-green':isError?'pill-red':'pill-amber';
  let pillText=isComplete?'Complete':isError?'Error':'Running';
  document.getElementById('status-pill').className='pill '+pillCls;
  document.getElementById('status-pill').textContent=pillText;
  document.getElementById('header-sync').textContent='Elapsed '+fmtElapsed(d);

  // Progress
  let total=7,done=d.steps.filter(s=>s.status==='done').length;
  let pct=Math.round(done/total*100);
  document.getElementById('progress-pct').textContent=pct+'%';
  document.getElementById('progress-fill').style.width=pct+'%';
  if(isComplete)document.getElementById('progress-fill').style.background='var(--green)';

  // Stat cards
  let totalRows=0,pairCount=0;
  for(let[p,v]of Object.entries(d.data_info)){if(typeof v==='number'){totalRows+=v;pairCount++;}}
  document.getElementById('stat-rows').textContent=totalRows.toLocaleString();
  document.getElementById('stat-rows-sub').textContent=pairCount+' trading pairs';

  let modelsFitted=(d.summary.models_fitted||0);
  let modelsTotal=Object.keys(d.model_status).length||'--';
  document.getElementById('stat-models').textContent=modelsFitted+'/'+modelsTotal;
  let modelsErr=d.summary.models_errored||0;
  let mSub=document.getElementById('stat-models-sub');
  mSub.textContent=modelsErr?modelsErr+' errored':'All healthy';
  mSub.className='stat-sub '+(modelsErr?'red':'green');

  let totalSigs=d.summary.total_signals||0;
  document.getElementById('stat-signals').textContent=totalSigs;
  let avgConf=d.summary.avg_confidence||0;
  document.getElementById('stat-signals-sub').textContent=avgConf?(avgConf*100).toFixed(1)+'% avg confidence':'--';

  // Validation steps
  let stepsHtml='';
  d.steps.forEach(s=>{
    let icon=s.status==='done'?SVG_CHECK:s.status==='error'?SVG_ERROR:SVG_SPINNER;
    let pillCls=s.status==='done'?'pill-green':s.status==='error'?'pill-red':'pill-amber';
    let pillText=s.status==='done'?'Pass':s.status==='error'?'Fail':'Running';
    let timeStr=s.ts?new Date(s.ts).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'';
    stepsHtml+=`<div class="step-row">
      <div class="step-left">${icon}<span class="step-name">${s.name}</span></div>
      <div class="step-right"><span class="pill ${pillCls}">${pillText}</span><span class="step-time">${timeStr}</span></div>
    </div>`;
  });
  document.getElementById('steps-list').innerHTML=stepsHtml||'<div style="padding:12px 0;color:var(--text-secondary)">Waiting for steps...</div>';

  // Model status cards
  let modelHtml='';
  let modelEntries=Object.entries(d.model_status);
  if(modelEntries.length){
    modelEntries.forEach(([name,m])=>{
      let color=MODEL_COLORS[name]||'var(--green)';
      let pillCls=m.status==='fitted'?'pill-green':m.status==='error'?'pill-red':'pill-amber';
      let pillText=m.status.toUpperCase();
      // Extract a key metric from detail
      let metric='--';
      let metricLabel='Status';
      if(m.detail){
        if(name==='momentum'&&m.detail.includes('GARCH')){
          metricLabel='GARCH Params';
          let match=m.detail.match(/α=([\d.]+)/);
          metric=match?'α = '+match[1]:m.detail.substring(0,30);
        }else if(name==='mean_reversion'&&m.detail.includes('half_lives')){
          metricLabel='Half Lives';
          metric=m.detail.replace('half_lives=','').substring(0,30);
        }else if(name==='ml_model'&&m.detail.includes('features')){
          metricLabel='Top Features';
          metric=m.detail.replace('Top features: ','').substring(0,30);
        }else{
          metricLabel='Detail';
          metric=m.detail.substring(0,30);
        }
      }
      // Confidence for bar
      let sigs=d.signals[name]||[];
      let avgC=sigs.length?sigs.reduce((a,s)=>a+s.confidence,0)/sigs.length:0;
      modelHtml+=`<div class="card">
        <div class="model-card-header"><span class="model-name">${name.replace(/_/g,' ')}</span><span class="pill ${pillCls}">${pillText}</span></div>
        <div class="model-metric"><div class="model-metric-label">${metricLabel}</div><div class="model-metric-value" style="font-size:13px;font-weight:500">${metric}</div></div>
        <div class="model-metric"><div class="model-metric-label">Signals</div><div class="model-metric-value">${sigs.length}</div></div>
        <div class="model-bar"><div class="model-bar-fill" style="width:${avgC*100}%;background:${color}"></div></div>
      </div>`;
    });
  }else{
    modelHtml='<div class="card" style="grid-column:1/-1"><div style="padding:12px 0;color:var(--text-secondary);text-align:center">Waiting for models...</div></div>';
  }
  document.getElementById('model-grid').innerHTML=modelHtml;

  // Signals
  let allSignals=[];
  for(let[model,sigs]of Object.entries(d.signals)){sigs.forEach(s=>{s._model=model;allSignals.push(s);});}
  d.ensemble_signals.forEach(s=>{s._model='ensemble';allSignals.push(s);});

  let sigHtml='';
  if(allSignals.length){
    sigHtml='<table class="signals-table"><thead><tr><th>Pair</th><th>Model</th><th>Direction</th><th>Confidence</th><th>Stop Loss</th><th>Take Profit</th></tr></thead><tbody>';
    allSignals.forEach(s=>{
      let dirCls=s.direction==='LONG'?'pill-green':s.direction==='SHORT'?'pill-red':'pill-gray';
      let dirText=s.direction==='LONG'?'Buy':s.direction==='SHORT'?'Sell':'Hold';
      let confPct=(s.confidence*100).toFixed(1);
      let confColor=s.confidence>0.5?'var(--green)':s.confidence>0.25?'var(--amber)':'var(--red)';
      sigHtml+=`<tr>
        <td class="mono">${s.pair}</td>
        <td style="color:var(--text-secondary);font-size:12px">${(s._model||'').replace(/_/g,' ')}</td>
        <td><span class="pill ${dirCls}">${dirText}</span></td>
        <td><span class="conf-bar"><span class="conf-fill" style="width:${confPct}%;background:${confColor}"></span></span><span class="mono">${confPct}%</span></td>
        <td class="mono">${s.stop_loss?'$'+s.stop_loss.toLocaleString():'--'}</td>
        <td class="mono">${s.take_profit?'$'+s.take_profit.toLocaleString():'--'}</td>
      </tr>`;
    });
    sigHtml+='</tbody></table>';
  }else{
    let msg=isComplete?'No signals generated':'Waiting for model output...';
    let sub=isComplete?'Models did not find actionable opportunities at the current bar':'Pipeline is still running';
    sigHtml=`<div class="empty-state"><div class="empty-title">${msg}</div><div class="empty-sub">${sub}</div></div>`;
  }
  document.getElementById('signals-container').innerHTML=sigHtml;

  // Errors
  if(d.errors.length){
    document.getElementById('errors-section').style.display='block';
    document.getElementById('errors-container').innerHTML=d.errors.map(e=>`<div class="error-box">${e}</div>`).join('');
  }else{
    document.getElementById('errors-section').style.display='none';
  }
}
window.onload=()=>{load();refreshTimer=setInterval(load,2000);};
</script>
</head><body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <span class="header-title">Crypto Trader</span>
      <span id="status-pill" class="pill pill-amber">Running</span>
    </div>
    <span id="header-sync" class="header-sync">Elapsed --</span>
  </div>

  <!-- Progress -->
  <div class="progress-section">
    <div class="progress-header">
      <span class="progress-label">Pipeline Progress</span>
      <span class="progress-label" id="progress-pct">0%</span>
    </div>
    <div class="progress-track"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
  </div>

  <!-- Stat cards -->
  <div class="stats-grid">
    <div class="card">
      <div class="stat-label">Data Rows</div>
      <div class="stat-value" id="stat-rows">--</div>
      <div class="stat-sub green" id="stat-rows-sub">--</div>
    </div>
    <div class="card">
      <div class="stat-label">Models Fitted</div>
      <div class="stat-value" id="stat-models">--</div>
      <div class="stat-sub green" id="stat-models-sub">--</div>
    </div>
    <div class="card">
      <div class="stat-label">Total Signals</div>
      <div class="stat-value" id="stat-signals">--</div>
      <div class="stat-sub green" id="stat-signals-sub">--</div>
    </div>
  </div>

  <!-- Validation Steps -->
  <div class="steps-section">
    <div class="section-header">Validation Steps</div>
    <div class="card">
      <div id="steps-list"></div>
    </div>
  </div>

  <!-- Model Status -->
  <div style="margin-bottom:32px">
    <div class="section-header">Model Status</div>
    <div class="model-grid" id="model-grid"></div>
  </div>

  <!-- Signals -->
  <div class="signals-section">
    <div class="section-header">Signals</div>
    <div class="card" id="signals-container">
      <div class="empty-state">
        <div class="empty-title">Waiting for model output...</div>
        <div class="empty-sub">Pipeline is still running</div>
      </div>
    </div>
  </div>

  <!-- Errors (hidden by default) -->
  <div class="errors-section" id="errors-section" style="display:none">
    <div class="section-header">Errors</div>
    <div id="errors-container"></div>
  </div>
</div>
</body></html>"""


@app.route("/")
def index():
    return Response(DASHBOARD_HTML, content_type="text/html")

@app.route("/api/state")
def api_state():
    return jsonify(state)


if __name__ == "__main__":
    # Start test runner in background thread
    t = threading.Thread(target=run_tests, daemon=True)
    t.start()
    # Serve dashboard
    print("Dashboard running at http://localhost:8888")
    app.run(host="0.0.0.0", port=8888, debug=False, use_reloader=False)
