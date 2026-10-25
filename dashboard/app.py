"""
dashboard/app.py
================
Ground Vibration Monitoring System (GVMS)
Flask web dashboard with real-time ML inference endpoint.

Routes
------
GET  /                    -> Main dashboard HTML
GET  /api/status          -> System status JSON
POST /api/infer           -> Run ML inference on posted sensor data
POST /api/infer/batch     -> Batch inference on array of samples
GET  /api/model/info      -> Model metadata
GET  /api/history         -> Last 200 inference results
POST /api/reset           -> Reset rolling buffer

Start server
------------
    python dashboard/app.py
    # or
    python -m flask --app dashboard/app run --host 0.0.0.0 --port 5000
"""

import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, render_template_string
import socket
import threading
from zeroconf import Zeroconf, ServiceInfo
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference import GVMSInference

# â”€â”€ App setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# â”€â”€ Global inference engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_inference = GVMSInference(smoothing_n=3)
_history: deque = deque(maxlen=200)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HTML Dashboard
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GVMS â€” Ground Vibration Monitoring System</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@400;700;800&display=swap');

    :root {
      --black:  #000000;
      --white:  #ffffff;
      --yellow: #ffe600;
      --red:    #ff2b2b;
      --green:  #00ff6a;
      --border: 3px solid #000;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Space Grotesk', monospace;
      background: var(--white);
      color: var(--black);
      min-height: 100vh;
    }

    /* â”€â”€ HEADER â”€â”€ */
    header {
      background: var(--black);
      border-bottom: 5px solid var(--yellow);
      padding: 20px 32px;
      display: flex;
      align-items: center;
      gap: 20px;
      flex-wrap: wrap;
    }
    header h1 {
      font-family: 'Space Grotesk', sans-serif;
      color: var(--yellow);
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: 3px;
      text-transform: uppercase;
    }
    header .subtitle {
      color: #aaa;
      font-family: 'Space Mono', monospace;
      font-size: 0.78rem;
      letter-spacing: 1px;
    }
    .badge {
      background: var(--yellow);
      color: var(--black);
      padding: 4px 12px;
      font-size: 0.75rem;
      font-weight: 700;
      font-family: 'Space Mono', monospace;
      text-transform: uppercase;
      border: 3px solid var(--black);
      box-shadow: 3px 3px 0 var(--black);
      margin-left: auto;
    }

    /* â”€â”€ LAYOUT â”€â”€ */
    .container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      padding: 32px;
      max-width: 1400px;
      margin: 0 auto;
    }

    /* â”€â”€ CARDS â”€â”€ */
    .card {
      background: var(--white);
      border: var(--border);
      padding: 24px;
      position: relative;
      margin: -1px;   /* collapse adjacent borders */
    }
    .card:hover { z-index: 1; }

    .card h2 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.85rem;
      font-weight: 800;
      letter-spacing: 3px;
      text-transform: uppercase;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 3px solid var(--black);
    }

    /* â”€â”€ METRICS â”€â”€ */
    .metric {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 10px 0;
      border-bottom: 2px solid #e0e0e0;
    }
    .metric:last-child { border-bottom: none; }
    .metric-label {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      color: #555;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .metric-value {
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      font-size: 1rem;
      color: var(--black);
    }
    .metric-value.highlight { color: var(--black); font-size: 1.3rem; }
    .metric-value.alert-val { color: var(--red); }

    /* â”€â”€ STATUS DOT â”€â”€ */
    .status-dot {
      display: inline-block;
      width: 12px; height: 12px;
      border: 2px solid var(--black);
      margin-right: 8px;
      vertical-align: middle;
    }
    .status-dot.ok  { background: var(--green); }
    .status-dot.err { background: var(--red); }

    /* â”€â”€ PREDICTION BOX â”€â”€ */
    .prediction-box {
      text-align: center;
      padding: 28px 16px;
      background: var(--black);
      border: var(--border);
      margin-bottom: 16px;
    }
    .pred-eyebrow {
      font-family: 'Space Mono', monospace;
      font-size: 0.7rem;
      color: #888;
      letter-spacing: 3px;
      text-transform: uppercase;
    }
    .pred-label {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 2.6rem;
      font-weight: 800;
      color: var(--yellow);
      letter-spacing: 2px;
      margin: 8px 0;
      word-break: break-word;
    }
    .pred-conf {
      font-family: 'Space Mono', monospace;
      color: #aaa;
      font-size: 0.85rem;
    }
    .pred-meta {
      margin-top: 14px;
      display: flex;
      justify-content: center;
      gap: 24px;
    }
    .pred-meta span {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      color: #888;
    }
    .pred-meta strong { color: var(--yellow); }

    /* â”€â”€ BAR CHART â”€â”€ */
    .class-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 8px 0;
    }
    .class-name {
      width: 138px;
      font-family: 'Space Mono', monospace;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      flex-shrink: 0;
    }
    .bar-track {
      flex: 1;
      background: #eee;
      border: 2px solid var(--black);
      height: 18px;
    }
    .bar-fill {
      height: 100%;
      background: var(--yellow);
      border-right: 2px solid var(--black);
      transition: width 0.4s ease;
    }
    .bar-count {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      font-weight: 700;
      width: 28px;
      text-align: right;
    }

    /* â”€â”€ LOG BOX â”€â”€ */
    .log-box {
      background: var(--black);
      border: var(--border);
      padding: 14px;
      height: 220px;
      overflow-y: auto;
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      line-height: 1.8;
    }
    .log-box::-webkit-scrollbar { width: 6px; }
    .log-box::-webkit-scrollbar-thumb { background: var(--yellow); }
    .log-entry { color: #888; }
    .log-entry .ts  { color: var(--green); margin-right: 10px; }
    .log-entry .lbl { color: var(--yellow); font-weight: 700; }
    .log-entry .conf { color: #ff9900; }

    /* â”€â”€ TEST FORM â”€â”€ */
    #test-form label {
      font-family: 'Space Mono', monospace;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #555;
      margin-right: 4px;
    }
    #test-form input {
      background: var(--white);
      border: 3px solid var(--black);
      color: var(--black);
      padding: 6px 8px;
      width: 80px;
      font-family: 'Space Mono', monospace;
      font-size: 0.82rem;
      font-weight: 700;
    }
    #test-form input:focus {
      outline: none;
      box-shadow: 4px 4px 0 var(--black);
    }
    #test-form .row {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }

    /* â”€â”€ BUTTONS â”€â”€ */
    button {
      background: var(--black);
      color: var(--yellow);
      border: 3px solid var(--black);
      padding: 8px 18px;
      cursor: pointer;
      font-family: 'Space Mono', monospace;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      box-shadow: 4px 4px 0 #555;
      transition: box-shadow 0.1s, transform 0.1s;
    }
    button:hover  { box-shadow: 2px 2px 0 #555; transform: translate(2px, 2px); }
    button:active { box-shadow: none; transform: translate(4px, 4px); }
    button.danger {
      background: var(--red);
      color: var(--white);
      border-color: var(--black);
      box-shadow: 4px 4px 0 #555;
    }
    button.danger:hover { box-shadow: 2px 2px 0 #555; transform: translate(2px, 2px); }

    /* â”€â”€ RESULT BOX â”€â”€ */
    #result-box {
      margin-top: 14px;
      padding: 12px;
      background: var(--black);
      border: var(--border);
      font-family: 'Space Mono', monospace;
      font-size: 0.78rem;
      min-height: 52px;
      color: var(--green);
      white-space: pre-wrap;
    }

    /* â”€â”€ FULL WIDTH â”€â”€ */
    .full-width { grid-column: 1 / -1; }

    /* â”€â”€ FOOTER â”€â”€ */
    footer {
      text-align: center;
      font-family: 'Space Mono', monospace;
      font-size: 0.7rem;
      color: #888;
      padding: 20px 32px;
      border-top: 5px solid var(--black);
      background: #f5f5f5;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    /* â”€â”€ TICKER STRIP â”€â”€ */
    .ticker {
      background: var(--yellow);
      border-top: 3px solid var(--black);
      border-bottom: 3px solid var(--black);
      padding: 6px 0;
      overflow: hidden;
      white-space: nowrap;
    }
    .ticker-inner {
      display: inline-block;
      animation: ticker 22s linear infinite;
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--black);
    }
    @keyframes ticker {
      0%   { transform: translateX(100vw); }
      100% { transform: translateX(-100%); }
    }
  </style>
</head>
<body>

<header>
  <h1>/// GROUND VIBRATION MONITORING SYSTEM</h1>
  <div class="subtitle">ESP32 + MPU-6050 &nbsp;Â·&nbsp; REAL-TIME ML CLASSIFICATION</div>
  <div class="badge">GVMS v1.0</div>
</header>

<div class="ticker">
  <span class="ticker-inner">
    &nbsp;&nbsp;&nbsp;&nbsp;
    SYSTEM ONLINE &nbsp;â—†&nbsp; ESP32 CONNECTED &nbsp;â—†&nbsp;
    MPU-6050 ACTIVE &nbsp;â—†&nbsp; ML MODEL LOADED &nbsp;â—†&nbsp;
    MONITORING GROUND VIBRATIONS &nbsp;â—†&nbsp;
    GENTLE_TAP &nbsp;Â·&nbsp; HAND_TAP &nbsp;Â·&nbsp; MULTI_TAP &nbsp;Â·&nbsp; MULTIHAND_TAP &nbsp;Â·&nbsp; WALK &nbsp;â—†&nbsp;
    SYSTEM ONLINE &nbsp;â—†&nbsp; ESP32 CONNECTED &nbsp;â—†&nbsp;
    MPU-6050 ACTIVE &nbsp;â—†&nbsp; ML MODEL LOADED &nbsp;â—†&nbsp;
    MONITORING GROUND VIBRATIONS &nbsp;â—†&nbsp;
  </span>
</div>

<div class="container">

  <!-- System Status -->
  <div class="card">
    <h2>// System Status</h2>
    <div id="status-metrics">
      <div class="metric">
        <span class="metric-label">Model Status</span>
        <span class="metric-value" id="model-status">Loading...</span>
      </div>
      <div class="metric">
        <span class="metric-label">Model Name</span>
        <span class="metric-value" id="model-name">â€”</span>
      </div>
      <div class="metric">
        <span class="metric-label">Features</span>
        <span class="metric-value" id="n-features">â€”</span>
      </div>
      <div class="metric">
        <span class="metric-label">Window Size</span>
        <span class="metric-value" id="window-size">â€”</span>
      </div>
      <div class="metric">
        <span class="metric-label">Total Inferences</span>
        <span class="metric-value" id="total-inferences">0</span>
      </div>
      <div class="metric">
        <span class="metric-label">Avg Latency</span>
        <span class="metric-value" id="avg-latency">â€”</span>
      </div>
    </div>
  </div>

  <!-- Latest Prediction -->
  <div class="card">
    <h2>// Latest Prediction</h2>
    <div class="prediction-box">
      <div class="pred-eyebrow">Detected Activity</div>
      <div class="pred-label" id="current-pred">â€”</div>
      <div class="pred-conf" id="current-conf">CONFIDENCE: â€”</div>
      <div class="pred-meta">
        <span>SMOOTHED: <strong id="current-smooth">â€”</strong></span>
        <span>LATENCY: <strong id="current-latency">â€”</strong></span>
      </div>
    </div>
  </div>

  <!-- Test Inference Panel -->
  <div class="card">
    <h2>// Test Inference</h2>
    <form id="test-form">
      <div class="row">
        <label>AX <input type="number" id="ax" value="-2020"></label>
        <label>AY <input type="number" id="ay" value="450"></label>
        <label>AZ <input type="number" id="az" value="16700"></label>
        <label>VS <input type="number" id="vs" value="0"></label>
      </div>
      <div class="row">
        <label>N Samples <input type="number" id="n-samples" value="60" style="width:64px"></label>
        <button type="button" onclick="sendBatch()">&#9654; SEND BATCH</button>
        <button type="button" class="danger" onclick="resetBuffer()">&#8635; RESET</button>
      </div>
    </form>
    <div id="result-box">// Click SEND BATCH to test inference.</div>
  </div>

  <!-- Class Distribution Chart -->
  <div class="card">
    <h2>// Prediction Distribution</h2>
    <div id="class-bars">
      <div class="class-bar">
        <span class="class-name">Gentle_Tap</span>
        <div class="bar-track"><div class="bar-fill" id="bar-Gentle_Tap" style="width:0%"></div></div>
        <span class="bar-count" id="cnt-Gentle_Tap">0</span>
      </div>
      <div class="class-bar">
        <span class="class-name">Hand_Tap</span>
        <div class="bar-track"><div class="bar-fill" id="bar-Hand_Tap" style="width:0%"></div></div>
        <span class="bar-count" id="cnt-Hand_Tap">0</span>
      </div>
      <div class="class-bar">
        <span class="class-name">Multi_Tap</span>
        <div class="bar-track"><div class="bar-fill" id="bar-Multi_Tap" style="width:0%"></div></div>
        <span class="bar-count" id="cnt-Multi_Tap">0</span>
      </div>
      <div class="class-bar">
        <span class="class-name">MultiHand_Tap</span>
        <div class="bar-track"><div class="bar-fill" id="bar-MultiHand_Tap" style="width:0%"></div></div>
        <span class="bar-count" id="cnt-MultiHand_Tap">0</span>
      </div>
      <div class="class-bar">
        <span class="class-name">Walk1</span>
        <div class="bar-track"><div class="bar-fill" id="bar-Walk1" style="width:0%"></div></div>
        <span class="bar-count" id="cnt-Walk1">0</span>
      </div>
    </div>
  </div>

  <!-- Inference Log -->
  <div class="card full-width">
    <h2>// Inference Log (last 50)</h2>
    <div class="log-box" id="log-box"><span style="color:#555">// Awaiting data...</span></div>
  </div>

</div>

<footer>
  GROUND VIBRATION MONITORING SYSTEM (GVMS) &nbsp;Â·&nbsp; ESP32 + MPU-6050 &nbsp;Â·&nbsp; ML: SCIKIT-LEARN
  <br><br>
  âš  NOTE: THIS SYSTEM CLASSIFIES CONTROLLED VIBRATION PATTERNS. IT DOES NOT DETECT OR PREDICT SEISMIC EVENTS.
</footer>

<script>
const CLASSES = ["Gentle_Tap","Hand_Tap","Multi_Tap","MultiHand_Tap","Walk1"];
const classCounts = {};
CLASSES.forEach(c => classCounts[c] = 0);
let totalInferences = 0;
let totalLatency = 0;

async function fetchStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    document.getElementById("model-status").innerHTML =
      d.model_ready
        ? '<span class="status-dot ok"></span>Ready'
        : '<span class="status-dot err"></span>Not Ready';
    document.getElementById("model-name").textContent   = d.model_name || "â€”";
    document.getElementById("n-features").textContent   = d.n_features || "â€”";
    document.getElementById("window-size").textContent  = d.window_size ? d.window_size + " samples" : "â€”";
  } catch(e) {}
}

async function sendBatch() {
  const ax = parseFloat(document.getElementById("ax").value);
  const ay = parseFloat(document.getElementById("ay").value);
  const az = parseFloat(document.getElementById("az").value);
  const vs = parseFloat(document.getElementById("vs").value);
  const n  = parseInt(document.getElementById("n-samples").value);

  const samples = [];
  for(let i = 0; i < n; i++) {
    samples.push({
      ax: ax + (Math.random()-0.5)*200,
      ay: ay + (Math.random()-0.5)*150,
      az: az + (Math.random()-0.5)*300,
      vibration_score: vs + Math.random()*100
    });
  }

  try {
    const r = await fetch("/api/infer/batch", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({samples})
    });
    const data = await r.json();
    const last = data.results[data.results.length - 1];
    if(last && last.ready) {
      updatePrediction(last);
      document.getElementById("result-box").textContent =
        JSON.stringify({label: last.label, confidence: last.confidence,
                        smoothed: last.smoothed_label, ms: last.inference_ms}, null, 2);
    } else {
      document.getElementById("result-box").textContent =
        "Buffer not yet full â€” send more samples.";
    }
  } catch(e) {
    document.getElementById("result-box").textContent = "Error: " + e;
  }
}

function updatePrediction(res) {
  if(!res.ready) return;
  document.getElementById("current-pred").textContent    = res.label || "â€”";
  document.getElementById("current-conf").textContent    = "CONFIDENCE: " + (res.confidence != null ? (res.confidence*100).toFixed(1)+"%" : "â€”");
  document.getElementById("current-smooth").textContent  = res.smoothed_label || "â€”";
  document.getElementById("current-latency").textContent = res.inference_ms + " ms";

  if(res.smoothed_label && CLASSES.includes(res.smoothed_label)) {
    classCounts[res.smoothed_label]++;
    totalInferences++;
    totalLatency += res.inference_ms || 0;
    document.getElementById("total-inferences").textContent = totalInferences;
    document.getElementById("avg-latency").textContent =
      (totalLatency / totalInferences).toFixed(2) + " ms";
    updateBars();
    addLogEntry(res);
  }
}

function updateBars() {
  const total = Object.values(classCounts).reduce((a,b)=>a+b,0) || 1;
  CLASSES.forEach(c => {
    const pct = (classCounts[c] / total * 100).toFixed(1);
    document.getElementById("bar-"+c).style.width = pct + "%";
    document.getElementById("cnt-"+c).textContent = classCounts[c];
  });
}

function addLogEntry(res) {
  const box = document.getElementById("log-box");
  const ts  = new Date().toLocaleTimeString();
  const div = document.createElement("div");
  div.className = "log-entry";
  div.innerHTML = `<span class="ts">[${ts}]</span> <span class="lbl">${res.label}</span> ` +
    `<span class="conf">(${res.confidence!=null?(res.confidence*100).toFixed(1)+"%":"â€”"})</span> ` +
    `â†’ smoothed: ${res.smoothed_label}  |  ${res.inference_ms}ms`;
  box.insertBefore(div, box.firstChild);
  while(box.children.length > 50) box.removeChild(box.lastChild);
}

async function resetBuffer() {
  await fetch("/api/reset", {method:"POST"});
  document.getElementById("result-box").textContent = "Buffer reset.";
}

fetchStatus();
setInterval(fetchStatus, 10000);
</script>
</body>
</html>
"""


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# API Routes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    meta = {}
    if _inference.is_ready and _inference._meta:
        meta = _inference._meta
    return jsonify({
        "model_ready":  _inference.is_ready,
        "model_name":   meta.get("model_name", "Not loaded"),
        "n_features":   len(meta.get("feature_names", [])),
        "window_size":  meta.get("window_size", 50),
        "n_classes":    meta.get("n_classes", 5),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """
    POST JSON: {"ax": float, "ay": float, "az": float, "vibration_score": float}
    Returns inference result dict.
    """
    data = request.get_json(force=True)
    try:
        ax = float(data["ax"])
        ay = float(data["ay"])
        az = float(data["az"])
        vs = float(data.get("vibration_score", 0.0))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result = _inference.infer(ax, ay, az, vs)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    if result.get("ready"):
        _history.append(result)

    return jsonify(result)


@app.route("/api/infer/batch", methods=["POST"])
def api_infer_batch():
    """
    POST JSON (two accepted formats):
      1. Named-key:  {"samples": [{"ax":..,"ay":..,"az":..,"vibration_score":..}, ...]}
      2. Compact:    {"samples": [[ax, ay, az, vs], ...]}   <-- ESP32 sends this
    """
    data = request.get_json(force=True)
    samples = data.get("samples", [])

    if not isinstance(samples, list) or len(samples) == 0:
        return jsonify({"error": "samples must be a non-empty list"}), 400

    results = []
    for s in samples:
        try:
            # Accept both compact [ax,ay,az,vs] arrays and named-key dicts
            if isinstance(s, (list, tuple)):
                ax_val = float(s[0])
                ay_val = float(s[1])
                az_val = float(s[2])
                vs_val = float(s[3]) if len(s) > 3 else 0.0
            else:
                ax_val = float(s["ax"])
                ay_val = float(s["ay"])
                az_val = float(s["az"])
                vs_val = float(s.get("vibration_score", 0.0))
        except (KeyError, IndexError, ValueError, TypeError) as e:
            results.append({"error": str(e), "ready": False})
            continue

        r = _inference.infer(ax_val, ay_val, az_val, vs_val)
        r["timestamp"] = datetime.now(timezone.utc).isoformat()
        results.append(r)
        if r.get("ready"):
            _history.append(r)

    return jsonify({"results": results, "n_processed": len(results)})


@app.route("/api/model/info")
def api_model_info():
    if not _inference.is_ready:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(_inference._meta)


@app.route("/api/history")
def api_history():
    return jsonify({"history": list(_history), "count": len(_history)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _inference.reset()
    return jsonify({"status": "ok", "message": "Rolling buffer reset."})


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    # ── Announce this server as gvms.local on the LAN via mDNS ──────────────
    # ESP32 uses ESPmDNS to resolve "gvms.local" instead of a hardcoded IP.
    # This means the IP can change freely and nothing breaks.
    def _mdns_announce():
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            ip_bytes = socket.inet_aton(local_ip)
            zc = Zeroconf()
            info = ServiceInfo(
                "_http._tcp.local.",
                "gvms._http._tcp.local.",
                addresses=[ip_bytes],
                port=5000,
                properties={b"path": b"/"},
                server="gvms.local.",
            )
            zc.register_service(info)
            log.info(f"mDNS: announced as gvms.local ({local_ip}:5000)")
        except Exception as e:
            log.warning(f"mDNS announcement failed (non-fatal): {e}")

    threading.Thread(target=_mdns_announce, daemon=True).start()
    log.info("Starting GVMS Dashboard on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

