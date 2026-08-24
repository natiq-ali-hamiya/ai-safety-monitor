#!/usr/bin/env python3
# ============================================================
#  Presentation Dashboard v1.0
#  Flask web server — run alongside main.py for demo
#  Open http://localhost:5000 in your browser
# ============================================================
"""
FEATURES:
  • Live incident counter cards (Fight, Snatch, Weapon, Accident, Zone)
  • Incident log table (last 50 events, auto-refreshes every 5s)
  • Plate reader log table
  • Snapshot gallery (click to enlarge)
  • Chart: incident types breakdown (last 24h)
  • CSV export buttons
  • Works while main.py is running (shared SQLite DB)

INSTALL:
  pip install flask

RUN:
  python dashboard.py
  # Then open http://localhost:5000

DEMO TIP:
  Run this on a second monitor during your presentation.
  It auto-refreshes so the audience sees detections live.
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, send_file

from data_store import DataStore


app   = Flask(__name__)
store = DataStore()

# ── HTML Template (single-file, no external deps except CDN) ──

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Safety Monitor — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1d2e; padding: 16px 32px; display:flex; align-items:center; gap:16px; border-bottom: 2px solid #e53935; }
  .header h1 { font-size: 22px; font-weight: 600; color: #fff; }
  .header .badge { background: #e53935; color: #fff; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .live-dot { width:10px; height:10px; background:#4caf50; border-radius:50%; animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .main { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }
  .section-title { font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#888; margin:24px 0 12px; }
  /* Stat cards */
  .stats-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px,1fr)); gap:12px; }
  .stat-card { background:#1a1d2e; border-radius:12px; padding:16px; border:1px solid #2a2d3e; }
  .stat-card .label { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#888; margin-bottom:8px; }
  .stat-card .value { font-size:32px; font-weight:700; }
  .stat-fight   { color:#ef5350; }
  .stat-snatch  { color:#ff9800; }
  .stat-weapon  { color:#e040fb; }
  .stat-accident{ color:#ff5722; }
  .stat-zone    { color:#42a5f5; }
  .stat-plate   { color:#26c6da; }
  .stat-snap    { color:#66bb6a; }
  .stat-24h     { color:#ffd54f; }
  /* Tables */
  .table-wrap { overflow-x:auto; background:#1a1d2e; border-radius:12px; border:1px solid #2a2d3e; margin-bottom:24px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { background:#12141f; color:#888; font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:10px 14px; text-align:left; border-bottom:1px solid #2a2d3e; }
  td { padding:9px 14px; border-bottom:1px solid #1e2130; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#1e2230; }
  .badge-type { display:inline-block; padding:2px 10px; border-radius:10px; font-size:11px; font-weight:600; }
  .badge-FIGHT            { background:#7f0000; color:#ef9a9a; }
  .badge-SNATCHING        { background:#7f3000; color:#ffcc80; }
  .badge-WEAPON_DISPLAYED { background:#4a0070; color:#e040fb; }
  .badge-ACCIDENT         { background:#7f2d00; color:#ff8a65; }
  .badge-CHILD_IN_DANGER_ZONE { background:#00335c; color:#81d4fa; }
  .badge-PERSON_IN_ZONE   { background:#1b3e5c; color:#81d4fa; }
  .badge-PERSON_LYING_DOWN{ background:#002d1a; color:#80cbc4; }
  /* Gallery */
  .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; }
  .gallery-item { background:#1a1d2e; border-radius:8px; overflow:hidden; border:1px solid #2a2d3e; cursor:pointer; transition:transform .15s; }
  .gallery-item:hover { transform:scale(1.02); border-color:#ef5350; }
  .gallery-item img { width:100%; height:140px; object-fit:cover; display:block; }
  .gallery-item .gi-label { padding:8px 10px; font-size:11px; color:#aaa; }
  /* Chart */
  .chart-wrap { background:#1a1d2e; border-radius:12px; border:1px solid #2a2d3e; padding:20px; margin-bottom:24px; max-width:600px; }
  /* Buttons */
  .btn { display:inline-block; padding:8px 16px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; border:none; text-decoration:none; }
  .btn-export { background:#1e3a5f; color:#42a5f5; }
  .btn-export:hover { background:#1a4a7a; }
  .btn-row { display:flex; gap:10px; margin-bottom:16px; }
  /* Modal */
  .modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:1000; align-items:center; justify-content:center; }
  .modal.open { display:flex; }
  .modal img { max-width:90vw; max-height:90vh; border-radius:8px; }
  .modal-close { position:absolute; top:16px; right:24px; font-size:28px; color:#fff; cursor:pointer; }
  /* Last update */
  .last-update { font-size:11px; color:#555; margin-left:auto; }
</style>
</head>
<body>
<div class="header">
  <div class="live-dot"></div>
  <h1>AI Safety Monitor</h1>
  <span class="badge">LIVE</span>
  <span class="last-update" id="last-update">—</span>
</div>

<div class="main">

  <div class="section-title">Incident summary</div>
  <div class="stats-grid" id="stats-grid">
    <div class="stat-card"><div class="label">Fights</div><div class="value stat-fight" id="s-FIGHT">0</div></div>
    <div class="stat-card"><div class="label">Snatching</div><div class="value stat-snatch" id="s-SNATCHING">0</div></div>
    <div class="stat-card"><div class="label">Weapons</div><div class="value stat-weapon" id="s-WEAPON_DISPLAYED">0</div></div>
    <div class="stat-card"><div class="label">Accidents</div><div class="value stat-accident" id="s-ACCIDENT">0</div></div>
    <div class="stat-card"><div class="label">Zone breaches</div><div class="value stat-zone" id="s-PERSON_IN_ZONE">0</div></div>
    <div class="stat-card"><div class="label">Plates read</div><div class="value stat-plate" id="s-PLATES_READ">0</div></div>
    <div class="stat-card"><div class="label">Snapshots</div><div class="value stat-snap" id="s-SNAPSHOTS">0</div></div>
    <div class="stat-card"><div class="label">Events (24h)</div><div class="value stat-24h" id="s-INCIDENTS_24H">0</div></div>
  </div>

  <div class="section-title">Incident breakdown (24h)</div>
  <div class="chart-wrap">
    <canvas id="incidentChart" height="200"></canvas>
  </div>

  <div class="section-title">Recent incidents</div>
  <div class="btn-row">
    <a class="btn btn-export" href="/export/incidents">Export incidents CSV</a>
    <a class="btn btn-export" href="/export/plates">Export plates CSV</a>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>#</th><th>Time</th><th>Type</th><th>Confidence</th><th>Zones</th><th>Persons</th><th>Snapshot</th></tr></thead>
      <tbody id="incident-tbody"></tbody>
    </table>
  </div>

  <div class="section-title">License plates</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>#</th><th>Time</th><th>Plate</th><th>Vehicle</th><th>Confidence</th></tr></thead>
      <tbody id="plate-tbody"></tbody>
    </table>
  </div>

  <div class="section-title">Snapshot gallery</div>
  <div class="gallery" id="gallery"></div>

</div>

<!-- Modal for full-size image -->
<div class="modal" id="modal" onclick="closeModal()">
  <span class="modal-close">✕</span>
  <img id="modal-img" src="" alt="Incident snapshot">
</div>

<script>
let chart = null;

function initChart(labels, values) {
  const ctx = document.getElementById('incidentChart').getContext('2d');
  const colors = ['#ef5350','#ff9800','#e040fb','#ff5722','#42a5f5','#26c6da'];
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Events (24h)',
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color:'#2a2d3e' }, ticks: { color:'#888' } },
        y: { grid: { color:'#2a2d3e' }, ticks: { color:'#888', stepSize: 1 }, beginAtZero: true }
      }
    }
  });
}

function refresh() {
  fetch('/api/data').then(r => r.json()).then(data => {
    // Stats
    Object.entries(data.stats).forEach(([k, v]) => {
      const el = document.getElementById('s-' + k);
      if (el) el.textContent = v;
    });

    // Chart
    const chartOrder = ['FIGHT','SNATCHING','WEAPON_DISPLAYED','ACCIDENT','PERSON_IN_ZONE','PERSON_LYING_DOWN'];
    initChart(
      chartOrder.map(k => k.replace(/_/g,' ')),
      chartOrder.map(k => data.stats[k] || 0)
    );

    // Incidents table
    const tbody = document.getElementById('incident-tbody');
    tbody.innerHTML = data.incidents.map(inc => `
      <tr>
        <td>${inc.id}</td>
        <td>${inc.timestamp}</td>
        <td><span class="badge-type badge-${inc.type}">${inc.type.replace(/_/g,' ')}</span></td>
        <td>${inc.confidence ? (inc.confidence*100).toFixed(0)+'%' : '—'}</td>
        <td>${inc.zone_names ? JSON.parse(inc.zone_names).join(', ') : '—'}</td>
        <td>${inc.person_count || 0}</td>
        <td>${inc.snapshot_path ? '<a href="/snapshot?path='+encodeURIComponent(inc.snapshot_path)+'" target="_blank" style="color:#42a5f5">View</a>' : '—'}</td>
      </tr>
    `).join('');

    // Plates table
    const ptbody = document.getElementById('plate-tbody');
    ptbody.innerHTML = data.plates.map(p => `
      <tr>
        <td>${p.id}</td>
        <td>${p.timestamp}</td>
        <td style="font-family:monospace;font-weight:600;color:#26c6da;">${p.plate_text}</td>
        <td>${p.vehicle_type}</td>
        <td>${p.confidence ? (p.confidence*100).toFixed(0)+'%' : '—'}</td>
      </tr>
    `).join('');

    // Gallery
    const gallery = document.getElementById('gallery');
    gallery.innerHTML = data.gallery.map(g => `
      <div class="gallery-item" onclick="openModal('/snapshot?path=${encodeURIComponent(g.filepath)}')">
        <img src="/snapshot?path=${encodeURIComponent(g.filepath)}" alt="${g.incident_type}" loading="lazy" onerror="this.style.display='none'">
        <div class="gi-label">${g.incident_type.replace(/_/g,' ')}<br><span style="color:#666">${g.timestamp}</span></div>
      </div>
    `).join('');

    document.getElementById('last-update').textContent =
      'Updated ' + new Date().toLocaleTimeString();
  }).catch(console.error);
}

function openModal(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').classList.add('open');
}
function closeModal() {
  document.getElementById('modal').classList.remove('open');
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/data')
def api_data():
    return jsonify({
        "stats"    : store.get_stats(),
        "incidents": store.get_recent_incidents(50),
        "plates"   : store.get_recent_plates(20),
        "gallery"  : store.build_gallery_index()[:30],
    })


@app.route('/snapshot')
def snapshot():
    from flask import request, abort
    path = request.args.get('path', '')
    if not path or not os.path.isfile(path):
        abort(404)
    # Security: only serve files from the incidents folder
    real = os.path.realpath(path)
    allowed = os.path.realpath(os.path.join(os.getcwd(), 'incidents'))
    if not real.startswith(allowed):
        abort(403)
    return send_file(real)


@app.route('/export/incidents')
def export_incidents():
    path = store.export_incidents_csv()
    return send_file(path, as_attachment=True)


@app.route('/export/plates')
def export_plates():
    path = store.export_plates_csv()
    return send_file(path, as_attachment=True)


# ── Entry point ────────────────────────────────────────────

if __name__ == '__main__':
    print("\n[Dashboard] Starting on http://localhost:5000")
    print("[Dashboard] Open this in your browser during the demo.\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
