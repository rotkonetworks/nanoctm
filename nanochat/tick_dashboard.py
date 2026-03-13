"""Live 3D tick dashboard — streams training tick data to browser.

Tails /tmp/ctm_ticks.jsonl (written by train_qwen_ctm.py) and pushes
updates to a WebGPU/Canvas3D browser visualization via websocket.

Usage:
  # On training machine (or locally if you rsync/ssh-forward the file):
  python -m nanochat.tick_dashboard --port 8080

  # Then open http://localhost:8080
  # Or SSH tunnel: ssh -L 8080:localhost:8080 root@<gpu-machine>

  # Remote mode — tail from remote machine:
  python -m nanochat.tick_dashboard --remote "ssh -p 59217 root@171.101.230.39" --port 8080
"""

import json
import asyncio
import http.server
import threading
import time
from pathlib import Path

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CTM Tick Dashboard — Live Training</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; overflow: hidden; }
  canvas { width: 100vw; height: 100vh; display: block; }
  #overlay {
    position: absolute; top: 16px; left: 16px;
    background: rgba(0,0,0,0.8); padding: 14px; border-radius: 8px;
    border: 1px solid #333; min-width: 220px;
  }
  #overlay h2 { font-size: 13px; color: #7af; margin-bottom: 8px; letter-spacing: 1px; }
  .stat { font-size: 11px; margin: 3px 0; }
  .stat span { color: #888; }
  .stat .val { color: #fff; }
  #controls {
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.8); padding: 10px 20px; border-radius: 20px;
    border: 1px solid #444; display: flex; gap: 16px; align-items: center; font-size: 12px;
  }
  #controls label { color: #888; }
  #controls select, #controls input { background: #222; color: #e0e0e0; border: 1px solid #555;
    padding: 4px 8px; border-radius: 4px; font-family: inherit; font-size: 11px; }
</style>
</head>
<body>
<canvas id="c"></canvas>

<div id="overlay">
  <h2>CTM TICK DASHBOARD</h2>
  <div class="stat">step: <span class="val" id="step">-</span></div>
  <div class="stat">loss: <span class="val" id="loss">-</span></div>
  <div class="stat">certainty: <span class="val" id="cert">-</span></div>
  <div class="stat">snapshots: <span class="val" id="n-snaps">0</span></div>
  <div class="stat">top tick: <span class="val" id="top-tick">-</span></div>
</div>

<div id="controls">
  <label>color:</label>
  <select id="color-mode">
    <option value="loss">loss</option>
    <option value="selection">selection %</option>
  </select>
  <label>view:</label>
  <select id="view-mode">
    <option value="surface">surface</option>
    <option value="heatmap">heatmap</option>
  </select>
  <label>last N steps:</label>
  <input id="window" type="number" value="200" min="10" max="5000" step="50" />
</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });

let snapshots = [];  // [{step, loss, ticks: [{k, loss, selected_pct}], certainty_mean}]
let rotation = { x: -0.4, y: 0.6 };
let dragging = false, lastMouse = {x:0, y:0};

canvas.addEventListener('mousedown', e => { dragging = true; lastMouse = {x:e.clientX, y:e.clientY}; });
canvas.addEventListener('mouseup', () => dragging = false);
canvas.addEventListener('mousemove', e => {
  if (dragging) {
    rotation.y += (e.clientX - lastMouse.x) * 0.005;
    rotation.x += (e.clientY - lastMouse.y) * 0.005;
    lastMouse = {x:e.clientX, y:e.clientY};
  }
});

function connectWS() {
  const ws = new WebSocket(`ws://${location.hostname}:${parseInt(location.port)+1}`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'snapshot') {
      snapshots.push(msg.data);
      // Keep max 2000 snapshots in memory
      if (snapshots.length > 2000) snapshots = snapshots.slice(-2000);
    } else if (msg.type === 'history') {
      snapshots = msg.data;
    }
    updateStats();
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}

function updateStats() {
  if (snapshots.length === 0) return;
  const last = snapshots[snapshots.length - 1];
  document.getElementById('step').textContent = last.step;
  document.getElementById('loss').textContent = last.loss.toFixed(4);
  document.getElementById('cert').textContent = (last.certainty_mean || 0).toFixed(3);
  document.getElementById('n-snaps').textContent = snapshots.length;
  // Find top tick
  const topTick = last.ticks.reduce((a, b) => a.selected_pct > b.selected_pct ? a : b);
  document.getElementById('top-tick').textContent = `t${topTick.k} (${topTick.selected_pct.toFixed(1)}%)`;
}

function project3D(x, y, z) {
  const cy = Math.cos(rotation.y), sy = Math.sin(rotation.y);
  const cx = Math.cos(rotation.x), sx = Math.sin(rotation.x);
  let x1 = x * cy - z * sy;
  let z1 = x * sy + z * cy;
  let y1 = y * cx - z1 * sx;
  let z2 = y * sx + z1 * cx;
  const fov = 400;
  const scale = fov / (z2 + 5);
  return { px: canvas.width/2 + x1 * scale, py: canvas.height/2 + y1 * scale, depth: z2 };
}

function valToColor(val, mode) {
  if (mode === 'loss') {
    // Low loss = green, high loss = red
    const t = Math.min(1, Math.max(0, (val - 1.5) / 3.0));
    const r = Math.floor(50 + t * 205);
    const g = Math.floor(220 - t * 180);
    const b = Math.floor(50);
    return `rgb(${r},${g},${b})`;
  } else {
    // Selection: 0% = dark, high% = bright cyan
    const t = Math.min(1, val / 25);
    const r = Math.floor(20 + t * 40);
    const g = Math.floor(40 + t * 200);
    const b = Math.floor(80 + t * 175);
    return `rgb(${r},${g},${b})`;
  }
}

function render() {
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const colorMode = document.getElementById('color-mode').value;
  const viewMode = document.getElementById('view-mode').value;
  const windowSize = parseInt(document.getElementById('window').value) || 200;

  if (snapshots.length < 2) {
    ctx.fillStyle = '#555';
    ctx.font = '14px monospace';
    ctx.fillText('waiting for tick data...', canvas.width/2 - 100, canvas.height/2);
    requestAnimationFrame(render);
    return;
  }

  // Get visible window
  const visible = snapshots.slice(-windowSize);
  const K = visible[0].ticks.length;
  const N = visible.length;

  // Draw axes
  const o = project3D(0, 0, 0);
  const labels = [
    {v: [2, 0, 0], label: `tick (0-${K-1})`, color: '#f55'},
    {v: [0, -1.5, 0], label: 'value', color: '#5f5'},
    {v: [0, 0, 2], label: `step`, color: '#55f'},
  ];
  for (const ax of labels) {
    const e = project3D(ax.v[0], ax.v[1], ax.v[2]);
    ctx.strokeStyle = ax.color;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(o.px, o.py); ctx.lineTo(e.px, e.py); ctx.stroke();
    ctx.globalAlpha = 0.6;
    ctx.fillStyle = ax.color;
    ctx.font = '10px monospace';
    ctx.fillText(ax.label, e.px + 5, e.py);
    ctx.globalAlpha = 1;
  }

  if (viewMode === 'surface') {
    // 3D surface: X=tick, Y=value, Z=step
    for (let si = 0; si < N; si++) {
      const snap = visible[si];
      const z = (si / N) * 3 - 1.5;  // normalize to [-1.5, 1.5]

      // Draw line connecting ticks at this step
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let ki = 0; ki < K; ki++) {
        const tick = snap.ticks[ki];
        const x = (ki / (K-1)) * 3 - 1.5;
        const val = colorMode === 'loss' ? tick.loss : tick.selected_pct;
        const yNorm = colorMode === 'loss' ? -(val / 5) * 1.5 : -(val / 30) * 1.5;
        const p = project3D(x, yNorm, z);
        if (ki === 0) ctx.moveTo(p.px, p.py);
        else ctx.lineTo(p.px, p.py);
      }
      ctx.stroke();

      // Draw dots at each tick
      for (let ki = 0; ki < K; ki++) {
        const tick = snap.ticks[ki];
        const x = (ki / (K-1)) * 3 - 1.5;
        const val = colorMode === 'loss' ? tick.loss : tick.selected_pct;
        const yNorm = colorMode === 'loss' ? -(val / 5) * 1.5 : -(val / 30) * 1.5;
        const p = project3D(x, yNorm, z);

        ctx.fillStyle = valToColor(val, colorMode);
        ctx.globalAlpha = si === N-1 ? 1.0 : 0.3 + 0.5 * (si / N);
        const r = si === N-1 ? 3 : 1.5;
        ctx.beginPath(); ctx.arc(p.px, p.py, r, 0, Math.PI*2); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // Tick labels along X axis for latest step
    ctx.fillStyle = '#888';
    ctx.font = '9px monospace';
    for (let ki = 0; ki < K; ki += 4) {
      const x = (ki / (K-1)) * 3 - 1.5;
      const p = project3D(x, 0.15, 1.6);
      ctx.fillText(`t${ki}`, p.px - 6, p.py);
    }
  } else {
    // Heatmap view: X=tick, Y=step, color=value
    const cellW = Math.max(4, Math.floor(canvas.width * 0.6 / K));
    const cellH = Math.max(1, Math.floor(canvas.height * 0.7 / N));
    const offX = canvas.width * 0.2;
    const offY = canvas.height * 0.1;

    for (let si = 0; si < N; si++) {
      const snap = visible[si];
      for (let ki = 0; ki < K; ki++) {
        const tick = snap.ticks[ki];
        const val = colorMode === 'loss' ? tick.loss : tick.selected_pct;
        ctx.fillStyle = valToColor(val, colorMode);
        ctx.fillRect(offX + ki * cellW, offY + si * cellH, cellW - 1, cellH);
      }
    }

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    for (let ki = 0; ki < K; ki += 4) {
      ctx.fillText(`t${ki}`, offX + ki * cellW, offY - 5);
    }
    const firstStep = visible[0].step;
    const lastStep = visible[N-1].step;
    ctx.fillText(`step ${firstStep}`, offX - 60, offY + 10);
    ctx.fillText(`step ${lastStep}`, offX - 60, offY + N * cellH);
  }

  requestAnimationFrame(render);
}

connectWS();
requestAnimationFrame(render);
</script>
</body>
</html>"""


def serve_dashboard(jsonl_path='/tmp/ctm_ticks.jsonl', http_port=8080, ws_port=8081,
                    remote_cmd=None):
    """Serve the live tick dashboard.

    Args:
        jsonl_path: path to ctm_ticks.jsonl (on local machine)
        http_port: HTTP server port
        ws_port: WebSocket server port
        remote_cmd: if set, tail from remote machine via SSH
    """
    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        return

    clients = set()
    history = []

    # HTTP server
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(('0.0.0.0', http_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Dashboard: http://localhost:{http_port}")

    # Load existing data
    path = Path(jsonl_path)
    if path.exists() and not remote_cmd:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"Loaded {len(history)} existing snapshots")

    # File tailer (local or remote)
    async def tail_file():
        if remote_cmd:
            # Remote tail via SSH
            import subprocess
            cmd = f"{remote_cmd} 'tail -f {jsonl_path}'"
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            while True:
                line = await proc.stdout.readline()
                if not line:
                    await asyncio.sleep(1)
                    continue
                line = line.decode().strip()
                if line:
                    try:
                        snap = json.loads(line)
                        history.append(snap)
                        msg = json.dumps({'type': 'snapshot', 'data': snap})
                        for ws in list(clients):
                            try:
                                await ws.send(msg)
                            except:
                                clients.discard(ws)
                    except json.JSONDecodeError:
                        pass
        else:
            # Local tail
            last_pos = path.stat().st_size if path.exists() else 0
            while True:
                await asyncio.sleep(1)
                if not path.exists():
                    continue
                current_size = path.stat().st_size
                if current_size > last_pos:
                    with open(path) as f:
                        f.seek(last_pos)
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    snap = json.loads(line)
                                    history.append(snap)
                                    msg = json.dumps({'type': 'snapshot', 'data': snap})
                                    for ws in list(clients):
                                        try:
                                            await ws.send(msg)
                                        except:
                                            clients.discard(ws)
                                except json.JSONDecodeError:
                                    pass
                        last_pos = f.tell()

    async def handle_ws(websocket):
        clients.add(websocket)
        print(f"Client connected ({len(clients)} total)")
        # Send history
        if history:
            await websocket.send(json.dumps({'type': 'history', 'data': history[-500:]}))
        try:
            async for msg in websocket:
                pass  # no client messages expected
        finally:
            clients.discard(websocket)

    async def main():
        async with websockets.serve(handle_ws, '0.0.0.0', ws_port):
            print(f"WebSocket: ws://localhost:{ws_port}")
            await tail_file()

    asyncio.run(main())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CTM Tick Dashboard — live 3D training visualization')
    parser.add_argument('--port', type=int, default=8080, help='HTTP port')
    parser.add_argument('--ws-port', type=int, default=8081, help='WebSocket port')
    parser.add_argument('--file', default='/tmp/ctm_ticks.jsonl', help='Path to tick JSONL')
    parser.add_argument('--remote', default=None, help='SSH command for remote tail, e.g. "ssh -p 59217 root@host"')
    args = parser.parse_args()

    serve_dashboard(
        jsonl_path=args.file,
        http_port=args.port,
        ws_port=args.ws_port,
        remote_cmd=args.remote,
    )
