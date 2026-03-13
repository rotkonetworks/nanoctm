"""CTM tick visualization data capture and WebGPU server.

Captures internal state vectors at every tick during inference,
projects to 3D via PCA, serves to browser via websocket.

Usage:
  # Capture mode: run inference and save tick trajectories
  python -m nanochat.visualize --checkpoint PATH --prompt "Hello world"

  # Server mode: serve captured data + WebGPU frontend
  python -m nanochat.visualize --serve --port 8765
"""

import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TickCapture:
    """Captured state at a single tick for a single token."""
    tick: int
    state: np.ndarray          # [D] neuron state vector
    sync_out: np.ndarray       # [n_synch] sync readout (output)
    sync_act: np.ndarray       # [n_synch] sync readout (action)
    delta: float               # state change from previous tick
    certainty: float = 0.0     # prediction certainty at this tick
    loss: float = 0.0          # prediction loss at this tick


@dataclass
class TokenTrajectory:
    """Full thinking trajectory for one token."""
    token_idx: int
    token_str: str
    ticks: list = field(default_factory=list)  # list of TickCapture
    best_tick: int = -1        # which tick was selected
    predicted_token: str = ""


@dataclass
class SequenceCapture:
    """All trajectories for a sequence."""
    prompt: str
    tokens: list = field(default_factory=list)  # list of TokenTrajectory
    K: int = 32
    pca_components: Optional[np.ndarray] = None  # [3, D] PCA projection matrix


def capture_hook(ctm_block):
    """Install a capture hook on a CTMBlock to record states at every tick.

    Returns a dict that will be populated during forward pass:
      capture['states']  -> list of K tensors, each [BT, D]
      capture['syncs']   -> list of K tensors, each [BT, n_synch]
      capture['deltas']  -> list of K floats
    """
    capture = {'states': [], 'syncs_out': [], 'syncs_act': [], 'deltas': []}
    original_forward = ctm_block.forward

    def capturing_forward(x, dream=False, intervene=None, multi_tick=False,
                          adaptive=False, ctm_cache=None, layer_idx=None):
        # Enable state capture via a flag
        ctm_block._capture = capture
        capture['states'].clear()
        capture['syncs_out'].clear()
        capture['syncs_act'].clear()
        capture['deltas'].clear()

        result = original_forward(x, dream=dream, intervene=intervene,
                                  multi_tick=multi_tick, adaptive=adaptive,
                                  ctm_cache=ctm_cache, layer_idx=layer_idx)
        ctm_block._capture = None
        return result

    ctm_block.forward = capturing_forward
    return capture


def install_tick_capture(ctm_block):
    """Monkey-patch the tick loop to save state vectors.

    Less invasive than capture_hook — patches _after_ the state update
    inside the existing forward loop. Call before inference.
    """
    capture = {'states': [], 'deltas': [], 'syncs_out': [], 'syncs_act': []}

    original_intervene = None

    def capture_intervene(k, state, trace):
        """Called at each tick via the intervene hook."""
        capture['states'].append(state.detach().cpu())
        return None  # don't modify state

    ctm_block._viz_capture = capture
    ctm_block._viz_intervene = capture_intervene
    return capture


def states_to_3d(states, method='pca'):
    """Project list of [BT, D] state tensors to [K, BT, 3].

    Args:
        states: list of K numpy arrays, each [BT, D]
        method: 'pca' or 'umap'

    Returns:
        positions: [K, BT, 3] array
        components: [3, D] PCA components (for consistent projection)
    """
    # Stack all states: [K*BT, D]
    K = len(states)
    all_states = np.concatenate(states, axis=0)  # [K*BT, D]

    if method == 'pca':
        # Center
        mean = all_states.mean(axis=0)
        centered = all_states - mean

        # SVD for top 3 components
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:3]  # [3, D]

        # Project
        projected = centered @ components.T  # [K*BT, 3]

        # Normalize to [-1, 1] range for WebGPU
        maxval = np.abs(projected).max()
        if maxval > 0:
            projected /= maxval

        BT = states[0].shape[0]
        positions = projected.reshape(K, BT, 3)
        return positions, components

    raise ValueError(f"Unknown method: {method}")


def build_visualization_data(model, tokenizer, prompt, device='cuda', K=None):
    """Run inference on a prompt and capture full tick trajectories.

    Returns a SequenceCapture with all token trajectories projected to 3D.
    """
    import torch.nn.functional as F
    import math

    model.eval()

    # Tokenize
    token_ids = tokenizer.encode(prompt)
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    # Get CTM block
    if hasattr(model, 'ctm_block'):
        ctm = model.ctm_block
    else:
        # Find CTM block in GPT model
        ctm = None
        for block in model.transformer.h:
            if hasattr(block.mlp, 'synch_out_left'):
                ctm = block.mlp
                break
        if ctm is None:
            raise ValueError("No CTM block found in model")

    # Set up capture via intervene hook
    captured_states = []
    captured_deltas = []

    def capture_fn(k, state, trace):
        captured_states.append(state.detach().cpu().float().numpy())
        if k > 0:
            prev = captured_states[-2]
            delta = np.linalg.norm(state.detach().cpu().float().numpy() - prev, axis=-1).mean()
            captured_deltas.append(float(delta))
        else:
            captured_deltas.append(0.0)
        return None

    # Run forward pass with intervene hook
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.reset_cache()

    with torch.no_grad():
        if hasattr(model, 'ctm_block'):
            # QwenBackboneGPT
            model._viz_intervene = capture_fn
            logits = model(input_ids)
            model._viz_intervene = None
        else:
            # GPT model — use intervene parameter
            logits = model(input_ids, intervene=capture_fn)

    # captured_states: list of K arrays, each [BT, D] where BT = len(token_ids)
    K_actual = len(captured_states)
    BT = len(token_ids)

    # Compute per-tick certainty and loss from logits at each tick
    # (we only have final logits, not per-tick — would need multi_tick for that)

    # Project to 3D
    positions, pca_components = states_to_3d(captured_states)  # [K, BT, 3]

    # Build trajectories
    seq = SequenceCapture(prompt=prompt, K=K_actual)
    seq.pca_components = pca_components

    for t in range(BT):
        traj = TokenTrajectory(
            token_idx=t,
            token_str=token_strs[t],
        )
        for k in range(K_actual):
            tc = TickCapture(
                tick=k,
                state=captured_states[k][t],  # [D]
                sync_out=np.zeros(1),  # placeholder
                sync_act=np.zeros(1),
                delta=captured_deltas[k] if k < len(captured_deltas) else 0.0,
            )
            traj.ticks.append(tc)
        seq.tokens.append(traj)

    return seq


def capture_to_json(seq: SequenceCapture) -> dict:
    """Convert SequenceCapture to JSON-serializable dict for WebGPU frontend."""
    tokens = []
    for traj in seq.tokens:
        ticks = []
        for tc in traj.ticks:
            ticks.append({
                'tick': tc.tick,
                'pos': tc.state[:3].tolist() if len(tc.state) >= 3 else [0, 0, 0],
                'delta': tc.delta,
                'certainty': tc.certainty,
                'loss': tc.loss,
            })
        tokens.append({
            'idx': traj.token_idx,
            'str': traj.token_str,
            'ticks': ticks,
            'best_tick': traj.best_tick,
        })

    return {
        'prompt': seq.prompt,
        'K': seq.K,
        'n_tokens': len(seq.tokens),
        'tokens': tokens,
    }


# ---- WebGPU Frontend HTML ----

WEBGPU_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CTM Tick Visualizer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; overflow: hidden; }
  canvas { width: 100vw; height: 100vh; display: block; }
  #overlay {
    position: absolute; top: 20px; left: 20px;
    background: rgba(0,0,0,0.7); padding: 16px; border-radius: 8px;
    border: 1px solid #333; max-width: 300px;
  }
  #overlay h2 { font-size: 14px; color: #7af; margin-bottom: 8px; }
  #overlay .stat { font-size: 12px; margin: 4px 0; }
  #overlay .stat span { color: #aaa; }
  #prompt-bar {
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.8); padding: 12px 20px; border-radius: 24px;
    border: 1px solid #444; display: flex; gap: 12px; align-items: center;
  }
  #prompt-input {
    background: transparent; border: none; color: #e0e0e0; font-size: 14px;
    font-family: inherit; width: 400px; outline: none;
  }
  #prompt-btn {
    background: #7af; color: #000; border: none; padding: 6px 16px;
    border-radius: 12px; cursor: pointer; font-family: inherit; font-size: 13px;
  }
  #tick-slider {
    position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
    writing-mode: vertical-lr; height: 60vh;
  }
  #legend {
    position: absolute; top: 20px; right: 20px;
    background: rgba(0,0,0,0.7); padding: 12px; border-radius: 8px;
    border: 1px solid #333; font-size: 11px;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<canvas id="canvas"></canvas>

<div id="overlay">
  <h2>CTM Tick Visualizer</h2>
  <div class="stat">tokens: <span id="n-tokens">-</span></div>
  <div class="stat">K: <span id="k-val">-</span></div>
  <div class="stat">tick: <span id="cur-tick">all</span></div>
  <div class="stat">hovered: <span id="hover-token">-</span></div>
</div>

<div id="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#4af"></div> uncertain</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fa4"></div> certain</div>
  <div class="legend-item"><div class="legend-dot" style="background:#fff; opacity:0.3"></div> trail</div>
</div>

<div id="prompt-bar">
  <input id="prompt-input" placeholder="type a prompt and watch it think..." />
  <button id="prompt-btn">think</button>
</div>

<script>
// CTM Tick Visualizer — WebGPU + Canvas2D fallback
// Renders token trajectories through thinking-space

let data = null;
let canvas, ctx;
let rotation = { x: -0.3, y: 0.0 };
let dragging = false;
let lastMouse = { x: 0, y: 0 };
let animTick = 0;
let animating = true;
let ws = null;

function init() {
  canvas = document.getElementById('canvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  ctx = canvas.getContext('2d');

  // Mouse controls
  canvas.addEventListener('mousedown', e => { dragging = true; lastMouse = {x: e.clientX, y: e.clientY}; });
  canvas.addEventListener('mouseup', () => dragging = false);
  canvas.addEventListener('mousemove', e => {
    if (dragging) {
      rotation.y += (e.clientX - lastMouse.x) * 0.005;
      rotation.x += (e.clientY - lastMouse.y) * 0.005;
      lastMouse = {x: e.clientX, y: e.clientY};
    }
  });

  // Prompt submission
  document.getElementById('prompt-btn').addEventListener('click', submitPrompt);
  document.getElementById('prompt-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') submitPrompt();
  });

  // WebSocket
  connectWS();
  requestAnimationFrame(render);
}

function connectWS() {
  const wsUrl = `ws://${location.hostname}:${parseInt(location.port)+1}`;
  ws = new WebSocket(wsUrl);
  ws.onmessage = (e) => {
    data = JSON.parse(e.data);
    document.getElementById('n-tokens').textContent = data.n_tokens;
    document.getElementById('k-val').textContent = data.K;
    animTick = 0;
    animating = true;
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}

function submitPrompt() {
  const prompt = document.getElementById('prompt-input').value;
  if (prompt && ws && ws.readyState === 1) {
    ws.send(JSON.stringify({type: 'prompt', text: prompt}));
  }
}

function project3D(x, y, z) {
  // Rotate
  const cy = Math.cos(rotation.y), sy = Math.sin(rotation.y);
  const cx = Math.cos(rotation.x), sx = Math.sin(rotation.x);
  let x1 = x * cy - z * sy;
  let z1 = x * sy + z * cy;
  let y1 = y * cx - z1 * sx;
  let z2 = y * sx + z1 * cx;
  // Perspective
  const scale = 300 / (z2 + 4);
  return {
    px: canvas.width/2 + x1 * scale,
    py: canvas.height/2 + y1 * scale,
    depth: z2,
    scale: scale
  };
}

function certaintyColor(cert) {
  // Blue (uncertain) -> Orange (certain)
  const r = Math.floor(70 + cert * 180);
  const g = Math.floor(170 - cert * 80);
  const b = Math.floor(255 - cert * 200);
  return `rgb(${r},${g},${b})`;
}

function render() {
  if (!ctx) { requestAnimationFrame(render); return; }
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw axes
  drawAxes();

  if (!data) {
    ctx.fillStyle = '#555';
    ctx.font = '16px JetBrains Mono, monospace';
    ctx.fillText('waiting for data... type a prompt below', canvas.width/2 - 180, canvas.height/2);
    requestAnimationFrame(render);
    return;
  }

  const maxTick = animating ? Math.min(Math.floor(animTick), data.K - 1) : data.K - 1;

  // Draw all token trajectories
  for (const token of data.tokens) {
    const ticks = token.ticks;
    if (ticks.length === 0) continue;

    // Draw trail
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k <= Math.min(maxTick, ticks.length-1); k++) {
      const t = ticks[k];
      const p = project3D(t.pos[0], t.pos[1], t.pos[2]);
      if (k === 0) ctx.moveTo(p.px, p.py);
      else ctx.lineTo(p.px, p.py);
    }
    ctx.stroke();

    // Draw current position
    const curTick = Math.min(maxTick, ticks.length-1);
    const t = ticks[curTick];
    const p = project3D(t.pos[0], t.pos[1], t.pos[2]);
    const radius = 2 + t.delta * 10;

    ctx.fillStyle = certaintyColor(t.certainty);
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.arc(p.px, p.py, Math.max(2, radius), 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Label for first/last tick
    if (curTick === maxTick && maxTick === data.K - 1) {
      ctx.fillStyle = '#aaa';
      ctx.font = '9px monospace';
      ctx.fillText(token.str, p.px + 6, p.py + 3);
    }
  }

  // Animate
  if (animating && animTick < data.K) {
    animTick += 0.15;  // speed of animation
    document.getElementById('cur-tick').textContent = Math.floor(animTick);
  } else {
    animating = false;
    document.getElementById('cur-tick').textContent = 'all';
  }

  requestAnimationFrame(render);
}

function drawAxes() {
  const axes = [
    {v: [1,0,0], label: 'PC1', color: '#f44'},
    {v: [0,1,0], label: 'PC2', color: '#4f4'},
    {v: [0,0,1], label: 'PC3', color: '#44f'},
  ];
  for (const ax of axes) {
    const o = project3D(0, 0, 0);
    const e = project3D(ax.v[0]*0.8, ax.v[1]*0.8, ax.v[2]*0.8);
    ctx.strokeStyle = ax.color;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(o.px, o.py);
    ctx.lineTo(e.px, e.py);
    ctx.stroke();
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = ax.color;
    ctx.font = '10px monospace';
    ctx.fillText(ax.label, e.px + 4, e.py);
    ctx.globalAlpha = 1.0;
  }
}

window.addEventListener('resize', () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

init();
</script>
</body>
</html>"""


def serve(model, tokenizer, device='cuda', http_port=8080, ws_port=8081):
    """Serve the WebGPU visualization with live inference.

    HTTP server serves the HTML page.
    WebSocket server handles prompt submissions and streams back tick data.
    """
    import asyncio
    import http.server
    import threading

    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        return

    # HTTP server for the page
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = WEBGPU_HTML.replace(
                f"parseInt(location.port)+1",
                f"{ws_port}"
            )
            self.wfile.write(html.encode())
        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(('0.0.0.0', http_port), Handler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    print(f"HTTP server: http://localhost:{http_port}")

    # WebSocket handler
    async def handle_ws(websocket):
        print("Client connected")
        async for message in websocket:
            msg = json.loads(message)
            if msg['type'] == 'prompt':
                prompt = msg['text']
                print(f"Prompt: {prompt}")

                # Run inference and capture
                seq = build_visualization_data(model, tokenizer, prompt, device=device)
                result = capture_to_json(seq)

                # Use PCA positions instead of raw state[:3]
                if seq.pca_components is not None:
                    states = [tc.state for traj in seq.tokens for tc in traj.ticks]
                    K = seq.K
                    BT = len(seq.tokens)
                    all_states = np.array(states).reshape(BT, K, -1)

                    # Project each token's trajectory
                    for t_idx, traj in enumerate(seq.tokens):
                        token_states = all_states[t_idx]  # [K, D]
                        mean = token_states.mean(axis=0)
                        centered = token_states - mean
                        projected = centered @ seq.pca_components.T
                        maxval = np.abs(projected).max()
                        if maxval > 0:
                            projected /= maxval

                        for k, tc in enumerate(traj.ticks):
                            result['tokens'][t_idx]['ticks'][k]['pos'] = projected[k].tolist()

                await websocket.send(json.dumps(result))

    async def ws_main():
        async with websockets.serve(handle_ws, '0.0.0.0', ws_port):
            print(f"WebSocket server: ws://localhost:{ws_port}")
            await asyncio.Future()  # run forever

    asyncio.run(ws_main())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--http-port', type=int, default=8080)
    parser.add_argument('--ws-port', type=int, default=8081)
    parser.add_argument('--ctm-iterations', type=int, default=32)
    args = parser.parse_args()

    import os
    os.environ.setdefault("NANOCHAT_NO_COMPILE", "1")

    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    model = QwenBackboneGPT.from_pretrained(
        args.backbone,
        ctm_kwargs={
            "ctm_iterations": args.ctm_iterations,
            "ctm_n_synch": 512,
            "ctm_memory_length": 16,
            "ctm_memory_hidden": 32,
            "ctm_synapse_depth": 32,
        },
    )
    model = model.to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_ctm_state_dict(ckpt["ctm_state_dict"])
        print(f"Loaded: {args.checkpoint}")
        del ckpt

    model.eval()
    tokenizer = QwenTokenizer.from_pretrained(args.backbone)

    if args.serve:
        serve(model, tokenizer, device=str(device),
              http_port=args.http_port, ws_port=args.ws_port)
    elif args.prompt:
        seq = build_visualization_data(model, tokenizer, args.prompt, device=str(device))
        result = capture_to_json(seq)
        print(json.dumps(result, indent=2))
    else:
        print("Use --serve or --prompt")
