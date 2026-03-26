import { createSignal } from 'solid-js'
import type { Snapshot } from './types'

// live snapshot store
export const [snapshots, setSnapshots] = createSignal<Snapshot[]>([])
export const [connected, setConnected] = createSignal(false)

// Current selected step index
export const [selectedIdx, setSelectedIdx] = createSignal<number | null>(null)

// Window for the chart view
export const [windowStart, setWindowStart] = createSignal(0)
export const WINDOW_SIZE = 200

// --- WebSocket live data ---

function wsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}/ws`
}

let ws: WebSocket | null = null
let reconnectTimer: number | null = null

function connect() {
  if (ws?.readyState === WebSocket.OPEN) return

  ws = new WebSocket(wsUrl())

  ws.onopen = () => {
    setConnected(true)
    // request full history on connect
    ws?.send(JSON.stringify({ type: 'subscribe', channel: 'training' }))
  }

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data)

      if (msg.type === 'snapshot') {
        // single new snapshot from live training
        setSnapshots(prev => [...prev, msg.data as Snapshot])
      } else if (msg.type === 'history') {
        // full history dump on connect
        setSnapshots(msg.data as Snapshot[])
      } else if (Array.isArray(msg)) {
        // raw array (backwards compat)
        setSnapshots(msg as Snapshot[])
      }
    } catch {}
  }

  ws.onclose = () => {
    setConnected(false)
    ws = null
    // reconnect after 3s
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = setTimeout(connect, 3000) as unknown as number
  }

  ws.onerror = () => ws?.close()
}

// also fetch initial data via HTTP (in case WS is slow or backend doesn't support WS yet)
async function fetchInitial() {
  try {
    const resp = await fetch('/api/snapshots')
    if (resp.ok) {
      const data = await resp.json()
      if (Array.isArray(data) && data.length > 0) {
        setSnapshots(data)
      }
    }
  } catch {
    // try static fallback
    try {
      const resp = await fetch('/ticks.json')
      const data = await resp.json()
      if (Array.isArray(data)) setSnapshots(data)
    } catch {}
  }
}

// init: fetch history then connect websocket
fetchInitial().then(() => connect())
