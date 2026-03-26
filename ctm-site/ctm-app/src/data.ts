import { createSignal, createResource, onCleanup } from 'solid-js'
import type { Snapshot } from './types'

const fetchSnapshots = async (): Promise<Snapshot[]> => {
  try {
    const resp = await fetch('/api/snapshots')
    if (resp.ok) return resp.json()
  } catch {}
  // fallback to static file
  const fallback = await fetch('/ticks.json')
  return fallback.json()
}

export const [snapshots, { refetch }] = createResource(fetchSnapshots)

// live refresh every 5s during training
const _refreshId = setInterval(() => refetch(), 5000)

// Current selected step index
export const [selectedIdx, setSelectedIdx] = createSignal<number | null>(null)

// Window for the chart view
export const [windowStart, setWindowStart] = createSignal(0)
export const WINDOW_SIZE = 200
