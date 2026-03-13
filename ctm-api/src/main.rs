use axum::{
    Json, Router,
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tower_http::cors::CorsLayer;
use tracing_subscriber;

/// Single tick within a training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TickData {
    k: u32,
    #[serde(default)]
    loss: f64,
    #[serde(default)]
    selected_pct: f64,
}

/// One training step snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snapshot {
    step: u64,
    loss: f64,
    #[serde(default)]
    ticks: Vec<TickData>,
    #[serde(default)]
    certainty_mean: f64,
    #[serde(default)]
    grad_tick_frac: f64,
    #[serde(default)]
    gpu_util: Option<f64>,
    #[serde(default)]
    gpu_temp: Option<f64>,
    #[serde(default)]
    gpu_power: Option<f64>,
    #[serde(default)]
    vram_used: Option<f64>,
    #[serde(default)]
    tok_per_sec: Option<f64>,
    #[serde(default)]
    lr: Option<f64>,
    #[serde(default)]
    c_proj_rank90: Option<u32>,
    #[serde(default)]
    c_proj_rank99: Option<u32>,
    #[serde(default)]
    c_proj_concentration: Option<f64>,
    #[serde(default)]
    c_proj_sigma_max: Option<f64>,
    #[serde(default)]
    c_proj_condition: Option<f64>,
}

struct AppState {
    /// All snapshots in memory (ring buffer).
    snapshots: RwLock<Vec<Snapshot>>,
    /// Broadcast channel for real-time WebSocket multicast.
    tx: broadcast::Sender<String>,
    /// Path to persist ticks.json.
    ticks_file: String,
    /// Optional Bearer token for ingest auth.
    ingest_key: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let ticks_file = std::env::var("TICKS_FILE")
        .unwrap_or_else(|_| "/usr/share/nginx/html/ticks.json".into());
    let ingest_key = std::env::var("INGEST_KEY").unwrap_or_default();
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);

    // Load existing ticks.json
    let existing: Vec<Snapshot> = std::fs::read_to_string(&ticks_file)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    tracing::info!("Loaded {} existing snapshots from {}", existing.len(), ticks_file);

    let (tx, _) = broadcast::channel::<String>(1024);

    let state = Arc::new(AppState {
        snapshots: RwLock::new(existing),
        tx,
        ticks_file,
        ingest_key,
    });

    let app = Router::new()
        .route("/api/ingest", post(handle_ingest))
        .route("/api/ticks", get(handle_ticks))
        .route("/ws", get(handle_ws))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("CTM API listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// POST /api/ingest — receive training snapshots, broadcast via WebSocket, persist.
async fn handle_ingest(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Auth check
    if !state.ingest_key.is_empty() {
        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if auth != format!("Bearer {}", state.ingest_key) {
            return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "unauthorized"})));
        }
    }

    // Accept single or batch
    let items: Vec<serde_json::Value> = if payload.is_array() {
        payload.as_array().cloned().unwrap_or_default()
    } else {
        vec![payload]
    };

    let mut count = 0u64;
    let mut snapshots = state.snapshots.write().await;

    for item in items {
        if item.get("step").is_none() || item.get("loss").is_none() {
            continue;
        }
        if let Ok(snap) = serde_json::from_value::<Snapshot>(item) {
            // Broadcast JSON to all WebSocket clients
            if let Ok(json) = serde_json::to_string(&snap) {
                let _ = state.tx.send(json);
            }
            snapshots.push(snap);
            count += 1;
        }
    }

    // Persist every 10 ingests
    if snapshots.len() % 10 < count as usize || count > 0 {
        let data = serde_json::to_string(&*snapshots).unwrap_or_default();
        let tmp = format!("{}.tmp", state.ticks_file);
        if tokio::fs::write(&tmp, &data).await.is_ok() {
            let _ = tokio::fs::rename(&tmp, &state.ticks_file).await;
        }
    }

    (StatusCode::OK, Json(serde_json::json!({"ok": true, "ingested": count})))
}

/// GET /api/ticks — return all snapshots as JSON array.
async fn handle_ticks(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snapshots = state.snapshots.read().await;
    Json(serde_json::to_value(&*snapshots).unwrap_or_default())
}

/// GET /ws — WebSocket upgrade for real-time snapshot streaming.
async fn handle_ws(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_handler(socket, state))
}

async fn ws_handler(mut socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("WebSocket client connected");
    let mut rx = state.tx.subscribe();

    loop {
        tokio::select! {
            // Broadcast new snapshots to client
            msg = rx.recv() => {
                match msg {
                    Ok(json) => {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("WebSocket client lagged by {n} messages");
                    }
                    Err(_) => break,
                }
            }
            // Handle client messages (ping/pong/close)
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        if socket.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    tracing::info!("WebSocket client disconnected");
}
