//! CTM Time-Travel Debugger
//!
//! 3D visualization of Continuous Thought Machine tick activations during training.
//! Works as native app (tailing JSONL) or WASM in browser (loading JSON).
//!
//! Controls:
//!   - Mouse drag: rotate 3D view
//!   - Scroll: zoom
//!   - Time slider: scrub through training history

mod data;
mod render;

use data::SnapshotBuffer;
use render::{Camera, ColorMode, Overlays, ViewMode};

use eframe::egui;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

struct DebuggerApp {
    buffer: Arc<Mutex<SnapshotBuffer>>,
    camera: Camera,
    color_mode: ColorMode,
    view_mode: ViewMode,
    window_size: usize,
    max_window: usize,

    // Time travel
    scrub_step: Option<u64>,
    playing: bool,
    play_speed: f32,
    last_play_tick: Instant,

    // Autoplay: scroll window through time when idle
    autoplay: bool,
    autoplay_direction: i64, // +1 forward, -1 backward
    last_interaction: Instant,
    window_end: Option<usize>, // sliding window end index (None = end of data)

    // Overlay line toggles
    show_loss_line: bool,
    show_certainty_line: bool,
    show_tok_sec: bool,
    show_lr: bool,
    show_grad_frac: bool,

    // File polling (native only)
    #[cfg(not(target_arch = "wasm32"))]
    jsonl_path: PathBuf,
    #[cfg(not(target_arch = "wasm32"))]
    file_offset: u64,
    #[cfg(not(target_arch = "wasm32"))]
    last_poll: Instant,
}

impl DebuggerApp {
    #[cfg(not(target_arch = "wasm32"))]
    fn new_native(jsonl_path: PathBuf) -> Self {
        let buffer = if jsonl_path.exists() {
            match SnapshotBuffer::load_jsonl(&jsonl_path) {
                Ok(buf) => {
                    log::info!("Loaded {} snapshots from {}", buf.len(), jsonl_path.display());
                    buf
                }
                Err(e) => {
                    log::warn!("Failed to load {}: {}", jsonl_path.display(), e);
                    SnapshotBuffer::new(10_000)
                }
            }
        } else {
            log::info!("Waiting for {}", jsonl_path.display());
            SnapshotBuffer::new(10_000)
        };

        let offset = std::fs::metadata(&jsonl_path)
            .map(|m| m.len())
            .unwrap_or(0);

        Self {
            buffer: Arc::new(Mutex::new(buffer)),
            camera: Camera::default(),
            color_mode: ColorMode::Selection,
            view_mode: ViewMode::Surface,
            window_size: 200,
            max_window: 1000,
            scrub_step: None,
            playing: false,
            play_speed: 1.0,
            last_play_tick: Instant::now(),
            autoplay: true,
            autoplay_direction: 1,
            last_interaction: Instant::now(),
            window_end: None,
            show_loss_line: true,
            show_certainty_line: false,
            show_tok_sec: false,
            show_lr: false,
            show_grad_frac: false,
            jsonl_path,
            file_offset: offset,
            last_poll: Instant::now(),
        }
    }

    fn new_with_data(json_data: &str) -> Self {
        let buffer = match SnapshotBuffer::from_json_array(json_data) {
            Ok(buf) => {
                log::info!("Loaded {} snapshots from JSON", buf.len());
                buf
            }
            Err(e) => {
                log::warn!("Failed to parse JSON: {}", e);
                SnapshotBuffer::new(10_000)
            }
        };

        let n = buffer.len();
        Self {
            buffer: Arc::new(Mutex::new(buffer)),
            camera: Camera::default(),
            color_mode: ColorMode::Selection,
            view_mode: ViewMode::Surface,
            window_size: 200,
            max_window: 1000,
            scrub_step: None,
            playing: false,
            play_speed: 1.0,
            last_play_tick: Instant::now(),
            autoplay: true,
            autoplay_direction: -1,
            last_interaction: Instant::now(),
            window_end: Some(n), // starts at end of data
            show_loss_line: true,
            show_certainty_line: false,
            show_tok_sec: false,
            show_lr: false,
            show_grad_frac: false,
            #[cfg(not(target_arch = "wasm32"))]
            jsonl_path: PathBuf::new(),
            #[cfg(not(target_arch = "wasm32"))]
            file_offset: 0,
            #[cfg(not(target_arch = "wasm32"))]
            last_poll: Instant::now(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_file(&mut self) {
        if self.last_poll.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_poll = Instant::now();

        if let Ok(mut buf) = self.buffer.lock() {
            match buf.load_new_lines(&self.jsonl_path, self.file_offset) {
                Ok(new_offset) => {
                    if new_offset > self.file_offset {
                        self.file_offset = new_offset;
                    }
                }
                Err(_) => {}
            }
        }
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_file();

        ctx.request_repaint_after(Duration::from_millis(100));

        let buf = self.buffer.lock().unwrap();
        let (step_min, step_max) = buf.step_range();
        let n_snapshots = buf.len();
        let k = buf.k();
        drop(buf);

        // Top panel
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("CTM TIME-TRAVEL DEBUGGER").strong().color(egui::Color32::from_rgb(91, 138, 245)));
                ui.separator();

                ui.label("color:");
                if ui.selectable_label(self.color_mode == ColorMode::Loss, "loss").clicked() {
                    self.color_mode = ColorMode::Loss;
                }
                if ui.selectable_label(self.color_mode == ColorMode::Selection, "selection %").clicked() {
                    self.color_mode = ColorMode::Selection;
                }
                ui.separator();

                ui.label("view:");
                if ui.selectable_label(self.view_mode == ViewMode::Surface, "3D").clicked() {
                    self.view_mode = ViewMode::Surface;
                }
                if ui.selectable_label(self.view_mode == ViewMode::Heatmap, "heatmap").clicked() {
                    self.view_mode = ViewMode::Heatmap;
                }
                ui.separator();

                ui.label("window:");
                ui.add(egui::DragValue::new(&mut self.window_size).range(10..=self.max_window).speed(10));
                ui.separator();

                ui.label("overlays:");
                ui.checkbox(&mut self.show_loss_line, "loss");
                ui.checkbox(&mut self.show_certainty_line, "certainty");
                ui.checkbox(&mut self.show_tok_sec, "tok/s");
                ui.checkbox(&mut self.show_lr, "lr");
                ui.checkbox(&mut self.show_grad_frac, "grad%");
            });
        });

        // Bottom: timeline — drag through data window
        egui::TopBottomPanel::bottom("timeline").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Play/pause autoplay
                let play_label = if self.autoplay { "||" } else { ">" };
                if ui.button(play_label).clicked() {
                    self.autoplay = !self.autoplay;
                    self.last_play_tick = Instant::now();
                    self.last_interaction = Instant::now();
                }

                // Window position slider — drag to scroll through all data
                let win_end = self.window_end.unwrap_or(n_snapshots);
                let mut pos = win_end as f32;
                let min_pos = self.window_size as f32;
                let max_pos = n_snapshots as f32;

                let buf = self.buffer.lock().unwrap();
                let first_vis_step = buf.window(self.window_size, self.window_end)
                    .next().map(|s| s.step).unwrap_or(step_min);
                let last_vis_step = buf.window(self.window_size, self.window_end)
                    .last().map(|s| s.step).unwrap_or(step_max);
                drop(buf);

                ui.label(format!("step {}", first_vis_step));
                let response = ui.add(
                    egui::Slider::new(&mut pos, min_pos..=max_pos)
                        .text("position")
                        .show_value(false),
                );
                if response.changed() || response.drag_stopped() {
                    self.window_end = Some(pos as usize);
                    self.autoplay = false;
                    self.last_interaction = Instant::now();
                }
                ui.label(format!("step {}", last_vis_step));

                ui.separator();
                ui.label("speed:");
                ui.add(egui::DragValue::new(&mut self.play_speed).range(0.1..=10.0).speed(0.1));

                // Density control: allow user to see all data at once (sparse)
                ui.separator();
                ui.label("density:");
                let mut density = self.window_size;
                if ui.add(egui::DragValue::new(&mut density).range(50..=self.max_window).speed(5).suffix(" pts")).changed() {
                    self.window_size = density;
                    self.last_interaction = Instant::now();
                }
            });

            // Autoplay: bounce window back and forth
            if self.autoplay && n_snapshots > self.window_size {
                let elapsed = self.last_play_tick.elapsed().as_secs_f32();
                if elapsed > 0.05 / self.play_speed {
                    self.last_play_tick = Instant::now();
                    let current_end = self.window_end.unwrap_or(n_snapshots) as i64;
                    let next = current_end + self.autoplay_direction;
                    if next > n_snapshots as i64 {
                        self.autoplay_direction = -1;
                        self.window_end = Some(n_snapshots);
                    } else if next < self.window_size as i64 {
                        self.autoplay_direction = 1;
                        self.window_end = Some(self.window_size);
                    } else {
                        self.window_end = Some(next as usize);
                    }
                }
            }

            // Resume autoplay after 8s idle
            if !self.autoplay && self.last_interaction.elapsed() > Duration::from_secs(8) {
                self.autoplay = true;
                self.autoplay_direction = 1;
                self.last_play_tick = Instant::now();
            }
        });

        // Right panel: stats
        egui::SidePanel::right("stats").min_width(180.0).show(ctx, |ui| {
            ui.heading("Stats");
            ui.separator();

            let buf = self.buffer.lock().unwrap();

            ui.label(format!("snapshots: {n_snapshots}"));
            ui.label(format!("K: {k}"));
            ui.label(format!("steps: {step_min} - {step_max}"));

            if let Some(last) = buf.last() {
                ui.separator();
                ui.label(format!("latest step: {}", last.step));
                ui.label(format!("loss: {:.4}", last.loss));
                ui.label(format!("certainty: {:.3}", last.certainty_mean));

                ui.separator();
                ui.label("top ticks:");
                let mut sorted_ticks: Vec<_> = last.ticks.iter().collect();
                sorted_ticks.sort_by(|a, b| b.selected_pct.partial_cmp(&a.selected_pct).unwrap());
                for tick in sorted_ticks.iter().take(5) {
                    let bar_width = (tick.selected_pct / 30.0 * 100.0).min(100.0);
                    ui.horizontal(|ui| {
                        ui.label(format!("t{:>2}: {:>5.1}%", tick.k, tick.selected_pct));
                        let (rect, _) = ui.allocate_exact_size(
                            egui::Vec2::new(bar_width as f32, 12.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().rect_filled(
                            rect,
                            2.0,
                            render::value_to_color(tick.selected_pct, ColorMode::Selection),
                        );
                    });
                }
            }

            // Show stats for the last visible snapshot in the current window
            if let Some(snap) = buf.window(self.window_size, self.window_end).last() {
                ui.separator();
                ui.heading(format!("@ step {}", snap.step));
                ui.label(format!("loss: {:.4}", snap.loss));
                ui.label(format!("certainty: {:.3}", snap.certainty_mean));

                ui.separator();
                ui.label("visible window ticks:");
                let mut sorted: Vec<_> = snap.ticks.iter().collect();
                sorted.sort_by(|a, b| b.selected_pct.partial_cmp(&a.selected_pct).unwrap());
                for tick in sorted.iter().take(5) {
                    let bar_width = (tick.selected_pct / 30.0 * 100.0).min(100.0);
                    ui.horizontal(|ui| {
                        ui.label(format!("t{:>2}: {:>5.1}%", tick.k, tick.selected_pct));
                        let (rect, _) = ui.allocate_exact_size(
                            egui::Vec2::new(bar_width as f32, 12.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().rect_filled(
                            rect,
                            2.0,
                            render::value_to_color(tick.selected_pct, ColorMode::Selection),
                        );
                    });
                }
            }

            drop(buf);
        });

        // Central: 3D view
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(10)))
            .show(ctx, |ui| {
                let (response, painter) = ui.allocate_painter(
                    ui.available_size(),
                    egui::Sense::click_and_drag(),
                );

                if response.dragged() {
                    let delta = response.drag_delta();
                    self.camera.rotation_y += delta.x * 0.005;
                    self.camera.rotation_x += delta.y * 0.005;
                    self.last_interaction = Instant::now();
                    self.autoplay = false;
                }

                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.zoom *= 1.0 + scroll * 0.002;
                    self.camera.zoom = self.camera.zoom.clamp(0.2, 5.0);
                    self.last_interaction = Instant::now();
                    self.autoplay = false;
                }

                // Gentle auto-rotation during autoplay
                if self.autoplay {
                    self.camera.rotation_y += 0.002;
                }

                let buf = self.buffer.lock().unwrap();

                let overlays = Overlays {
                    loss: self.show_loss_line,
                    certainty: self.show_certainty_line,
                    tok_sec: self.show_tok_sec,
                    lr: self.show_lr,
                    grad_frac: self.show_grad_frac,
                };

                match self.view_mode {
                    ViewMode::Surface | ViewMode::Trajectories => {
                        render::render_surface(
                            &painter, response.rect, &buf, &self.camera,
                            self.color_mode, self.window_size, self.scrub_step,
                            &overlays, self.window_end,
                        );
                    }
                    ViewMode::Heatmap => {
                        render::render_heatmap(
                            &painter, response.rect, &buf,
                            self.color_mode, self.window_size, self.scrub_step,
                        );
                    }
                }

                drop(buf);
            });
    }
}

// ============================================================
// Native entry point
// ============================================================
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let jsonl_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/ctm_ticks.jsonl"));

    println!("CTM Time-Travel Debugger");
    println!("  data: {}", jsonl_path.display());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("CTM Time-Travel Debugger"),
        ..Default::default()
    };

    eframe::run_native(
        "ctm-debugger",
        options,
        Box::new(move |_cc| Ok(Box::new(DebuggerApp::new_native(jsonl_path)))),
    )
    .unwrap();
}

// ============================================================
// WASM entry point
// ============================================================
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
fn setup_websocket(buffer: Arc<Mutex<SnapshotBuffer>>) {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    let window = web_sys::window().unwrap();
    let location = window.location();
    let host = location.host().unwrap_or_else(|_| "localhost".into());
    let protocol = location.protocol().unwrap_or_else(|_| "http:".into());
    let ws_protocol = if protocol == "https:" { "wss:" } else { "ws:" };
    let ws_url = format!("{}//{}/ws", ws_protocol, host);

    let ws = match web_sys::WebSocket::new(&ws_url) {
        Ok(ws) => ws,
        Err(e) => {
            log::warn!("WebSocket connect failed: {:?}", e);
            return;
        }
    };

    // On message: parse snapshot JSON and push into buffer
    let buf = buffer.clone();
    let onmessage = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MessageEvent| {
        if let Some(text) = e.data().as_string() {
            if let Ok(snap) = serde_json::from_str::<data::Snapshot>(&text) {
                if let Ok(mut b) = buf.lock() {
                    b.push(snap);
                }
            }
        }
    });
    ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    let onerror = Closure::<dyn FnMut(_)>::new(|_: web_sys::ErrorEvent| {
        log::warn!("WebSocket error");
    });
    ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
    onerror.forget();

    let onopen = Closure::<dyn FnMut()>::new(|| {
        log::info!("WebSocket connected — live data streaming");
    });
    ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
    onopen.forget();

    log::info!("WebSocket connecting to {}", ws_url);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn wasm_main() {
    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();

    // Always fetch ticks.json for historical data (skip localStorage cache for freshness)
    let json_str = {
        let resp_value = wasm_bindgen_futures::JsFuture::from(
            window.fetch_with_str("ticks.json")
        ).await.unwrap();
        let resp: web_sys::Response = resp_value.dyn_into().unwrap();
        let json_text = wasm_bindgen_futures::JsFuture::from(
            resp.text().unwrap()
        ).await.unwrap();
        json_text.as_string().unwrap_or_default()
    };

    let app = DebuggerApp::new_with_data(&json_str);
    let buffer_for_ws = app.buffer.clone();

    // Start WebSocket for real-time updates
    setup_websocket(buffer_for_ws);

    // Get the canvas element
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("ctm-canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();

    let web_options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(
            canvas,
            web_options,
            Box::new(move |_cc| Ok(Box::new(app))),
        )
        .await
        .expect("failed to start eframe");
}
