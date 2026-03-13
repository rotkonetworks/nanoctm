//! 3D rendering of tick activation landscape.
//!
//! Renders a surface/point cloud where:
//!   X = tick index (0..K-1)
//!   Y = metric value (loss or selection %)
//!   Z = training step (time axis)
//!
//! Supports mouse rotation, zoom, and time-scrubbing.

use egui::{Color32, Painter, Pos2, Rect, Stroke};
use glam::{Mat4, Vec3, Vec4};

use crate::data::{Snapshot, SnapshotBuffer};

/// Camera state for 3D projection.
pub struct Camera {
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub zoom: f32,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            rotation_x: -0.4,
            rotation_y: 0.6,
            zoom: 1.0,
            fov: 400.0,
        }
    }
}

impl Camera {
    /// Project a 3D point to 2D screen coordinates.
    pub fn project(&self, point: Vec3, center: Pos2) -> Option<(Pos2, f32)> {
        let rot_y = Mat4::from_rotation_y(self.rotation_y);
        let rot_x = Mat4::from_rotation_x(self.rotation_x);
        let rotated = rot_x * rot_y * Vec4::new(point.x, point.y, point.z, 1.0);

        let depth = rotated.z + 5.0;
        if depth < 0.1 {
            return None;
        }

        let scale = self.fov * self.zoom / depth;
        let screen_x = center.x + rotated.x * scale;
        let screen_y = center.y + rotated.y * scale;

        Some((Pos2::new(screen_x, screen_y), depth))
    }
}

/// What metric to display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMode {
    Loss,
    Selection,
}

/// Visualization mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewMode {
    Surface,
    Heatmap,
    Trajectories,
}

/// Map a value to a color.
pub fn value_to_color(val: f64, mode: ColorMode) -> Color32 {
    match mode {
        ColorMode::Loss => {
            // Low loss = green, high loss = red
            let t = ((val - 1.5) / 3.0).clamp(0.0, 1.0) as f32;
            Color32::from_rgb(
                (50.0 + t * 205.0) as u8,
                (220.0 - t * 180.0) as u8,
                50,
            )
        }
        ColorMode::Selection => {
            // 0% = dark, high% = bright cyan
            let t = (val / 25.0).clamp(0.0, 1.0) as f32;
            Color32::from_rgb(
                (20.0 + t * 40.0) as u8,
                (40.0 + t * 200.0) as u8,
                (80.0 + t * 175.0) as u8,
            )
        }
    }
}

/// Certainty to color (blue=uncertain, orange=certain).
pub fn certainty_color(cert: f64) -> Color32 {
    let t = cert.clamp(0.0, 1.0) as f32;
    Color32::from_rgb(
        (70.0 + t * 180.0) as u8,
        (170.0 - t * 80.0) as u8,
        (255.0 - t * 200.0) as u8,
    )
}

/// Which metric overlays to render as 3D lines.
pub struct Overlays {
    pub loss: bool,
    pub certainty: bool,
    pub tok_sec: bool,
    pub lr: bool,
    pub grad_frac: bool,
}

/// Render the 3D surface view.
pub fn render_surface(
    painter: &Painter,
    rect: Rect,
    buffer: &SnapshotBuffer,
    camera: &Camera,
    color_mode: ColorMode,
    window_size: usize,
    highlight_step: Option<u64>,
    overlays: &Overlays,
    window_end: Option<usize>,
) {
    if buffer.is_empty() {
        return;
    }

    let center = rect.center();
    let visible: Vec<&Snapshot> = buffer.window(window_size, window_end).collect();
    let n = visible.len();
    if n < 2 {
        return;
    }
    let k = buffer.k();

    // Draw axes
    let axes = [
        (Vec3::new(2.0, 0.0, 0.0), Color32::from_rgb(255, 80, 80), "tick"),
        (Vec3::new(0.0, -1.5, 0.0), Color32::from_rgb(80, 255, 80), "value"),
        (Vec3::new(0.0, 0.0, 2.0), Color32::from_rgb(80, 80, 255), "step"),
    ];
    if let Some((origin, _)) = camera.project(Vec3::ZERO, center) {
        for (dir, color, label) in &axes {
            if let Some((end, _)) = camera.project(*dir * 0.8, center) {
                painter.line_segment(
                    [origin, end],
                    Stroke::new(1.0, color.linear_multiply(0.3)),
                );
                painter.text(
                    end + egui::Vec2::new(5.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::monospace(10.0),
                    color.linear_multiply(0.6),
                );
            }
        }
    }

    // Draw surface
    for (si, snap) in visible.iter().enumerate() {
        let z = (si as f32 / n as f32) * 3.0 - 1.5;
        let is_highlighted = highlight_step.is_some_and(|hs| snap.step == hs);
        let is_last = si == n - 1;

        // Line connecting ticks
        let snap_k = snap.ticks.len();
        if snap_k < 2 {
            continue;
        }
        let mut points = Vec::with_capacity(snap_k);
        for ki in 0..snap_k {
            let tick = &snap.ticks[ki];
            let x = (ki as f32 / (snap_k - 1) as f32) * 3.0 - 1.5;
            let val = match color_mode {
                ColorMode::Loss => tick.loss,
                ColorMode::Selection => tick.selected_pct,
            };
            let y_norm = match color_mode {
                ColorMode::Loss => -(val / 5.0) as f32 * 1.5,
                ColorMode::Selection => -(val / 30.0) as f32 * 1.5,
            };

            if let Some((screen, _depth)) = camera.project(Vec3::new(x, y_norm, z), center) {
                points.push((screen, val, ki));
            }
        }

        // Draw connecting line
        if points.len() >= 2 {
            let line_alpha = if is_highlighted || is_last { 0.6 } else { 0.08 };
            let line_color = Color32::WHITE.linear_multiply(line_alpha);
            for w in points.windows(2) {
                painter.line_segment([w[0].0, w[1].0], Stroke::new(1.0, line_color));
            }
        }

        // Draw dots
        for (screen, val, _ki) in &points {
            let color = value_to_color(*val, color_mode);
            let alpha = if is_highlighted || is_last {
                1.0
            } else {
                0.3 + 0.5 * (si as f32 / n as f32)
            };
            let radius = if is_highlighted || is_last { 3.0 } else { 1.5 };
            painter.circle_filled(*screen, radius, color.linear_multiply(alpha));
        }
    }

    // Tick labels
    for ki in (0..k).step_by(4) {
        let x = (ki as f32 / (k - 1) as f32) * 3.0 - 1.5;
        if let Some((pos, _)) = camera.project(Vec3::new(x, 0.15, 1.6), center) {
            painter.text(
                pos,
                egui::Align2::CENTER_TOP,
                format!("t{ki}"),
                egui::FontId::monospace(9.0),
                Color32::from_gray(128),
            );
        }
    }

    // Step labels along the Z (time) axis
    {
        let first_step = visible.first().map(|s| s.step).unwrap_or(0);
        let last_step = visible.last().map(|s| s.step).unwrap_or(0);
        let step_range = last_step.saturating_sub(first_step).max(1);
        // Pick ~5 evenly spaced labels
        let num_labels = 5usize;
        for li in 0..=num_labels {
            let frac = li as f32 / num_labels as f32;
            let z = frac * 3.0 - 1.5;
            let step_val = first_step + (frac as f64 * step_range as f64) as u64;
            let x_pos = -1.6; // just left of the surface
            if let Some((pos, _)) = camera.project(Vec3::new(x_pos, 0.15, z), center) {
                painter.text(
                    pos,
                    egui::Align2::RIGHT_CENTER,
                    format!("{}", step_val),
                    egui::FontId::monospace(9.0),
                    Color32::from_gray(100),
                );
            }
        }
        // Arrow at the end of Z axis showing direction
        let arrow_z = 1.7;
        if let (Some((tip, _)), Some((base, _))) = (
            camera.project(Vec3::new(-1.6, 0.0, arrow_z), center),
            camera.project(Vec3::new(-1.6, 0.0, arrow_z - 0.3), center),
        ) {
            painter.line_segment(
                [base, tip],
                Stroke::new(1.5, Color32::from_rgb(80, 80, 255).linear_multiply(0.5)),
            );
            painter.text(
                tip + egui::Vec2::new(-5.0, 0.0),
                egui::Align2::RIGHT_CENTER,
                "step →",
                egui::FontId::monospace(9.0),
                Color32::from_rgb(80, 80, 255).linear_multiply(0.6),
            );
        }
    }

    // GPU metric overlays as 3D lines alongside the tick surface
    // Each metric gets its own X-lane to the right of the tick surface
    struct MetricLine {
        x_pos: f32,
        color: Color32,
        label: &'static str,
        max_val: f32,
        scientific: bool,
    }

    let mut metric_lines: Vec<(MetricLine, Vec<f32>)> = Vec::new();
    let mut lane = 0;

    if overlays.loss {
        let vals: Vec<f32> = visible.iter().map(|s| s.loss as f32).collect();
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(245, 91, 91),
            label: "loss",
            max_val: 5.0,
            scientific: false,
        }, vals));
        lane += 1;
    }
    if overlays.certainty {
        let vals: Vec<f32> = visible.iter().map(|s| s.certainty_mean as f32).collect();
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(91, 200, 245),
            label: "certainty",
            max_val: 1.0,
            scientific: false,
        }, vals));
        lane += 1;
    }
    if overlays.tok_sec {
        let vals: Vec<f32> = visible.iter().map(|s| s.tok_per_sec.unwrap_or(0.0) as f32).collect();
        let mv = vals.iter().cloned().fold(0.0_f32, f32::max).max(1.0);
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(61, 220, 132),
            label: "tok/s",
            max_val: mv,
            scientific: false,
        }, vals));
        lane += 1;
    }
    if overlays.lr {
        let vals: Vec<f32> = visible.iter().map(|s| s.lr.unwrap_or(0.0) as f32).collect();
        let mv = vals.iter().cloned().fold(0.0_f32, f32::max).max(1e-6);
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(245, 166, 35),
            label: "lr",
            max_val: mv,
            scientific: true,
        }, vals));
        lane += 1;
    }
    if overlays.grad_frac {
        let vals: Vec<f32> = visible.iter().map(|s| s.grad_tick_frac as f32).collect();
        metric_lines.push((MetricLine {
            x_pos: 1.7 + lane as f32 * 0.2,
            color: Color32::from_rgb(180, 130, 255),
            label: "grad%",
            max_val: 1.0,
            scientific: false,
        }, vals));
    }

    for (ml, vals) in &metric_lines {
        // Draw a faint vertical rail
        if let (Some((top, _)), Some((bot, _))) = (
            camera.project(Vec3::new(ml.x_pos, -1.5, -1.5), center),
            camera.project(Vec3::new(ml.x_pos, 0.0, -1.5), center),
        ) {
            painter.line_segment([top, bot], Stroke::new(0.5, ml.color.linear_multiply(0.15)));
        }

        // Draw the 3D metric line
        let mut prev_screen: Option<Pos2> = None;
        for (si, v) in vals.iter().enumerate() {
            let z = (si as f32 / n as f32) * 3.0 - 1.5;
            let y = -(v / ml.max_val).clamp(0.0, 1.0) * 1.5;
            if let Some((screen, _)) = camera.project(Vec3::new(ml.x_pos, y, z), center) {
                if let Some(prev) = prev_screen {
                    let alpha = 0.3 + 0.6 * (si as f32 / n as f32);
                    painter.line_segment([prev, screen], Stroke::new(1.5, ml.color.linear_multiply(alpha)));
                }
                prev_screen = Some(screen);
            }
        }

        // Label at the front
        if let Some((label_pos, _)) = camera.project(Vec3::new(ml.x_pos, 0.15, 1.6), center) {
            painter.text(
                label_pos,
                egui::Align2::CENTER_TOP,
                ml.label,
                egui::FontId::monospace(9.0),
                ml.color.linear_multiply(0.8),
            );
        }

        // Value label at the last point
        if let Some(last_val) = vals.last() {
            let z_last = (((n - 1) as f32) / n as f32) * 3.0 - 1.5;
            let y_last = -(last_val / ml.max_val).clamp(0.0, 1.0) * 1.5;
            if let Some((pos, _)) = camera.project(Vec3::new(ml.x_pos + 0.15, y_last, z_last), center) {
                painter.text(
                    pos,
                    egui::Align2::LEFT_CENTER,
                    if ml.scientific { format!("{:.1e}", last_val) } else { format!("{:.1}", last_val) },
                    egui::FontId::monospace(8.0),
                    ml.color,
                );
            }
        }
    }
}

/// Render the 2D heatmap view.
pub fn render_heatmap(
    painter: &Painter,
    rect: Rect,
    buffer: &SnapshotBuffer,
    color_mode: ColorMode,
    window_size: usize,
    highlight_step: Option<u64>,
) {
    if buffer.is_empty() {
        return;
    }

    let visible: Vec<&Snapshot> = buffer.tail(window_size).collect();
    let n = visible.len();
    if n == 0 {
        return;
    }
    let k = buffer.k();

    let margin = 60.0;
    let area = Rect::from_min_max(
        Pos2::new(rect.min.x + margin, rect.min.y + margin),
        Pos2::new(rect.max.x - margin, rect.max.y - margin),
    );

    let cell_w = area.width() / k as f32;
    let cell_h = area.height() / n as f32;

    for (si, snap) in visible.iter().enumerate() {
        let is_highlighted = highlight_step.is_some_and(|hs| snap.step == hs);

        let snap_k = snap.ticks.len();
        if snap_k == 0 {
            continue;
        }
        let snap_cell_w = area.width() / snap_k as f32;
        for ki in 0..snap_k {
            let tick = &snap.ticks[ki];
            let val = match color_mode {
                ColorMode::Loss => tick.loss,
                ColorMode::Selection => tick.selected_pct,
            };
            let color = value_to_color(val, color_mode);
            let alpha = if is_highlighted { 1.0 } else { 0.8 };

            let cell_rect = Rect::from_min_size(
                Pos2::new(area.min.x + ki as f32 * snap_cell_w, area.min.y + si as f32 * cell_h),
                egui::Vec2::new(snap_cell_w - 0.5, cell_h.max(1.0)),
            );
            painter.rect_filled(cell_rect, 0.0, color.linear_multiply(alpha));
        }
    }

    // Tick labels
    for ki in (0..k).step_by(4) {
        let x = area.min.x + ki as f32 * cell_w + cell_w / 2.0;
        painter.text(
            Pos2::new(x, area.min.y - 8.0),
            egui::Align2::CENTER_BOTTOM,
            format!("t{ki}"),
            egui::FontId::monospace(10.0),
            Color32::from_gray(128),
        );
    }

    // Step labels
    if let (Some(first), Some(last)) = (visible.first(), visible.last()) {
        painter.text(
            Pos2::new(area.min.x - 5.0, area.min.y),
            egui::Align2::RIGHT_TOP,
            format!("{}", first.step),
            egui::FontId::monospace(9.0),
            Color32::from_gray(100),
        );
        painter.text(
            Pos2::new(area.min.x - 5.0, area.max.y),
            egui::Align2::RIGHT_BOTTOM,
            format!("{}", last.step),
            egui::FontId::monospace(9.0),
            Color32::from_gray(100),
        );
    }
}
