//! Canvas2D display and SVG encoding for worker-produced scenes.
//!
//! The renderer deliberately uses two DOM canvases for large line-only scenes.
//! [`CanvasScene::render_preview`] paints an opaque base containing a bounded,
//! whole-stream preview. Repeated calls to [`CanvasScene::render_exact_batch`]
//! append exact lines to a transparent overlay in bounded batches. Once the
//! overlay is complete, callers repaint the base with
//! [`CanvasScene::render_static`] so sampled lines are not permanently drawn
//! below their exact counterparts. Mixed polygon scenes stay entirely on the
//! base canvas because their fill/line ordering cannot be split safely.
//!
//! Spatial scenes retain worker-produced XYZ coordinates and project them only
//! for display. Their source widths are normalized scene-relatively, while the
//! shared emerald palette, camera-space rod/surface light, and opaque depth cue
//! are resolved identically for live Canvas and current-orbit SVG output.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{self, Write as _};

use braken_viz::targets::{
    Palette, adaptive_scene_stroke_width, adaptive_scene_svg_stroke_width,
    adaptive_spatial_scene_stroke_width, adaptive_spatial_scene_svg_stroke_width,
    spatial_lit_color_bounded, spatial_theme_default_bucket, spatial_theme_default_color,
    turtle_stroke_rgb,
};
use braken_viz::{StrokeColor, normalized_turtle_3d_width};
use js_sys::Float32Array;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::camera::{
    Camera2d, Orbit3d, ViewBounds, ViewBounds3d, ViewTransform, ViewTransform3d, ViewportSize,
    WorldPoint, WorldPoint3d,
};
use crate::theme_palette::{
    THEME_DEFAULT_COLOR_BUCKETS, theme_default_bucket, theme_default_color,
};
use crate::worker_protocol::{
    LINE_3D_TRANSFER_VALUES, LINE_TRANSFER_CHUNK_LINES, LINE_TRANSFER_VALUES,
    MORPH_TRANSFER_CHUNK_LINES, MORPH_TRANSFER_VALUES, WorkerPolygon, WorkerPolygon3d,
    WorkerRenderResult, WorkerScene, WorkerText, WorkerTextRole, WorkerTransitionResult,
    decode_morph_space, decode_stroke_color,
};

/// Maximum number of representative segments painted into the immediate base
/// preview. Sampling spans the complete stream, including its final segment.
pub(crate) const PREVIEW_LINES: usize = 4 * 1024;
/// Exact Canvas2D work performed by one animation-frame callback.
pub(crate) const EXACT_LINES_PER_BATCH: usize = 2 * 1024;
/// SVG work performed before yielding to the browser event loop.
pub(crate) const SVG_LINES_PER_BATCH: usize = 2 * 1024;

const COLOR_BUCKETS: usize = THEME_DEFAULT_COLOR_BUCKETS;
const MAX_PATH_SEGMENTS: usize = 2 * 1024;
const DEPTH_BUCKETS: usize = 256;
const SVG_SPATIAL_SIZE: f64 = 1_000.0;
const LIGHT_BACKGROUND: [u8; 3] = [242, 245, 249];
const DARK_BACKGROUND: [u8; 3] = [15, 20, 32];
const LIGHT_FOREGROUND: [u8; 3] = [38, 50, 71];
const DARK_FOREGROUND: [u8; 3] = [238, 242, 248];

/// Failures at the Worker/Canvas boundary are reported to the app rather than
/// silently replacing data or retrying an unbounded allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CanvasError {
    InvalidTransfer {
        kind: &'static str,
        expected_values: usize,
        actual_values: usize,
    },
    InvalidMetadata(String),
    ResourceExhausted {
        operation: &'static str,
        requested_items: usize,
    },
    Browser {
        operation: &'static str,
        detail: String,
    },
    Cancelled,
    IncompleteSvg {
        completed: usize,
        total: usize,
    },
}

impl fmt::Display for CanvasError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransfer {
                kind,
                expected_values,
                actual_values,
            } => write!(
                formatter,
                "render worker transferred {actual_values} {kind} values; expected {expected_values}",
            ),
            Self::InvalidMetadata(message) => formatter.write_str(message),
            Self::ResourceExhausted {
                operation,
                requested_items,
            } => write!(
                formatter,
                "not enough browser memory to {operation} for {requested_items} items",
            ),
            Self::Browser { operation, detail } => {
                write!(formatter, "could not {operation}: {detail}")
            }
            Self::Cancelled => formatter.write_str("SVG export cancelled"),
            Self::IncompleteSvg { completed, total } => write!(
                formatter,
                "SVG encoding stopped after {completed} of {total} lines",
            ),
        }
    }
}

impl Error for CanvasError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CanvasViewport {
    pub logical_width: f64,
    pub logical_height: f64,
    pub device_pixel_ratio: f64,
    pub backing_width: u32,
    pub backing_height: u32,
}

impl CanvasViewport {
    pub(crate) fn camera_viewport(self) -> ViewportSize {
        ViewportSize::new(self.logical_width, self.logical_height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RefinementProgress {
    pub completed: usize,
    pub total: usize,
    pub cancelled: bool,
    /// True only when exact lines belong on the transparent overlay.
    pub uses_exact_overlay: bool,
}

impl RefinementProgress {
    pub(crate) fn is_complete(self) -> bool {
        self.completed >= self.total
    }

    pub(crate) fn is_refining(self) -> bool {
        self.uses_exact_overlay && !self.cancelled && !self.is_complete()
    }
}

/// Dimension-explicit display scene. Keeping this tag with the accepted Worker
/// result prevents an in-flight visualizer selection from changing the controls
/// or projection of the last completed scene.
pub(crate) enum CanvasDisplayScene {
    TwoD(CanvasScene),
    ThreeD(CanvasScene3d),
}

impl CanvasDisplayScene {
    pub(crate) fn from_worker(
        result: WorkerRenderResult,
        transferred_lines: js_sys::Array,
    ) -> Result<Self, CanvasError> {
        result
            .transfer_layout()
            .map_err(CanvasError::InvalidMetadata)?;
        if matches!(&result.scene, WorkerScene::ThreeD { .. }) {
            CanvasScene3d::from_worker(result, transferred_lines).map(Self::ThreeD)
        } else {
            CanvasScene::from_worker(result, transferred_lines).map(Self::TwoD)
        }
    }

    pub(crate) const fn is_spatial(&self) -> bool {
        matches!(self, Self::ThreeD(_))
    }

    pub(crate) const fn as_2d(&self) -> Option<&CanvasScene> {
        match self {
            Self::TwoD(scene) => Some(scene),
            Self::ThreeD(_) => None,
        }
    }

    pub(crate) fn element_count(&self) -> usize {
        match self {
            Self::TwoD(scene) => scene.element_count(),
            Self::ThreeD(scene) => scene.element_count(),
        }
    }

    pub(crate) fn navigation_enabled(&self) -> bool {
        match self {
            Self::TwoD(scene) => scene.navigation_enabled(),
            Self::ThreeD(scene) => scene.navigation_enabled(),
        }
    }

    pub(crate) fn bounds_2d(&self) -> Option<ViewBounds> {
        match self {
            Self::TwoD(scene) => Some(scene.bounds()),
            Self::ThreeD(_) => None,
        }
    }

    pub(crate) fn refinement_progress(&self) -> RefinementProgress {
        match self {
            Self::TwoD(scene) => scene.refinement_progress(),
            Self::ThreeD(scene) => scene.refinement_progress(),
        }
    }

    pub(crate) fn cancel_refinement(&self) {
        match self {
            Self::TwoD(scene) => scene.cancel_refinement(),
            Self::ThreeD(scene) => scene.cancel_refinement(),
        }
    }

    pub(crate) fn restart_view_refinement(&self) {
        match self {
            Self::TwoD(scene) => scene.restart_view_refinement(),
            Self::ThreeD(scene) => scene.restart_view_refinement(),
        }
    }

    pub(crate) fn render_preview(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<CanvasViewport, CanvasError> {
        match self {
            Self::TwoD(scene) => scene.render_preview(canvas, camera, dark, line_width_scale),
            Self::ThreeD(scene) => scene.render_preview(canvas, orbit, dark, line_width_scale),
        }
    }

    pub(crate) fn render_static(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<CanvasViewport, CanvasError> {
        match self {
            Self::TwoD(scene) => scene.render_static(canvas, camera, dark, line_width_scale),
            Self::ThreeD(scene) => scene.render_static(canvas, orbit, dark, line_width_scale),
        }
    }

    pub(crate) fn render_exact_batch(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<RefinementProgress, CanvasError> {
        match self {
            Self::TwoD(scene) => scene.render_exact_batch(canvas, camera, dark, line_width_scale),
            Self::ThreeD(scene) => scene.render_exact_batch(canvas, orbit, dark, line_width_scale),
        }
    }

    pub(crate) async fn encode_svg_yielding(
        &self,
        dark: bool,
        line_width_scale: f64,
        orbit: Orbit3d,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String, CanvasError> {
        match self {
            Self::TwoD(scene) => {
                scene
                    .encode_svg_yielding(dark, line_width_scale, is_cancelled)
                    .await
            }
            Self::ThreeD(scene) => {
                scene
                    .encode_svg_yielding(dark, line_width_scale, orbit, is_cancelled)
                    .await
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SvgProgress {
    pub completed: usize,
    pub total: usize,
    pub complete: bool,
}

#[derive(Debug, Clone, Copy)]
struct SourceLine {
    start: [f32; 2],
    end: [f32; 2],
    width: f32,
    color: StrokeColor,
}

struct LineStorage {
    chunks: Box<[Float32Array]>,
    line_count: usize,
}

impl LineStorage {
    fn from_transfer(values: js_sys::Array, line_count: usize) -> Result<Self, CanvasError> {
        let chunks = validate_chunks(
            values,
            line_count,
            LINE_TRANSFER_CHUNK_LINES,
            LINE_TRANSFER_VALUES,
            "line",
        )?;
        Ok(Self { chunks, line_count })
    }

    fn len(&self) -> usize {
        self.line_count
    }

    fn get(&self, index: usize) -> SourceLine {
        debug_assert!(index < self.line_count);
        let chunk_index = index / LINE_TRANSFER_CHUNK_LINES;
        let local_line = index % LINE_TRANSFER_CHUNK_LINES;
        let offset = u32::try_from(local_line * LINE_TRANSFER_VALUES)
            .expect("a protocol line chunk offset always fits u32");
        let chunk = &self.chunks[chunk_index];
        SourceLine {
            start: [chunk.get_index(offset), chunk.get_index(offset + 1)],
            end: [chunk.get_index(offset + 2), chunk.get_index(offset + 3)],
            width: chunk.get_index(offset + 4),
            color: decode_stroke_color(chunk.get_index(offset + 5)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RenderKey {
    logical_width: u64,
    logical_height: u64,
    device_pixel_ratio: u64,
    backing_width: u32,
    backing_height: u32,
    zoom: u64,
    focus_x: u64,
    focus_y: u64,
    dark: bool,
    line_width_scale: u64,
}

impl RenderKey {
    fn new(
        viewport: CanvasViewport,
        camera: Camera2d,
        bounds: ViewBounds,
        dark: bool,
        line_width_scale: f64,
    ) -> Self {
        let camera = camera.constrained(bounds, viewport.camera_viewport());
        Self {
            logical_width: viewport.logical_width.to_bits(),
            logical_height: viewport.logical_height.to_bits(),
            device_pixel_ratio: viewport.device_pixel_ratio.to_bits(),
            backing_width: viewport.backing_width,
            backing_height: viewport.backing_height,
            zoom: camera.zoom.to_bits(),
            focus_x: camera.focus[0].to_bits(),
            focus_y: camera.focus[1].to_bits(),
            dark,
            line_width_scale: valid_line_width_scale(line_width_scale).to_bits(),
        }
    }
}

struct ExactTarget {
    canvas: HtmlCanvasElement,
    key: RenderKey,
}

/// An immutable scene backed directly by the Worker's transferable chunks.
/// Geometry is decoded lazily; construction never flattens the typed arrays.
#[allow(dead_code)] // Kept as the renderer/app metadata boundary, even before every field is shown.
pub(crate) struct CanvasScene {
    lines: LineStorage,
    polygons: Box<[WorkerPolygon]>,
    texts: Box<[WorkerText]>,
    bounds: ViewBounds,
    had_bounds: bool,
    total_line_length: Option<f64>,
    background: Option<[u8; 3]>,
    element_count: usize,
    elapsed_millis: u64,
    derivation_backend: String,
    visualization_backend: String,
    transition: Option<WorkerTransitionResult>,
    refined_lines: Cell<usize>,
    refinement_cancelled: Cell<bool>,
    exact_target: RefCell<Option<ExactTarget>>,
}

#[allow(dead_code)] // Framework-facing methods intentionally form one stable integration surface.
impl CanvasScene {
    /// Validates Worker metadata and every transferred chunk boundary while
    /// retaining the individual Float32Array objects.
    pub(crate) fn from_worker(
        result: WorkerRenderResult,
        transferred_lines: js_sys::Array,
    ) -> Result<Self, CanvasError> {
        let WorkerRenderResult {
            line_count,
            total_line_length,
            width_reference: _,
            scene,
            background,
            elapsed_millis,
            derivation_backend,
            visualization_backend,
            transition,
            ..
        } = result;
        let WorkerScene::TwoD {
            polygons,
            texts,
            bounds,
        } = scene
        else {
            return Err(CanvasError::InvalidMetadata(String::from(
                "three-dimensional Worker geometry was sent to the planar renderer",
            )));
        };
        let element_count = line_count
            .checked_add(polygons.len())
            .and_then(|count| count.checked_add(texts.len()))
            .ok_or_else(|| {
                CanvasError::InvalidMetadata(String::from(
                    "browser scene contains too many elements",
                ))
            })?;
        let line_bounds = validate_bounds(bounds, "target")?;
        let bounds = bounds_including_polygons(line_bounds, &polygons)?;
        validate_line_length(total_line_length, "target")?;
        let lines = LineStorage::from_transfer(transferred_lines, line_count)?;
        let uses_exact_overlay =
            polygons.is_empty() && texts.is_empty() && line_count > PREVIEW_LINES;
        let refined_lines = if uses_exact_overlay { 0 } else { line_count };

        Ok(Self {
            lines,
            polygons: polygons.into_boxed_slice(),
            texts: texts.into_boxed_slice(),
            bounds: bounds.unwrap_or_default(),
            had_bounds: bounds.is_some(),
            total_line_length,
            background,
            element_count,
            elapsed_millis,
            derivation_backend,
            visualization_backend,
            transition,
            refined_lines: Cell::new(refined_lines),
            refinement_cancelled: Cell::new(false),
            exact_target: RefCell::new(None),
        })
    }

    pub(crate) fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub(crate) fn element_count(&self) -> usize {
        self.element_count
    }

    pub(crate) fn bounds(&self) -> ViewBounds {
        self.bounds
    }

    pub(crate) fn background(&self) -> Option<[u8; 3]> {
        self.background
    }

    pub(crate) fn total_line_length(&self) -> Option<f64> {
        self.total_line_length
    }

    pub(crate) fn elapsed_millis(&self) -> u64 {
        self.elapsed_millis
    }

    pub(crate) fn derivation_backend(&self) -> &str {
        &self.derivation_backend
    }

    pub(crate) fn visualization_backend(&self) -> &str {
        &self.visualization_backend
    }

    pub(crate) fn transition_metadata(&self) -> Option<&WorkerTransitionResult> {
        self.transition.as_ref()
    }

    pub(crate) fn has_polygons(&self) -> bool {
        !self.polygons.is_empty()
    }

    /// Camera gestures apply to geometric turtle scenes, including filled
    /// polygons. Inspector text retains its fixed presentation semantics.
    pub(crate) fn navigation_enabled(&self) -> bool {
        planar_navigation_enabled(self.lines.len(), self.polygons.len(), self.texts.len())
    }

    pub(crate) fn uses_exact_overlay(&self) -> bool {
        self.polygons.is_empty() && self.texts.is_empty() && self.lines.len() > PREVIEW_LINES
    }

    pub(crate) fn refinement_progress(&self) -> RefinementProgress {
        RefinementProgress {
            completed: self.refined_lines.get().min(self.lines.len()),
            total: self.lines.len(),
            cancelled: self.refinement_cancelled.get(),
            uses_exact_overlay: self.uses_exact_overlay(),
        }
    }

    pub(crate) fn is_refining(&self) -> bool {
        self.refinement_progress().is_refining()
    }

    pub(crate) fn refinement_cancelled(&self) -> bool {
        self.refinement_cancelled.get()
    }

    /// Cancellation is sticky across camera changes. Any partial exact layer
    /// is cleared immediately so the complete representative preview remains.
    pub(crate) fn cancel_refinement(&self) {
        if !self.refinement_progress().is_refining() {
            return;
        }
        self.refinement_cancelled.set(true);
        self.clear_and_forget_exact_target();
    }

    /// Starts an exact pass for a settled view unless manual cancellation was
    /// requested. The sticky cancellation flag is intentionally not reset.
    pub(crate) fn restart_view_refinement(&self) {
        self.clear_and_forget_exact_target();
        if !self.refinement_cancelled.get() && self.uses_exact_overlay() {
            self.refined_lines.set(0);
        }
    }

    /// Paints background, polygons, text, and the bounded whole-stream line
    /// preview into the opaque base layer.
    pub(crate) fn render_preview(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<CanvasViewport, CanvasError> {
        self.render_base(canvas, camera, dark, line_width_scale, true)
    }

    /// Paints only the non-line base. Call this after the exact overlay becomes
    /// complete to prevent permanent preview/exact edge darkening.
    pub(crate) fn render_static(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<CanvasViewport, CanvasError> {
        self.render_base(canvas, camera, dark, line_width_scale, false)
    }

    /// General base-layer entry point. Mixed polygon scenes always include all
    /// lines here, irrespective of include_preview_lines, because their exact
    /// line overlay is deliberately disabled.
    pub(crate) fn render_base(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        dark: bool,
        line_width_scale: f64,
        include_preview_lines: bool,
    ) -> Result<CanvasViewport, CanvasError> {
        let prepared = prepare_canvas(canvas)?;
        let viewport = prepared.viewport;
        let key = RenderKey::new(viewport, camera, self.bounds, dark, line_width_scale);
        let stale_exact = self
            .exact_target
            .borrow()
            .as_ref()
            .is_some_and(|target| target.key != key);
        if stale_exact {
            self.clear_and_forget_exact_target();
            if !self.refinement_cancelled.get() && self.uses_exact_overlay() {
                self.refined_lines.set(0);
            }
        }

        let context = prepared.context;
        context.set_global_alpha(1.0);
        context.set_fill_style_str(&css_rgb(display_background(self.background, dark)));
        context.fill_rect(0.0, 0.0, viewport.logical_width, viewport.logical_height);

        let transform = ViewTransform::new(self.bounds, viewport.camera_viewport(), camera);
        self.draw_polygons(&context, &transform, dark);

        let draw_lines = include_preview_lines || !self.uses_exact_overlay();
        if draw_lines {
            let selected = if self.uses_exact_overlay() {
                self.lines.len().min(PREVIEW_LINES)
            } else {
                // Polygons and text both keep lines on the base canvas so the
                // primitive ordering remains stable. With no exact overlay to
                // fill in omitted segments, these scenes must stay complete.
                self.lines.len()
            };
            let base_width = self.base_stroke_width(&transform, line_width_scale);
            draw_source_lines(
                &context,
                &self.lines,
                &transform,
                base_width,
                dark,
                (0..selected).map(|output_index| {
                    representative_index(output_index, selected, self.lines.len())
                }),
            );
        }
        self.draw_text(&context, dark);

        if self.element_count == 0 {
            draw_empty_placeholder(&context, viewport);
        }
        Ok(viewport)
    }

    /// Clears an exact layer and appends at most 2,048 exact segments. The
    /// first call for a new canvas/view clears stale pixels automatically.
    pub(crate) fn render_exact_batch(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<RefinementProgress, CanvasError> {
        let prepared = prepare_canvas(canvas)?;
        if !self.uses_exact_overlay() {
            clear_context(&prepared.context, prepared.viewport);
            self.refined_lines.set(self.lines.len());
            return Ok(self.refinement_progress());
        }
        if self.refinement_cancelled.get() {
            clear_context(&prepared.context, prepared.viewport);
            return Ok(self.refinement_progress());
        }

        let key = RenderKey::new(
            prepared.viewport,
            camera,
            self.bounds,
            dark,
            line_width_scale,
        );
        let same_target = !prepared.resized
            && self.exact_target.borrow().as_ref().is_some_and(|target| {
                target.key == key && js_sys::Object::is(target.canvas.as_ref(), canvas.as_ref())
            });
        if !same_target {
            clear_context(&prepared.context, prepared.viewport);
            self.refined_lines.set(0);
            *self.exact_target.borrow_mut() = Some(ExactTarget {
                canvas: canvas.clone(),
                key,
            });
        }

        let start = self.refined_lines.get().min(self.lines.len());
        if start >= self.lines.len() {
            return Ok(self.refinement_progress());
        }
        if start == 0 {
            clear_context(&prepared.context, prepared.viewport);
        }
        let end = start
            .saturating_add(EXACT_LINES_PER_BATCH)
            .min(self.lines.len());
        let transform =
            ViewTransform::new(self.bounds, prepared.viewport.camera_viewport(), camera);
        let base_width = self.base_stroke_width(&transform, line_width_scale);
        draw_source_lines(
            &prepared.context,
            &self.lines,
            &transform,
            base_width,
            dark,
            start..end,
        );
        self.refined_lines.set(end);
        Ok(self.refinement_progress())
    }

    /// Creates a resumable SVG encoder. Call encode_next_batch once per task
    /// quantum, or use encode_svg_yielding for the standard browser loop.
    pub(crate) fn svg_encoder(
        &self,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<SvgEncoder<'_>, CanvasError> {
        SvgEncoder::new(self, dark, line_width_scale)
    }

    pub(crate) async fn encode_svg_yielding(
        &self,
        dark: bool,
        line_width_scale: f64,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String, CanvasError> {
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        let mut encoder = self.svg_encoder(dark, line_width_scale)?;
        while !encoder.encode_next_batch(is_cancelled)?.complete {
            yield_to_browser().await;
        }
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        encoder.finish()
    }

    fn base_stroke_width(&self, transform: &ViewTransform, line_width_scale: f64) -> f64 {
        let zoom = transform.stroke_scale() / transform.fit_scale();
        adaptive_scene_stroke_width(
            self.total_line_length,
            self.lines.len(),
            bounds_extent(self.bounds),
            transform.fit_scale(),
        ) * zoom
            * valid_line_width_scale(line_width_scale)
    }

    fn draw_polygons(
        &self,
        context: &CanvasRenderingContext2d,
        transform: &ViewTransform,
        dark: bool,
    ) {
        context.set_global_alpha(1.0);
        for polygon in &self.polygons {
            let Some(first) = polygon.vertices.first() else {
                continue;
            };
            let first = transform.project(WorldPoint::new(first[0], first[1]));
            context.begin_path();
            context.move_to(first.x, first.y);
            for point in polygon.vertices.iter().skip(1) {
                let point = transform.project(WorldPoint::new(point[0], point[1]));
                context.line_to(point.x, point.y);
            }
            context.close_path();
            let color = turtle_stroke_rgb(polygon.color.into(), target_palette(dark))
                .map(ResolvedColor::Rgb)
                .unwrap_or(ResolvedColor::Palette(COLOR_BUCKETS / 2));
            set_fill_color(context, color, dark);
            context.fill();
        }
    }

    fn draw_text(&self, context: &CanvasRenderingContext2d, dark: bool) {
        context.set_global_alpha(1.0);
        context.set_text_baseline("alphabetic");
        for text in &self.texts {
            let color = match text.role {
                WorkerTextRole::Muted => [125, 137, 154],
                WorkerTextRole::Error => [220, 80, 80],
                WorkerTextRole::Title | WorkerTextRole::Heading | WorkerTextRole::Body => {
                    themed_foreground(dark)
                }
            };
            context.set_fill_style_str(&css_rgb(color));
            let size = if text.size.is_finite() && text.size > 0.0 {
                text.size
            } else {
                16.0
            };
            context.set_font(&format!("{size}px sans-serif"));
            let _ = context.fill_text(
                &text.content,
                finite_or(text.x, 0.0),
                finite_or(text.y, 0.0),
            );
        }
    }

    fn clear_and_forget_exact_target(&self) {
        if let Some(target) = self.exact_target.borrow_mut().take() {
            let _ = clear_canvas(&target.canvas);
        }
    }
}

fn planar_navigation_enabled(line_count: usize, polygon_count: usize, text_count: usize) -> bool {
    (line_count > 0 || polygon_count > 0) && text_count == 0
}

#[derive(Debug, Clone, Copy)]
struct SourceLine3d {
    start: [f32; 3],
    end: [f32; 3],
    width: f32,
    color: StrokeColor,
}

struct LineStorage3d {
    chunks: Box<[Float32Array]>,
    line_count: usize,
}

impl LineStorage3d {
    fn from_transfer(values: js_sys::Array, line_count: usize) -> Result<Self, CanvasError> {
        let chunks = validate_chunks(
            values,
            line_count,
            LINE_TRANSFER_CHUNK_LINES,
            LINE_3D_TRANSFER_VALUES,
            "three-dimensional line",
        )?;
        Ok(Self { chunks, line_count })
    }

    fn len(&self) -> usize {
        self.line_count
    }

    fn get(&self, index: usize) -> SourceLine3d {
        debug_assert!(index < self.line_count);
        let chunk_index = index / LINE_TRANSFER_CHUNK_LINES;
        let local_line = index % LINE_TRANSFER_CHUNK_LINES;
        let offset = u32::try_from(local_line * LINE_3D_TRANSFER_VALUES)
            .expect("a protocol line chunk offset always fits u32");
        let chunk = &self.chunks[chunk_index];
        SourceLine3d {
            start: [
                chunk.get_index(offset),
                chunk.get_index(offset + 1),
                chunk.get_index(offset + 2),
            ],
            end: [
                chunk.get_index(offset + 3),
                chunk.get_index(offset + 4),
                chunk.get_index(offset + 5),
            ],
            width: chunk.get_index(offset + 6),
            color: decode_stroke_color(chunk.get_index(offset + 7)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RenderKey3d {
    logical_width: u64,
    logical_height: u64,
    device_pixel_ratio: u64,
    backing_width: u32,
    backing_height: u32,
    orientation: [u64; 4],
    dark: bool,
    line_width_scale: u64,
}

impl RenderKey3d {
    fn new(viewport: CanvasViewport, orbit: Orbit3d, dark: bool, line_width_scale: f64) -> Self {
        Self {
            logical_width: viewport.logical_width.to_bits(),
            logical_height: viewport.logical_height.to_bits(),
            device_pixel_ratio: viewport.device_pixel_ratio.to_bits(),
            backing_width: viewport.backing_width,
            backing_height: viewport.backing_height,
            orientation: orbit.components().map(f64::to_bits),
            dark,
            line_width_scale: valid_line_width_scale(line_width_scale).to_bits(),
        }
    }
}

struct SpatialExactTarget {
    canvas: HtmlCanvasElement,
    key: RenderKey3d,
    depth_buckets: Box<[Vec<usize>]>,
    scanned: usize,
    draw_bucket: usize,
    draw_offset: usize,
    drawn: usize,
}

impl SpatialExactTarget {
    fn new(canvas: HtmlCanvasElement, key: RenderKey3d) -> Self {
        let depth_buckets = (0..DEPTH_BUCKETS)
            .map(|_| Vec::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            canvas,
            key,
            depth_buckets,
            scanned: 0,
            draw_bucket: 0,
            draw_offset: 0,
            drawn: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialPrimitive {
    Line(usize),
    Polygon(usize),
}

#[derive(Debug, Clone, Copy)]
struct DepthPrimitive {
    depth: f64,
    primitive: SpatialPrimitive,
    source_order: usize,
}

/// Canvas2D representation of one true spatial scene.
///
/// Canvas2D has no depth attachment, so primitives are painted far-to-near.
/// Moving views use a bounded representative line sample. For large settled
/// line-only scenes, exact indices are classified into depth buckets in bounded
/// animation-frame quanta before the transparent exact layer is painted.
pub(crate) struct CanvasScene3d {
    lines: LineStorage3d,
    polygons: Box<[WorkerPolygon3d]>,
    bounds: ViewBounds3d,
    total_line_length: Option<f64>,
    width_reference: f64,
    background: Option<[u8; 3]>,
    element_count: usize,
    refined_lines: Cell<usize>,
    refinement_cancelled: Cell<bool>,
    exact_target: RefCell<Option<SpatialExactTarget>>,
}

impl CanvasScene3d {
    fn from_worker(
        result: WorkerRenderResult,
        transferred_lines: js_sys::Array,
    ) -> Result<Self, CanvasError> {
        let WorkerRenderResult {
            line_count,
            total_line_length,
            width_reference,
            scene,
            background,
            transition,
            ..
        } = result;
        let WorkerScene::ThreeD { polygons, bounds } = scene else {
            return Err(CanvasError::InvalidMetadata(String::from(
                "planar Worker geometry was sent to the three-dimensional renderer",
            )));
        };
        if transition.is_some() {
            return Err(CanvasError::InvalidMetadata(String::from(
                "three-dimensional Worker geometry cannot contain transition data",
            )));
        }
        let element_count = line_count.checked_add(polygons.len()).ok_or_else(|| {
            CanvasError::InvalidMetadata(String::from(
                "browser spatial scene contains too many elements",
            ))
        })?;
        let line_bounds = validate_bounds_3d(bounds, "target")?;
        let bounds = bounds_3d_including_polygons(line_bounds, &polygons)?;
        validate_line_length(total_line_length, "target")?;
        let width_reference = width_reference
            .filter(|reference| reference.is_finite() && *reference >= 1.0)
            .ok_or_else(|| {
                CanvasError::InvalidMetadata(String::from(
                    "browser Worker produced an invalid 3D width reference",
                ))
            })?;
        let lines = LineStorage3d::from_transfer(transferred_lines, line_count)?;
        let uses_exact_overlay = polygons.is_empty() && line_count > PREVIEW_LINES;

        Ok(Self {
            lines,
            polygons: polygons.into_boxed_slice(),
            bounds: bounds.unwrap_or_default(),
            total_line_length,
            width_reference,
            background,
            element_count,
            refined_lines: Cell::new(if uses_exact_overlay { 0 } else { line_count }),
            refinement_cancelled: Cell::new(false),
            exact_target: RefCell::new(None),
        })
    }

    fn element_count(&self) -> usize {
        self.element_count
    }

    fn navigation_enabled(&self) -> bool {
        self.element_count > 0
    }

    fn uses_exact_overlay(&self) -> bool {
        self.polygons.is_empty() && self.lines.len() > PREVIEW_LINES
    }

    fn refinement_progress(&self) -> RefinementProgress {
        RefinementProgress {
            completed: self.refined_lines.get().min(self.lines.len()),
            total: self.lines.len(),
            cancelled: self.refinement_cancelled.get(),
            uses_exact_overlay: self.uses_exact_overlay(),
        }
    }

    fn cancel_refinement(&self) {
        if !self.refinement_progress().is_refining() {
            return;
        }
        self.refinement_cancelled.set(true);
        self.clear_and_forget_exact_target();
    }

    fn restart_view_refinement(&self) {
        self.clear_and_forget_exact_target();
        if !self.refinement_cancelled.get() && self.uses_exact_overlay() {
            self.refined_lines.set(0);
        }
    }

    fn render_preview(
        &self,
        canvas: &HtmlCanvasElement,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<CanvasViewport, CanvasError> {
        self.render_base(canvas, orbit, dark, line_width_scale, true)
    }

    fn render_static(
        &self,
        canvas: &HtmlCanvasElement,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<CanvasViewport, CanvasError> {
        self.render_base(canvas, orbit, dark, line_width_scale, false)
    }

    fn render_base(
        &self,
        canvas: &HtmlCanvasElement,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
        include_preview_lines: bool,
    ) -> Result<CanvasViewport, CanvasError> {
        let prepared = prepare_canvas(canvas)?;
        let viewport = prepared.viewport;
        let key = RenderKey3d::new(viewport, orbit, dark, line_width_scale);
        let stale_exact = self
            .exact_target
            .borrow()
            .as_ref()
            .is_some_and(|target| target.key != key);
        if stale_exact {
            self.clear_and_forget_exact_target();
            if !self.refinement_cancelled.get() && self.uses_exact_overlay() {
                self.refined_lines.set(0);
            }
        }

        let context = prepared.context;
        context.set_global_alpha(1.0);
        let background = display_background(self.background, dark);
        context.set_fill_style_str(&css_rgb(background));
        context.fill_rect(0.0, 0.0, viewport.logical_width, viewport.logical_height);

        let transform = ViewTransform3d::new(self.bounds, viewport.camera_viewport(), orbit);
        let draw_lines = include_preview_lines || !self.uses_exact_overlay();
        let selected_lines = if draw_lines {
            if self.uses_exact_overlay() {
                self.lines.len().min(PREVIEW_LINES)
            } else {
                self.lines.len()
            }
        } else {
            0
        };
        let order =
            spatial_primitive_order(&self.lines, &self.polygons, &transform, selected_lines)?;

        let style = SpatialPaintStyle {
            base_width: self.base_stroke_width(&transform),
            width_reference: self.width_reference,
            line_width_scale: valid_line_width_scale(line_width_scale),
            background,
            palette: target_palette(dark),
        };
        draw_spatial_primitives(
            &context,
            &self.lines,
            &self.polygons,
            &transform,
            style,
            &order,
        );
        if self.element_count == 0 {
            draw_empty_placeholder(&context, viewport);
        }
        Ok(viewport)
    }

    fn render_exact_batch(
        &self,
        canvas: &HtmlCanvasElement,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f64,
    ) -> Result<RefinementProgress, CanvasError> {
        let prepared = prepare_canvas(canvas)?;
        if !self.uses_exact_overlay() {
            clear_context(&prepared.context, prepared.viewport);
            self.refined_lines.set(self.lines.len());
            return Ok(self.refinement_progress());
        }
        if self.refinement_cancelled.get() {
            clear_context(&prepared.context, prepared.viewport);
            return Ok(self.refinement_progress());
        }

        let key = RenderKey3d::new(prepared.viewport, orbit, dark, line_width_scale);
        let same_target = !prepared.resized
            && self.exact_target.borrow().as_ref().is_some_and(|target| {
                target.key == key && js_sys::Object::is(target.canvas.as_ref(), canvas.as_ref())
            });
        if !same_target {
            clear_context(&prepared.context, prepared.viewport);
            self.refined_lines.set(0);
            *self.exact_target.borrow_mut() = Some(SpatialExactTarget::new(canvas.clone(), key));
        }

        let transform =
            ViewTransform3d::new(self.bounds, prepared.viewport.camera_viewport(), orbit);
        let radius = self.bounds.fit_radius();
        {
            let mut exact = self.exact_target.borrow_mut();
            let target = exact.as_mut().expect("exact target was initialized");
            if target.scanned < self.lines.len() {
                let end = target
                    .scanned
                    .saturating_add(EXACT_LINES_PER_BATCH)
                    .min(self.lines.len());
                for index in target.scanned..end {
                    let bucket = depth_bucket(
                        spatial_line_depth(self.lines.get(index), &transform),
                        radius,
                    );
                    target.depth_buckets[bucket].try_reserve(1).map_err(|_| {
                        CanvasError::ResourceExhausted {
                            operation: "order exact spatial lines",
                            requested_items: self.lines.len(),
                        }
                    })?;
                    target.depth_buckets[bucket].push(index);
                }
                target.scanned = end;
                return Ok(self.refinement_progress());
            }
        }

        let mut indices = Vec::new();
        indices
            .try_reserve_exact(EXACT_LINES_PER_BATCH)
            .map_err(|_| CanvasError::ResourceExhausted {
                operation: "prepare an exact spatial line batch",
                requested_items: EXACT_LINES_PER_BATCH,
            })?;
        {
            let mut exact = self.exact_target.borrow_mut();
            let target = exact.as_mut().expect("exact target was initialized");
            while indices.len() < EXACT_LINES_PER_BATCH
                && target.draw_bucket < target.depth_buckets.len()
            {
                let bucket = &target.depth_buckets[target.draw_bucket];
                let remaining = EXACT_LINES_PER_BATCH - indices.len();
                let end = target
                    .draw_offset
                    .saturating_add(remaining)
                    .min(bucket.len());
                indices.extend_from_slice(&bucket[target.draw_offset..end]);
                target.draw_offset = end;
                if target.draw_offset >= bucket.len() {
                    target.draw_bucket += 1;
                    target.draw_offset = 0;
                }
            }
            target.drawn = target
                .drawn
                .saturating_add(indices.len())
                .min(self.lines.len());
            self.refined_lines.set(target.drawn);
        }

        let style = SpatialPaintStyle {
            base_width: self.base_stroke_width(&transform),
            width_reference: self.width_reference,
            line_width_scale: valid_line_width_scale(line_width_scale),
            background: display_background(self.background, dark),
            palette: target_palette(dark),
        };
        draw_spatial_lines(&prepared.context, &self.lines, &transform, style, indices);
        Ok(self.refinement_progress())
    }

    fn base_stroke_width(&self, transform: &ViewTransform3d) -> f64 {
        let diameter = self.bounds.fit_radius() * 2.0;
        adaptive_spatial_scene_stroke_width(
            self.total_line_length,
            self.lines.len(),
            (diameter, diameter),
            transform.fit_scale(),
        )
    }

    async fn encode_svg_yielding(
        &self,
        dark: bool,
        line_width_scale: f64,
        orbit: Orbit3d,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String, CanvasError> {
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        let palette = target_palette(dark);
        let background_rgb = display_background(self.background, dark);
        let background = css_rgb(background_rgb);
        let viewport = ViewportSize::new(SVG_SPATIAL_SIZE, SVG_SPATIAL_SIZE);
        let transform = ViewTransform3d::new(self.bounds, viewport, orbit);
        let diameter = self.bounds.fit_radius() * 2.0;
        let stroke_width = adaptive_spatial_scene_svg_stroke_width(
            self.total_line_length,
            self.lines.len(),
            (diameter, diameter),
            SVG_SPATIAL_SIZE,
        );
        let line_width_scale = valid_line_width_scale(line_width_scale);
        let mut output = String::new();
        output
            .try_reserve(512)
            .map_err(|_| CanvasError::ResourceExhausted {
                operation: "reserve spatial SVG header data",
                requested_items: self.element_count,
            })?;
        let _ = writeln!(
            output,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {SVG_SPATIAL_SIZE} {SVG_SPATIAL_SIZE}\"><rect width=\"{SVG_SPATIAL_SIZE}\" height=\"{SVG_SPATIAL_SIZE}\" fill=\"{background}\"/>"
        );

        let order = self
            .spatial_svg_primitive_order(&transform, is_cancelled)
            .await?;
        let mut work = 0usize;
        for entry in order {
            if is_cancelled() {
                return Err(CanvasError::Cancelled);
            }
            match entry.primitive {
                SpatialPrimitive::Line(index) => {
                    output
                        .try_reserve(320)
                        .map_err(|_| CanvasError::ResourceExhausted {
                            operation: "reserve spatial SVG line data",
                            requested_items: self.lines.len(),
                        })?;
                    let line = self.lines.get(index);
                    if let Some(width_multiplier) =
                        spatial_width_multiplier(line.width, self.width_reference)
                        && line.start.into_iter().chain(line.end).all(f32::is_finite)
                    {
                        let start = project_3d(line.start, &transform);
                        let end = project_3d(line.end, &transform);
                        let stroke =
                            css_rgb(spatial_line_rgb(line, &transform, palette, background_rgb));
                        let width = stroke_width * width_multiplier * line_width_scale;
                        let _ = writeln!(
                            output,
                            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>",
                            start[0], start[1], end[0], end[1]
                        );
                    }
                    work += 1;
                }
                SpatialPrimitive::Polygon(index) => {
                    let polygon = &self.polygons[index];
                    let requested_items = polygon.vertices.len();
                    let reserve = requested_items
                        .checked_mul(64)
                        .and_then(|size| size.checked_add(320))
                        .ok_or(CanvasError::ResourceExhausted {
                            operation: "reserve spatial SVG polygon data",
                            requested_items,
                        })?;
                    output
                        .try_reserve(reserve)
                        .map_err(|_| CanvasError::ResourceExhausted {
                            operation: "reserve spatial SVG polygon data",
                            requested_items,
                        })?;
                    output.push_str("<polygon points=\"");
                    let mut centroid = None;
                    for (vertex_index, &[x, y, z]) in polygon.vertices.iter().enumerate() {
                        if is_cancelled() {
                            return Err(CanvasError::Cancelled);
                        }
                        if vertex_index > 0 {
                            output.push(' ');
                        }
                        let world_point = WorldPoint3d::new(x, y, z);
                        include_spatial_centroid_point(&mut centroid, world_point, vertex_index);
                        let point = transform.project(world_point);
                        let _ = write!(output, "{},{}", point.x, point.y);
                        work += 1;
                        if work >= SVG_LINES_PER_BATCH {
                            work = 0;
                            yield_to_browser().await;
                            if is_cancelled() {
                                return Err(CanvasError::Cancelled);
                            }
                        }
                    }
                    let fill = css_rgb(spatial_polygon_rgb_at(
                        polygon,
                        &transform,
                        palette,
                        background_rgb,
                        centroid,
                    ));
                    let _ = writeln!(output, "\" fill=\"{fill}\" stroke=\"none\"/>");
                    if polygon.vertices.is_empty() {
                        work += 1;
                    }
                }
            }
            if work >= SVG_LINES_PER_BATCH {
                work = 0;
                yield_to_browser().await;
                if is_cancelled() {
                    return Err(CanvasError::Cancelled);
                }
            }
        }
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        output.push_str("</svg>\n");
        Ok(output)
    }

    async fn spatial_svg_primitive_order(
        &self,
        transform: &ViewTransform3d,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<DepthPrimitive>, CanvasError> {
        let primitive_count = spatial_primitive_count(self.lines.len(), self.polygons.len())?;
        let mut order = Vec::new();
        order
            .try_reserve_exact(primitive_count)
            .map_err(|_| CanvasError::ResourceExhausted {
                operation: "order spatial SVG primitives",
                requested_items: primitive_count,
            })?;

        let mut work = 0usize;
        for index in 0..self.lines.len() {
            if is_cancelled() {
                return Err(CanvasError::Cancelled);
            }
            order.push(DepthPrimitive {
                depth: spatial_line_depth(self.lines.get(index), transform),
                primitive: SpatialPrimitive::Line(index),
                source_order: spatial_line_source_position(index, &self.polygons),
            });
            work += 1;
            if work >= SVG_LINES_PER_BATCH {
                work = 0;
                yield_to_browser().await;
            }
        }
        for (index, polygon) in self.polygons.iter().enumerate() {
            if is_cancelled() {
                return Err(CanvasError::Cancelled);
            }
            let mut centroid = None;
            for (vertex_index, &[x, y, z]) in polygon.vertices.iter().enumerate() {
                include_spatial_centroid_point(
                    &mut centroid,
                    WorldPoint3d::new(x, y, z),
                    vertex_index,
                );
                work += 1;
                if work >= SVG_LINES_PER_BATCH {
                    work = 0;
                    yield_to_browser().await;
                    if is_cancelled() {
                        return Err(CanvasError::Cancelled);
                    }
                }
            }
            let depth = if let Some(centroid) = centroid {
                transform.project(centroid).depth
            } else {
                work += 1;
                0.0
            };
            order.push(DepthPrimitive {
                depth,
                primitive: SpatialPrimitive::Polygon(index),
                source_order: spatial_polygon_source_position(index, polygon),
            });
            if work >= SVG_LINES_PER_BATCH {
                work = 0;
                yield_to_browser().await;
            }
        }

        sort_spatial_primitives_yielding(&mut order, is_cancelled).await?;
        Ok(order)
    }

    fn clear_and_forget_exact_target(&self) {
        if let Some(target) = self.exact_target.borrow_mut().take() {
            let _ = clear_canvas(&target.canvas);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MorphLine {
    source_start: [f32; 2],
    source_end: [f32; 2],
    source_width: f32,
    target_start: [f32; 2],
    target_end: [f32; 2],
    target_width: f32,
    source_opacity: f32,
    target_opacity: f32,
    source_palette_start: [f32; 2],
    source_palette_end: [f32; 2],
    target_palette_start: [f32; 2],
    target_palette_end: [f32; 2],
    source_color: StrokeColor,
    target_color: StrokeColor,
    source_space_is_target: bool,
    target_space_is_target: bool,
}

struct MorphStorage {
    chunks: Box<[Float32Array]>,
    line_count: usize,
}

impl MorphStorage {
    fn from_transfer(values: js_sys::Array, line_count: usize) -> Result<Self, CanvasError> {
        let chunks = validate_chunks(
            values,
            line_count,
            MORPH_TRANSFER_CHUNK_LINES,
            MORPH_TRANSFER_VALUES,
            "morph",
        )?;
        Ok(Self { chunks, line_count })
    }

    fn get(&self, index: usize) -> MorphLine {
        debug_assert!(index < self.line_count);
        let chunk_index = index / MORPH_TRANSFER_CHUNK_LINES;
        let local_line = index % MORPH_TRANSFER_CHUNK_LINES;
        let offset = u32::try_from(local_line * MORPH_TRANSFER_VALUES)
            .expect("a protocol morph chunk offset always fits u32");
        let chunk = &self.chunks[chunk_index];
        let value = |field: u32| chunk.get_index(offset + field);
        MorphLine {
            source_start: [value(0), value(1)],
            source_end: [value(2), value(3)],
            source_width: value(4),
            target_start: [value(5), value(6)],
            target_end: [value(7), value(8)],
            target_width: value(9),
            source_opacity: value(10),
            target_opacity: value(11),
            source_palette_start: [value(12), value(13)],
            source_palette_end: [value(14), value(15)],
            target_palette_start: [value(16), value(17)],
            target_palette_end: [value(18), value(19)],
            source_color: decode_stroke_color(value(20)),
            target_color: decode_stroke_color(value(21)),
            source_space_is_target: matches!(
                decode_morph_space(value(22)),
                braken_viz::MorphCoordinateSpace::Target
            ),
            target_space_is_target: matches!(
                decode_morph_space(value(23)),
                braken_viz::MorphCoordinateSpace::Target
            ),
        }
    }
}

/// A bounded adjacent-iteration deformation backed by the exact 24-float
/// Worker protocol. Source and target geometry retain independent fitted maps.
#[allow(dead_code)] // Iteration metadata remains available for status/debug UI.
pub(crate) struct CanvasTransition {
    lines: MorphStorage,
    from_iteration: usize,
    to_iteration: usize,
    source_bounds: ViewBounds,
    target_bounds: ViewBounds,
    source_line_count: usize,
    target_line_count: usize,
    source_total_line_length: Option<f64>,
    target_total_line_length: Option<f64>,
    background: Option<[u8; 3]>,
}

#[allow(dead_code)] // Framework-facing methods intentionally form one stable integration surface.
impl CanvasTransition {
    pub(crate) fn from_worker(
        metadata: WorkerTransitionResult,
        transferred_morphs: js_sys::Array,
        target: &CanvasScene,
    ) -> Result<Self, CanvasError> {
        if metadata.from_iteration.abs_diff(metadata.to_iteration) != 1 {
            return Err(CanvasError::InvalidMetadata(String::from(
                "browser transition iterations are not adjacent",
            )));
        }
        validate_line_length(metadata.source_total_line_length, "transition source")?;
        let source_bounds =
            validate_bounds(metadata.source_bounds, "transition source")?.unwrap_or_default();
        let lines = MorphStorage::from_transfer(transferred_morphs, metadata.morph_count)?;
        Ok(Self {
            lines,
            from_iteration: metadata.from_iteration,
            to_iteration: metadata.to_iteration,
            source_bounds,
            target_bounds: target.bounds,
            source_line_count: metadata.source_line_count,
            target_line_count: target.lines.len(),
            source_total_line_length: metadata.source_total_line_length,
            target_total_line_length: target.total_line_length,
            background: target.background,
        })
    }

    pub(crate) fn start_iteration(&self) -> usize {
        self.from_iteration
    }

    pub(crate) fn end_iteration(&self) -> usize {
        self.to_iteration
    }

    pub(crate) fn line_count(&self) -> usize {
        self.lines.line_count
    }

    pub(crate) fn background(&self) -> Option<[u8; 3]> {
        self.background
    }

    /// Paints an opaque, bounded transition preview. Callers should clear or
    /// hide the ordinary exact overlay before displaying this canvas.
    pub(crate) fn render(
        &self,
        canvas: &HtmlCanvasElement,
        camera: Camera2d,
        dark: bool,
        line_width_scale: f64,
        progress: f32,
    ) -> Result<CanvasViewport, CanvasError> {
        let prepared = prepare_canvas(canvas)?;
        let viewport = prepared.viewport;
        let context = prepared.context;
        context.set_global_alpha(1.0);
        context.set_fill_style_str(&css_rgb(display_background(self.background, dark)));
        context.fill_rect(0.0, 0.0, viewport.logical_width, viewport.logical_height);

        let source_bounds = if self.source_line_count == 0 {
            self.target_bounds
        } else {
            self.source_bounds
        };
        let target_bounds = if self.target_line_count == 0 {
            self.source_bounds
        } else {
            self.target_bounds
        };
        let source_transform =
            ViewTransform::new(source_bounds, viewport.camera_viewport(), camera);
        let target_transform =
            ViewTransform::new(target_bounds, viewport.camera_viewport(), camera);
        let source_width = transition_base_width(
            self.source_total_line_length,
            self.source_line_count,
            source_bounds,
            &source_transform,
            line_width_scale,
        );
        let target_width = transition_base_width(
            self.target_total_line_length,
            self.target_line_count,
            target_bounds,
            &target_transform,
            line_width_scale,
        );
        let progress = if progress.is_finite() {
            progress.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let selected = self.lines.line_count.min(PREVIEW_LINES);
        context.set_line_cap("round");
        context.set_line_join("round");

        for output_index in 0..selected {
            let index = representative_index(output_index, selected, self.lines.line_count);
            let line = self.lines.get(index);
            let source_endpoint_transform = if line.source_space_is_target {
                &target_transform
            } else {
                &source_transform
            };
            let target_endpoint_transform = if line.target_space_is_target {
                &target_transform
            } else {
                &source_transform
            };
            let source_endpoint_width = if line.source_space_is_target {
                target_width
            } else {
                source_width
            };
            let target_endpoint_width = if line.target_space_is_target {
                target_width
            } else {
                source_width
            };
            let source_start = project(line.source_start, source_endpoint_transform);
            let source_end = project(line.source_end, source_endpoint_transform);
            let target_start = project(line.target_start, target_endpoint_transform);
            let target_end = project(line.target_end, target_endpoint_transform);
            let source_color = resolved_components(
                resolve_styled_color(
                    line.source_color,
                    line.source_palette_start,
                    line.source_palette_end,
                    source_endpoint_transform,
                    dark,
                ),
                dark,
            );
            let target_color = resolved_components(
                resolve_styled_color(
                    line.target_color,
                    line.target_palette_start,
                    line.target_palette_end,
                    target_endpoint_transform,
                    dark,
                ),
                dark,
            );
            let width = lerp(
                source_endpoint_width * positive_f32(line.source_width),
                target_endpoint_width * positive_f32(line.target_width),
                f64::from(progress),
            );
            let alpha = lerp(
                finite_or(f64::from(line.source_opacity), 0.0),
                finite_or(f64::from(line.target_opacity), 0.0),
                f64::from(progress),
            )
            .clamp(0.0, 1.0);
            if width <= 0.0 || !width.is_finite() || alpha <= 0.0 {
                continue;
            }
            let color = [
                lerp(source_color[0], target_color[0], f64::from(progress)),
                lerp(source_color[1], target_color[1], f64::from(progress)),
                lerp(source_color[2], target_color[2], f64::from(progress)),
            ];
            context.set_global_alpha(alpha);
            context.set_stroke_style_str(&css_components(color));
            context.set_line_width(width);
            context.begin_path();
            context.move_to(
                lerp(source_start[0], target_start[0], f64::from(progress)),
                lerp(source_start[1], target_start[1], f64::from(progress)),
            );
            context.line_to(
                lerp(source_end[0], target_end[0], f64::from(progress)),
                lerp(source_end[1], target_end[1], f64::from(progress)),
            );
            context.stroke();
        }
        context.set_global_alpha(1.0);
        if selected == 0 {
            draw_empty_placeholder(&context, viewport);
        }
        Ok(viewport)
    }
}

/// Incremental complete-scene SVG encoder. Polygon vertices, exact lines, and
/// text characters all consume the same bounded per-call work budget.
pub(crate) struct SvgEncoder<'a> {
    scene: &'a CanvasScene,
    palette: Palette,
    foreground: &'static str,
    muted: &'static str,
    stroke_width: f64,
    output: Option<String>,
    polygon_index: usize,
    polygon_vertex: usize,
    polygon_open: bool,
    completed: usize,
    text_index: usize,
    text_byte: usize,
    text_open: bool,
    footer_written: bool,
}

impl<'a> SvgEncoder<'a> {
    fn new(scene: &'a CanvasScene, dark: bool, line_width_scale: f64) -> Result<Self, CanvasError> {
        let palette = target_palette(dark);
        let (themed_background, foreground, muted) = if dark {
            ("#0f1420", "#eef2f8", "#aab4c3")
        } else {
            ("#f2f5f9", "#263247", "#687386")
        };
        let (view_x, view_y, view_width, view_height, drawing_extent) = svg_view(scene);
        let background = scene.background.map(rgb_hex).unwrap_or_else(|| {
            // This allocation remains small and makes ownership uniform.
            themed_background.to_owned()
        });
        let stroke_width = adaptive_scene_svg_stroke_width(
            scene.total_line_length,
            scene.lines.len(),
            drawing_extent,
            view_width.max(view_height),
        ) * valid_line_width_scale(line_width_scale);
        let mut output = String::new();
        output
            .try_reserve(512)
            .map_err(|_| CanvasError::ResourceExhausted {
                operation: "reserve SVG header data",
                requested_items: scene.element_count,
            })?;
        let _ = writeln!(
            output,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{view_x} {view_y} {view_width} {view_height}\"><rect x=\"{view_x}\" y=\"{view_y}\" width=\"{view_width}\" height=\"{view_height}\" fill=\"{background}\"/>"
        );

        Ok(Self {
            scene,
            palette,
            foreground,
            muted,
            stroke_width,
            output: Some(output),
            polygon_index: 0,
            polygon_vertex: 0,
            polygon_open: false,
            completed: 0,
            text_index: 0,
            text_byte: 0,
            text_open: false,
            footer_written: false,
        })
    }

    pub(crate) fn progress(&self) -> SvgProgress {
        SvgProgress {
            completed: self.completed.min(self.scene.lines.len()),
            total: self.scene.lines.len(),
            complete: self.footer_written,
        }
    }

    pub(crate) fn encode_next_batch(
        &mut self,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<SvgProgress, CanvasError> {
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        if self.footer_written {
            return Ok(self.progress());
        }
        let additional =
            SVG_LINES_PER_BATCH
                .checked_mul(320)
                .ok_or(CanvasError::ResourceExhausted {
                    operation: "reserve SVG batch data",
                    requested_items: self.scene.element_count,
                })?;
        self.output
            .as_mut()
            .expect("unfinished SVG owns its buffer")
            .try_reserve(additional)
            .map_err(|_| CanvasError::ResourceExhausted {
                operation: "reserve SVG batch data",
                requested_items: self.scene.element_count,
            })?;

        let mut remaining = SVG_LINES_PER_BATCH;
        while remaining > 0 && !self.footer_written {
            if is_cancelled() {
                return Err(CanvasError::Cancelled);
            }
            if self.polygon_index < self.scene.polygons.len() {
                self.encode_polygon_work(&mut remaining);
                continue;
            }
            if self.completed < self.scene.lines.len() {
                self.encode_line_work(&mut remaining);
                continue;
            }
            if self.text_index < self.scene.texts.len() {
                self.encode_text_work(&mut remaining);
                continue;
            }
            self.output
                .as_mut()
                .expect("unfinished SVG owns its buffer")
                .push_str("</svg>\n");
            self.footer_written = true;
        }
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        Ok(self.progress())
    }

    pub(crate) fn finish(mut self) -> Result<String, CanvasError> {
        if !self.footer_written {
            return Err(CanvasError::IncompleteSvg {
                completed: self.completed,
                total: self.scene.lines.len(),
            });
        }
        Ok(self.output.take().expect("finished SVG owns its buffer"))
    }

    fn encode_polygon_work(&mut self, remaining: &mut usize) {
        let polygon = &self.scene.polygons[self.polygon_index];
        let output = self
            .output
            .as_mut()
            .expect("unfinished SVG owns its buffer");
        if !self.polygon_open {
            let _ = write!(output, "<polygon points=\"");
            self.polygon_open = true;
        }
        let end = self
            .polygon_vertex
            .saturating_add(*remaining)
            .min(polygon.vertices.len());
        for point in &polygon.vertices[self.polygon_vertex..end] {
            if self.polygon_vertex > 0 {
                output.push(' ');
            }
            let _ = write!(output, "{},{}", point[0], -point[1]);
            self.polygon_vertex += 1;
            *remaining -= 1;
        }
        if self.polygon_vertex >= polygon.vertices.len() {
            let fill = turtle_stroke_rgb(polygon.color.into(), self.palette)
                .map(rgb_hex)
                .unwrap_or_else(|| self.foreground.to_owned());
            let _ = writeln!(output, "\" fill=\"{fill}\" stroke=\"none\"/>");
            self.polygon_index += 1;
            self.polygon_vertex = 0;
            self.polygon_open = false;
            // Empty polygons still consume work, preventing an unbounded loop
            // over malformed or intentionally empty fill commands.
            if polygon.vertices.is_empty() {
                *remaining = remaining.saturating_sub(1);
            }
        }
    }

    fn encode_line_work(&mut self, remaining: &mut usize) {
        let start = self.completed;
        let end = start.saturating_add(*remaining).min(self.scene.lines.len());
        let output = self
            .output
            .as_mut()
            .expect("unfinished SVG owns its buffer");
        for index in start..end {
            let line = self.scene.lines.get(index);
            if let Some(width_multiplier) = positive_width(line.width)
                && line.start.into_iter().chain(line.end).all(f32::is_finite)
            {
                let stroke = turtle_stroke_rgb(line.color, self.palette)
                    .map(rgb_hex)
                    .unwrap_or_else(|| self.foreground.to_owned());
                let width = self.stroke_width * width_multiplier;
                let _ = writeln!(
                    output,
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>",
                    line.start[0], -line.start[1], line.end[0], -line.end[1]
                );
            }
        }
        let encoded = end - start;
        self.completed = end;
        *remaining -= encoded;
    }

    fn encode_text_work(&mut self, remaining: &mut usize) {
        let text = &self.scene.texts[self.text_index];
        let output = self
            .output
            .as_mut()
            .expect("unfinished SVG owns its buffer");
        if !self.text_open {
            let color = match text.role {
                WorkerTextRole::Muted => self.muted,
                WorkerTextRole::Error => "#dc5050",
                WorkerTextRole::Title | WorkerTextRole::Heading | WorkerTextRole::Body => {
                    self.foreground
                }
            };
            let size = if text.size.is_finite() && text.size > 0.0 {
                text.size
            } else {
                16.0
            };
            let _ = write!(
                output,
                "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"{size}\" fill=\"{color}\">",
                finite_or(text.x, 0.0),
                finite_or(text.y, 0.0),
            );
            self.text_open = true;
        }
        while *remaining > 0 && self.text_byte < text.content.len() {
            let character = text.content[self.text_byte..]
                .chars()
                .next()
                .expect("text byte cursor stays on a character boundary");
            write_xml_escaped_character(output, character);
            self.text_byte += character.len_utf8();
            *remaining -= 1;
        }
        if self.text_byte >= text.content.len() {
            output.push_str("</text>\n");
            self.text_index += 1;
            self.text_byte = 0;
            self.text_open = false;
            if text.content.is_empty() {
                *remaining = remaining.saturating_sub(1);
            }
        }
    }
}

struct PreparedCanvas {
    context: CanvasRenderingContext2d,
    viewport: CanvasViewport,
    resized: bool,
}

fn prepare_canvas(canvas: &HtmlCanvasElement) -> Result<PreparedCanvas, CanvasError> {
    let rect = canvas.get_bounding_client_rect();
    let fallback_width = f64::from(canvas.client_width().max(1));
    let fallback_height = f64::from(canvas.client_height().max(1));
    let logical_width = positive_finite_or(rect.width(), fallback_width).max(1.0);
    let logical_height = positive_finite_or(rect.height(), fallback_height).max(1.0);
    let device_pixel_ratio = web_sys::window()
        .map(|window| positive_finite_or(window.device_pixel_ratio(), 1.0))
        .unwrap_or(1.0);
    let backing_width = backing_dimension(logical_width, device_pixel_ratio)?;
    let backing_height = backing_dimension(logical_height, device_pixel_ratio)?;
    let resized = canvas.width() != backing_width || canvas.height() != backing_height;
    if canvas.width() != backing_width {
        canvas.set_width(backing_width);
    }
    if canvas.height() != backing_height {
        canvas.set_height(backing_height);
    }
    let context = canvas
        .get_context("2d")
        .map_err(|error| browser_error("request a Canvas2D context", error))?
        .ok_or_else(|| CanvasError::Browser {
            operation: "request a Canvas2D context",
            detail: String::from("the browser returned no context"),
        })?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| CanvasError::Browser {
            operation: "request a Canvas2D context",
            detail: String::from("the browser returned a different context type"),
        })?;
    context
        .set_transform(device_pixel_ratio, 0.0, 0.0, device_pixel_ratio, 0.0, 0.0)
        .map_err(|error| browser_error("scale the Canvas2D context for HiDPI", error))?;
    Ok(PreparedCanvas {
        context,
        resized,
        viewport: CanvasViewport {
            logical_width,
            logical_height,
            device_pixel_ratio,
            backing_width,
            backing_height,
        },
    })
}

/// Clears a canvas at its current CSS and device-pixel size.
pub(crate) fn clear_canvas(canvas: &HtmlCanvasElement) -> Result<CanvasViewport, CanvasError> {
    let prepared = prepare_canvas(canvas)?;
    clear_context(&prepared.context, prepared.viewport);
    Ok(prepared.viewport)
}

fn clear_context(context: &CanvasRenderingContext2d, viewport: CanvasViewport) {
    context.set_global_alpha(1.0);
    context.clear_rect(0.0, 0.0, viewport.logical_width, viewport.logical_height);
}

fn backing_dimension(logical: f64, scale: f64) -> Result<u32, CanvasError> {
    let pixels = (logical * scale).ceil();
    if !pixels.is_finite() || pixels <= 0.0 || pixels > f64::from(u32::MAX) {
        return Err(CanvasError::InvalidMetadata(String::from(
            "browser canvas dimensions are outside the supported range",
        )));
    }
    Ok(pixels as u32)
}

fn validate_chunks(
    values: js_sys::Array,
    line_count: usize,
    lines_per_chunk: usize,
    values_per_line: usize,
    kind: &'static str,
) -> Result<Box<[Float32Array]>, CanvasError> {
    let expected_values =
        line_count
            .checked_mul(values_per_line)
            .ok_or(CanvasError::InvalidTransfer {
                kind,
                expected_values: usize::MAX,
                actual_values: 0,
            })?;
    let expected_chunks = line_count.div_ceil(lines_per_chunk);
    if values.length() as usize != expected_chunks {
        return Err(CanvasError::InvalidTransfer {
            kind,
            expected_values,
            actual_values: 0,
        });
    }
    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(expected_chunks)
        .map_err(|_| CanvasError::ResourceExhausted {
            operation: "retain transferred geometry",
            requested_items: line_count,
        })?;
    let mut actual_values = 0usize;
    for (chunk_index, value) in values.iter().enumerate() {
        let chunk = value
            .dyn_into::<Float32Array>()
            .map_err(|_| CanvasError::InvalidTransfer {
                kind,
                expected_values,
                actual_values,
            })?;
        let remaining = line_count.saturating_sub(chunk_index.saturating_mul(lines_per_chunk));
        let expected_chunk_values = remaining
            .min(lines_per_chunk)
            .checked_mul(values_per_line)
            .ok_or(CanvasError::InvalidTransfer {
                kind,
                expected_values,
                actual_values,
            })?;
        let chunk_values = chunk.length() as usize;
        if chunk_values != expected_chunk_values {
            return Err(CanvasError::InvalidTransfer {
                kind,
                expected_values,
                actual_values: actual_values.saturating_add(chunk_values),
            });
        }
        actual_values = actual_values.saturating_add(chunk_values);
        chunks.push(chunk);
    }
    if actual_values != expected_values {
        return Err(CanvasError::InvalidTransfer {
            kind,
            expected_values,
            actual_values,
        });
    }
    Ok(chunks.into_boxed_slice())
}

fn validate_bounds(
    bounds: Option<[f32; 4]>,
    label: &'static str,
) -> Result<Option<ViewBounds>, CanvasError> {
    let Some([min_x, max_x, min_y, max_y]) = bounds else {
        return Ok(None);
    };
    if ![min_x, max_x, min_y, max_y].into_iter().all(f32::is_finite)
        || min_x > max_x
        || min_y > max_y
    {
        return Err(CanvasError::InvalidMetadata(format!(
            "browser worker returned invalid {label} bounds",
        )));
    }
    Ok(Some(ViewBounds::new(
        f64::from(min_x),
        f64::from(max_x),
        f64::from(min_y),
        f64::from(max_y),
    )))
}

/// Worker line bounds intentionally exclude separately transferred fills. Fold
/// polygon vertices into the camera authority so filled or polygon-only scenes
/// fit exactly like native scenes do.
fn bounds_including_polygons(
    bounds: Option<ViewBounds>,
    polygons: &[WorkerPolygon],
) -> Result<Option<ViewBounds>, CanvasError> {
    let mut bounds = bounds;
    for polygon in polygons {
        for &[x, y] in &polygon.vertices {
            if !x.is_finite() || !y.is_finite() {
                return Err(CanvasError::InvalidMetadata(String::from(
                    "browser worker returned a polygon with a non-finite vertex",
                )));
            }
            bounds = Some(match bounds {
                Some(bounds) => ViewBounds::new(
                    bounds.min_x.min(x),
                    bounds.max_x.max(x),
                    bounds.min_y.min(y),
                    bounds.max_y.max(y),
                ),
                None => ViewBounds::new(x, x, y, y),
            });
        }
    }
    Ok(bounds)
}

fn validate_bounds_3d(
    bounds: Option<[f32; 6]>,
    label: &'static str,
) -> Result<Option<ViewBounds3d>, CanvasError> {
    let Some([min_x, max_x, min_y, max_y, min_z, max_z]) = bounds else {
        return Ok(None);
    };
    if ![min_x, max_x, min_y, max_y, min_z, max_z]
        .into_iter()
        .all(f32::is_finite)
        || min_x > max_x
        || min_y > max_y
        || min_z > max_z
    {
        return Err(CanvasError::InvalidMetadata(format!(
            "browser worker returned invalid {label} spatial bounds",
        )));
    }
    Ok(Some(ViewBounds3d::new(
        f64::from(min_x),
        f64::from(max_x),
        f64::from(min_y),
        f64::from(max_y),
        f64::from(min_z),
        f64::from(max_z),
    )))
}

fn bounds_3d_including_polygons(
    bounds: Option<ViewBounds3d>,
    polygons: &[WorkerPolygon3d],
) -> Result<Option<ViewBounds3d>, CanvasError> {
    let mut bounds = bounds;
    for polygon in polygons {
        for &[x, y, z] in &polygon.vertices {
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(CanvasError::InvalidMetadata(String::from(
                    "browser worker returned a spatial polygon with a non-finite vertex",
                )));
            }
            bounds = Some(match bounds {
                Some(bounds) => ViewBounds3d::new(
                    bounds.min_x.min(x),
                    bounds.max_x.max(x),
                    bounds.min_y.min(y),
                    bounds.max_y.max(y),
                    bounds.min_z.min(z),
                    bounds.max_z.max(z),
                ),
                None => ViewBounds3d::new(x, x, y, y, z, z),
            });
        }
    }
    Ok(bounds)
}

fn validate_line_length(length: Option<f64>, label: &'static str) -> Result<(), CanvasError> {
    if length.is_some_and(|length| !length.is_finite() || length <= 0.0) {
        return Err(CanvasError::InvalidMetadata(format!(
            "browser worker returned an invalid {label} line-length summary",
        )));
    }
    Ok(())
}

fn draw_source_lines(
    context: &CanvasRenderingContext2d,
    lines: &LineStorage,
    transform: &ViewTransform,
    base_width: f64,
    dark: bool,
    indices: impl IntoIterator<Item = usize>,
) {
    context.set_global_alpha(1.0);
    context.set_line_cap("round");
    context.set_line_join("round");
    let mut batch = StrokeBatch::default();
    for index in indices {
        let line = lines.get(index);
        let Some(width_multiplier) = positive_width(line.width) else {
            continue;
        };
        let width = base_width * width_multiplier;
        if !width.is_finite() || width <= 0.0 {
            continue;
        }
        let color = resolve_styled_color(line.color, line.start, line.end, transform, dark);
        let start = project(line.start, transform);
        let end = project(line.end, transform);
        batch.push(context, color, width, start, end, dark);
    }
    batch.finish(context);
}

#[derive(Debug, Clone, Copy)]
struct SpatialPaintStyle {
    base_width: f64,
    width_reference: f64,
    line_width_scale: f64,
    background: [u8; 3],
    palette: Palette,
}

impl SpatialPaintStyle {
    fn dark(self) -> bool {
        self.palette == Palette::Dark
    }
}

fn draw_spatial_lines(
    context: &CanvasRenderingContext2d,
    lines: &LineStorage3d,
    transform: &ViewTransform3d,
    style: SpatialPaintStyle,
    indices: impl IntoIterator<Item = usize>,
) {
    context.set_global_alpha(1.0);
    context.set_line_cap("round");
    context.set_line_join("round");
    let mut batch = StrokeBatch::default();
    for index in indices {
        push_spatial_line(&mut batch, context, lines.get(index), transform, style);
    }
    batch.finish(context);
}

fn draw_spatial_primitives(
    context: &CanvasRenderingContext2d,
    lines: &LineStorage3d,
    polygons: &[WorkerPolygon3d],
    transform: &ViewTransform3d,
    style: SpatialPaintStyle,
    order: &[DepthPrimitive],
) {
    context.set_global_alpha(1.0);
    context.set_line_cap("round");
    context.set_line_join("round");
    let mut batch = StrokeBatch::default();
    for entry in order {
        match entry.primitive {
            SpatialPrimitive::Line(index) => {
                push_spatial_line(&mut batch, context, lines.get(index), transform, style)
            }
            SpatialPrimitive::Polygon(index) => {
                batch.finish(context);
                draw_spatial_polygon(context, &polygons[index], transform, style);
            }
        }
    }
    batch.finish(context);
}

fn push_spatial_line(
    batch: &mut StrokeBatch,
    context: &CanvasRenderingContext2d,
    line: SourceLine3d,
    transform: &ViewTransform3d,
    style: SpatialPaintStyle,
) {
    let Some(width_multiplier) = spatial_width_multiplier(line.width, style.width_reference) else {
        return;
    };
    let width = style.base_width * width_multiplier * style.line_width_scale;
    if !width.is_finite() || width <= 0.0 {
        return;
    }
    let color = ResolvedColor::Rgb(spatial_line_rgb(
        line,
        transform,
        style.palette,
        style.background,
    ));
    let start = project_3d(line.start, transform);
    let end = project_3d(line.end, transform);
    batch.push(context, color, width, start, end, style.dark());
}

fn draw_spatial_polygon(
    context: &CanvasRenderingContext2d,
    polygon: &WorkerPolygon3d,
    transform: &ViewTransform3d,
    style: SpatialPaintStyle,
) {
    let Some(&[x, y, z]) = polygon.vertices.first() else {
        return;
    };
    let first = transform.project(WorldPoint3d::new(x, y, z));
    context.begin_path();
    context.move_to(first.x, first.y);
    for &[x, y, z] in polygon.vertices.iter().skip(1) {
        let point = transform.project(WorldPoint3d::new(x, y, z));
        context.line_to(point.x, point.y);
    }
    context.close_path();
    let color = ResolvedColor::Rgb(spatial_polygon_rgb(
        polygon,
        transform,
        style.palette,
        style.background,
    ));
    set_fill_color(context, color, style.dark());
    context.fill();
}

fn spatial_line_depth(line: SourceLine3d, transform: &ViewTransform3d) -> f64 {
    let start = transform.project(world_point_3d(line.start));
    let end = transform.project(world_point_3d(line.end));
    finite_or((start.depth + end.depth) * 0.5, 0.0)
}

fn spatial_polygon_depth(polygon: &WorkerPolygon3d, transform: &ViewTransform3d) -> f64 {
    spatial_polygon_centroid(polygon)
        .map(|centroid| transform.project(centroid).depth)
        .map_or(0.0, |depth| finite_or(depth, 0.0))
}

fn spatial_primitive_count(line_count: usize, polygon_count: usize) -> Result<usize, CanvasError> {
    line_count
        .checked_add(polygon_count)
        .ok_or(CanvasError::ResourceExhausted {
            operation: "order spatial Canvas primitives",
            requested_items: usize::MAX,
        })
}

fn spatial_primitive_order(
    lines: &LineStorage3d,
    polygons: &[WorkerPolygon3d],
    transform: &ViewTransform3d,
    selected_lines: usize,
) -> Result<Vec<DepthPrimitive>, CanvasError> {
    let primitive_count = spatial_primitive_count(selected_lines, polygons.len())?;
    let mut order = Vec::new();
    order
        .try_reserve_exact(primitive_count)
        .map_err(|_| CanvasError::ResourceExhausted {
            operation: "order spatial Canvas primitives",
            requested_items: primitive_count,
        })?;
    for output_index in 0..selected_lines {
        let index = representative_index(output_index, selected_lines, lines.len());
        order.push(DepthPrimitive {
            depth: spatial_line_depth(lines.get(index), transform),
            primitive: SpatialPrimitive::Line(index),
            source_order: spatial_line_source_position(index, polygons),
        });
    }
    for (index, polygon) in polygons.iter().enumerate() {
        order.push(DepthPrimitive {
            depth: spatial_polygon_depth(polygon, transform),
            primitive: SpatialPrimitive::Polygon(index),
            source_order: spatial_polygon_source_position(index, polygon),
        });
    }
    order.sort_unstable_by(compare_depth_primitives);
    Ok(order)
}

fn compare_depth_primitives(left: &DepthPrimitive, right: &DepthPrimitive) -> Ordering {
    left.depth
        .total_cmp(&right.depth)
        .then_with(|| left.source_order.cmp(&right.source_order))
}

fn spatial_line_source_position(index: usize, polygons: &[WorkerPolygon3d]) -> usize {
    let polygons_before = polygons.partition_point(|polygon| polygon.lines_before <= index);
    index.saturating_add(polygons_before)
}

fn spatial_polygon_source_position(index: usize, polygon: &WorkerPolygon3d) -> usize {
    polygon.lines_before.saturating_add(index)
}

/// Sorts in bounded runs and merges those runs cooperatively. `sort_unstable`
/// is intentionally limited to one export quantum; merging checks cancellation
/// and yields after every bounded amount of work.
async fn sort_spatial_primitives_yielding(
    order: &mut Vec<DepthPrimitive>,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<(), CanvasError> {
    for chunk in order.chunks_mut(SVG_LINES_PER_BATCH) {
        if is_cancelled() {
            return Err(CanvasError::Cancelled);
        }
        chunk.sort_unstable_by(compare_depth_primitives);
        yield_to_browser().await;
    }
    if order.len() <= SVG_LINES_PER_BATCH {
        return if is_cancelled() {
            Err(CanvasError::Cancelled)
        } else {
            Ok(())
        };
    }

    let item_count = order.len();
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(item_count)
        .map_err(|_| CanvasError::ResourceExhausted {
            operation: "merge ordered spatial SVG primitives",
            requested_items: item_count,
        })?;
    let mut run_length = SVG_LINES_PER_BATCH;
    while run_length < item_count {
        scratch.clear();
        let mut run_start = 0usize;
        let mut work = 0usize;
        while run_start < item_count {
            let middle = run_start.saturating_add(run_length).min(item_count);
            let run_end = middle.saturating_add(run_length).min(item_count);
            let (mut left, mut right) = (run_start, middle);
            while left < middle || right < run_end {
                let take_left = right >= run_end
                    || (left < middle
                        && compare_depth_primitives(&order[left], &order[right])
                            != Ordering::Greater);
                if take_left {
                    scratch.push(order[left]);
                    left += 1;
                } else {
                    scratch.push(order[right]);
                    right += 1;
                }
                work += 1;
                if work >= SVG_LINES_PER_BATCH {
                    work = 0;
                    if is_cancelled() {
                        return Err(CanvasError::Cancelled);
                    }
                    yield_to_browser().await;
                }
            }
            run_start = run_end;
        }
        std::mem::swap(order, &mut scratch);
        run_length = run_length.saturating_mul(2);
    }
    if is_cancelled() {
        Err(CanvasError::Cancelled)
    } else {
        Ok(())
    }
}

fn depth_bucket(depth: f64, radius: f64) -> usize {
    let normalized = finite_or((depth / radius + 1.0) * 0.5, 0.5).clamp(0.0, 1.0);
    ((normalized * (DEPTH_BUCKETS - 1) as f64).floor() as usize).min(DEPTH_BUCKETS - 1)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ResolvedColor {
    Palette(usize),
    Rgb([u8; 3]),
}

#[derive(Default)]
struct StrokeBatch {
    color: Option<ResolvedColor>,
    width_bits: u64,
    segments: usize,
}

impl StrokeBatch {
    fn push(
        &mut self,
        context: &CanvasRenderingContext2d,
        color: ResolvedColor,
        width: f64,
        start: [f64; 2],
        end: [f64; 2],
        dark: bool,
    ) {
        let style_changed = self.color != Some(color) || self.width_bits != width.to_bits();
        if self.segments > 0 && (style_changed || self.segments >= MAX_PATH_SEGMENTS) {
            context.stroke();
            self.segments = 0;
        }
        if self.segments == 0 {
            context.begin_path();
            if style_changed || self.color.is_none() {
                set_stroke_color(context, color, dark);
                context.set_line_width(width);
                self.color = Some(color);
                self.width_bits = width.to_bits();
            }
        }
        context.move_to(start[0], start[1]);
        context.line_to(end[0], end[1]);
        self.segments += 1;
    }

    fn finish(&mut self, context: &CanvasRenderingContext2d) {
        if self.segments > 0 {
            context.stroke();
            self.segments = 0;
        }
    }
}

fn resolve_styled_color(
    color: StrokeColor,
    palette_start: [f32; 2],
    palette_end: [f32; 2],
    transform: &ViewTransform,
    dark: bool,
) -> ResolvedColor {
    turtle_stroke_rgb(color, target_palette(dark))
        .map(ResolvedColor::Rgb)
        .unwrap_or_else(|| {
            let position = transform.world_palette_position(
                WorldPoint::new(f64::from(palette_start[0]), f64::from(palette_start[1])),
                WorldPoint::new(f64::from(palette_end[0]), f64::from(palette_end[1])),
            );
            let bucket = theme_default_bucket(position);
            ResolvedColor::Palette(bucket)
        })
}

fn spatial_line_rgb(
    line: SourceLine3d,
    transform: &ViewTransform3d,
    palette: Palette,
    background: [u8; 3],
) -> [u8; 3] {
    let start = world_point_3d(line.start);
    let end = world_point_3d(line.end);
    let palette_position = transform.world_palette_position(start, end);
    spatial_appearance_rgb(
        line.color,
        palette_position,
        palette,
        background,
        transform.rod_light(start, end),
        transform.normalized_midpoint_depth(start, end),
    )
}

fn spatial_polygon_rgb(
    polygon: &WorkerPolygon3d,
    transform: &ViewTransform3d,
    palette: Palette,
    background: [u8; 3],
) -> [u8; 3] {
    let centroid = spatial_polygon_centroid(polygon);
    spatial_polygon_rgb_at(polygon, transform, palette, background, centroid)
}

fn spatial_polygon_rgb_at(
    polygon: &WorkerPolygon3d,
    transform: &ViewTransform3d,
    palette: Palette,
    background: [u8; 3],
    centroid: Option<WorldPoint3d>,
) -> [u8; 3] {
    let palette_position = centroid
        .map(|centroid| transform.world_palette_position(centroid, centroid))
        .unwrap_or(0.5);
    let light = transform.surface_light(
        polygon
            .vertices
            .iter()
            .map(|&[x, y, z]| WorldPoint3d::new(x, y, z)),
    );
    let near_depth = centroid
        .map(|centroid| transform.normalized_midpoint_depth(centroid, centroid))
        .unwrap_or(0.5);
    spatial_appearance_rgb(
        polygon.color.into(),
        palette_position,
        palette,
        background,
        light,
        near_depth,
    )
}

fn spatial_polygon_centroid(polygon: &WorkerPolygon3d) -> Option<WorldPoint3d> {
    let mut centroid = None;
    for (index, &[x, y, z]) in polygon.vertices.iter().enumerate() {
        include_spatial_centroid_point(&mut centroid, WorldPoint3d::new(x, y, z), index);
    }
    centroid
}

fn include_spatial_centroid_point(
    centroid: &mut Option<WorldPoint3d>,
    point: WorldPoint3d,
    index: usize,
) {
    let Some(current) = centroid.as_mut() else {
        *centroid = Some(point);
        return;
    };
    let weight = 1.0 / (index + 1) as f64;
    current.x += (point.x - current.x) * weight;
    current.y += (point.y - current.y) * weight;
    current.z += (point.z - current.z) * weight;
}

fn spatial_appearance_rgb(
    color: StrokeColor,
    palette_position: f64,
    palette: Palette,
    background: [u8; 3],
    light: f64,
    near_depth: f64,
) -> [u8; 3] {
    let base = turtle_stroke_rgb(color, palette)
        .map(normalized_rgb8)
        .unwrap_or_else(|| {
            spatial_theme_default_color(spatial_theme_default_bucket(palette_position), palette)
        });
    spatial_lit_color_bounded(base, normalized_rgb8(background), light, near_depth)
        .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn normalized_rgb8(rgb: [u8; 3]) -> [f32; 3] {
    rgb.map(|channel| f32::from(channel) / 255.0)
}

fn set_stroke_color(context: &CanvasRenderingContext2d, color: ResolvedColor, dark: bool) {
    context.set_stroke_style_str(&resolved_css(color, dark));
}

fn set_fill_color(context: &CanvasRenderingContext2d, color: ResolvedColor, dark: bool) {
    context.set_fill_style_str(&resolved_css(color, dark));
}

fn resolved_css(color: ResolvedColor, dark: bool) -> String {
    match color {
        ResolvedColor::Rgb(rgb) => css_rgb(rgb),
        ResolvedColor::Palette(index) => {
            css_components(resolved_components(ResolvedColor::Palette(index), dark))
        }
    }
}

fn resolved_components(color: ResolvedColor, dark: bool) -> [f64; 3] {
    match color {
        ResolvedColor::Rgb([red, green, blue]) => [
            f64::from(red) / 255.0,
            f64::from(green) / 255.0,
            f64::from(blue) / 255.0,
        ],
        ResolvedColor::Palette(index) => {
            let [red, green, blue] = theme_default_color(index, target_palette(dark));
            [f64::from(red), f64::from(green), f64::from(blue)]
        }
    }
}

fn css_rgb([red, green, blue]: [u8; 3]) -> String {
    format!("rgb({red} {green} {blue})")
}

fn css_components([red, green, blue]: [f64; 3]) -> String {
    format!(
        "rgb({} {} {})",
        red.clamp(0.0, 1.0) * 255.0,
        green.clamp(0.0, 1.0) * 255.0,
        blue.clamp(0.0, 1.0) * 255.0,
    )
}

fn target_palette(dark: bool) -> Palette {
    if dark { Palette::Dark } else { Palette::Light }
}

fn display_background(source: Option<[u8; 3]>, dark: bool) -> [u8; 3] {
    source.unwrap_or(if dark {
        DARK_BACKGROUND
    } else {
        LIGHT_BACKGROUND
    })
}

fn themed_foreground(dark: bool) -> [u8; 3] {
    if dark {
        DARK_FOREGROUND
    } else {
        LIGHT_FOREGROUND
    }
}

fn representative_index(output_index: usize, selected: usize, total: usize) -> usize {
    debug_assert!(selected <= total);
    if selected == total {
        output_index
    } else if selected <= 1 {
        total / 2
    } else {
        ((output_index as u128 * (total - 1) as u128) / (selected - 1) as u128) as usize
    }
}

fn project(point: [f32; 2], transform: &ViewTransform) -> [f64; 2] {
    let point = transform.project(WorldPoint::new(f64::from(point[0]), f64::from(point[1])));
    [point.x, point.y]
}

fn world_point_3d(point: [f32; 3]) -> WorldPoint3d {
    WorldPoint3d::new(
        f64::from(point[0]),
        f64::from(point[1]),
        f64::from(point[2]),
    )
}

fn project_3d(point: [f32; 3], transform: &ViewTransform3d) -> [f64; 2] {
    let point = transform.project(world_point_3d(point));
    [point.x, point.y]
}

fn transition_base_width(
    total_line_length: Option<f64>,
    line_count: usize,
    bounds: ViewBounds,
    transform: &ViewTransform,
    line_width_scale: f64,
) -> f64 {
    let zoom = transform.stroke_scale() / transform.fit_scale();
    adaptive_scene_stroke_width(
        total_line_length,
        line_count,
        bounds_extent(bounds),
        transform.fit_scale(),
    ) * zoom
        * valid_line_width_scale(line_width_scale)
}

fn bounds_extent(bounds: ViewBounds) -> (f64, f64) {
    (
        (bounds.max_x - bounds.min_x).max(0.0),
        (bounds.max_y - bounds.min_y).max(0.0),
    )
}

fn draw_empty_placeholder(context: &CanvasRenderingContext2d, viewport: CanvasViewport) {
    context.set_global_alpha(0.35);
    context.set_stroke_style_str("rgb(110 120 132)");
    context.set_line_width(1.0);
    context.stroke_rect(
        viewport.logical_width * 0.15,
        viewport.logical_height * 0.15,
        viewport.logical_width * 0.7,
        viewport.logical_height * 0.7,
    );
    context.set_global_alpha(1.0);
}

fn positive_width(width: f32) -> Option<f64> {
    (width.is_finite() && width > 0.0).then_some(f64::from(width))
}

fn spatial_width_multiplier(width: f32, width_reference: f64) -> Option<f64> {
    let width = normalized_turtle_3d_width(f64::from(width), width_reference);
    (width > 0.0).then_some(width)
}

fn positive_f32(width: f32) -> f64 {
    positive_width(width).unwrap_or(0.0)
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

fn positive_finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn valid_line_width_scale(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

fn lerp(left: f64, right: f64, progress: f64) -> f64 {
    left + (right - left) * progress
}

fn browser_error(operation: &'static str, error: JsValue) -> CanvasError {
    CanvasError::Browser {
        operation,
        detail: error.as_string().unwrap_or_else(|| format!("{error:?}")),
    }
}

fn svg_view(scene: &CanvasScene) -> (f64, f64, f64, f64, (f64, f64)) {
    let bounds = scene.had_bounds.then_some([
        scene.bounds.min_x,
        scene.bounds.max_x,
        scene.bounds.min_y,
        scene.bounds.max_y,
    ]);
    let Some([min_x, max_x, min_y, max_y]) = bounds else {
        return (0.0, 0.0, 1000.0, 700.0, (0.0, 0.0));
    };
    let drawing_width = (max_x - min_x).max(0.0);
    let drawing_height = (max_y - min_y).max(0.0);
    let width = drawing_width.max(0.1);
    let height = drawing_height.max(0.1);
    let margin = width.max(height) * 0.04;
    (
        min_x - margin,
        -max_y - margin,
        width + margin * 2.0,
        height + margin * 2.0,
        (drawing_width, drawing_height),
    )
}

fn rgb_hex([red, green, blue]: [u8; 3]) -> String {
    format!("#{red:02x}{green:02x}{blue:02x}")
}

fn write_xml_escaped_character(output: &mut String, character: char) {
    match character {
        '&' => output.push_str("&amp;"),
        '<' => output.push_str("&lt;"),
        '>' => output.push_str("&gt;"),
        '"' => output.push_str("&quot;"),
        '\'' => output.push_str("&apos;"),
        _ => output.push(character),
    }
}

async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if web_sys::window().is_some_and(|window| {
            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
                .is_ok()
        }) {
            return;
        }
        let _ = resolve.call0(&JsValue::UNDEFINED);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use braken_viz::StrokeColor;
    use braken_viz::targets::{
        Palette, SPATIAL_SOFTWARE_DEPTH_BUCKETS, SPATIAL_SOFTWARE_LIGHT_BUCKETS,
        SPATIAL_THEME_DEFAULT_COLOR_BUCKETS,
    };

    use super::{
        DepthPrimitive, ResolvedColor, SpatialPrimitive, compare_depth_primitives,
        planar_navigation_enabled, resolve_styled_color, spatial_appearance_rgb,
        spatial_line_source_position, spatial_polygon_centroid, spatial_polygon_source_position,
        spatial_width_multiplier,
    };
    use crate::camera::{Camera2d, ViewBounds, ViewTransform, ViewportSize};
    use crate::worker_protocol::{WorkerPolygon3d, WorkerStrokeColor};

    fn polygon(lines_before: usize) -> WorkerPolygon3d {
        WorkerPolygon3d {
            vertices: Vec::new(),
            color: WorkerStrokeColor::ThemeDefault,
            lines_before,
        }
    }

    #[test]
    fn mixed_spatial_primitives_sort_far_to_near_with_source_order_ties() {
        let mut order = [
            DepthPrimitive {
                depth: 1.0,
                primitive: SpatialPrimitive::Polygon(0),
                source_order: 2,
            },
            DepthPrimitive {
                depth: -1.0,
                primitive: SpatialPrimitive::Line(0),
                source_order: 0,
            },
            DepthPrimitive {
                depth: 1.0,
                primitive: SpatialPrimitive::Line(1),
                source_order: 1,
            },
            DepthPrimitive {
                depth: 0.0,
                primitive: SpatialPrimitive::Polygon(1),
                source_order: 3,
            },
        ];

        order.sort_unstable_by(compare_depth_primitives);

        assert_eq!(
            order.map(|entry| entry.primitive),
            [
                SpatialPrimitive::Line(0),
                SpatialPrimitive::Polygon(1),
                SpatialPrimitive::Line(1),
                SpatialPrimitive::Polygon(0),
            ]
        );
    }

    #[test]
    fn compact_polygon_offsets_reconstruct_mixed_source_positions() {
        let polygons = [polygon(0), polygon(1), polygon(1), polygon(2)];

        assert_eq!(spatial_line_source_position(0, &polygons), 1);
        assert_eq!(spatial_line_source_position(1, &polygons), 4);
        assert_eq!(spatial_polygon_source_position(0, &polygons[0]), 0);
        assert_eq!(spatial_polygon_source_position(1, &polygons[1]), 2);
        assert_eq!(spatial_polygon_source_position(2, &polygons[2]), 3);
        assert_eq!(spatial_polygon_source_position(3, &polygons[3]), 5);
    }

    #[test]
    fn filled_planar_geometry_supports_navigation_but_inspector_text_does_not() {
        assert!(planar_navigation_enabled(0, 1, 0));
        assert!(planar_navigation_enabled(1, 1, 0));
        assert!(!planar_navigation_enabled(0, 0, 0));
        assert!(!planar_navigation_enabled(1, 1, 1));
    }

    #[test]
    fn planar_color_resolution_keeps_the_existing_palette_path() {
        let transform = ViewTransform::new(
            ViewBounds::new(0.0, 1.0, 0.0, 1.0),
            ViewportSize::new(640.0, 480.0),
            Camera2d::fit(),
        );
        assert!(matches!(
            resolve_styled_color(
                StrokeColor::ThemeDefault,
                [0.0, 0.0],
                [1.0, 1.0],
                &transform,
                false,
            ),
            ResolvedColor::Palette(_)
        ));
        assert_eq!(
            resolve_styled_color(
                StrokeColor::Rgb([12, 34, 56]),
                [0.0, 0.0],
                [1.0, 1.0],
                &transform,
                true,
            ),
            ResolvedColor::Rgb([12, 34, 56])
        );
    }

    #[test]
    fn spatial_appearance_is_finite_and_bounded_by_shared_buckets() {
        let mut colors = BTreeSet::new();
        for palette_index in 0..=64 {
            let position = palette_index as f64 / 64.0;
            for light_index in 0..=32 {
                let light = 0.68 + 0.32 * light_index as f64 / 32.0;
                for depth_index in 0..=32 {
                    let depth = depth_index as f64 / 32.0;
                    colors.insert(spatial_appearance_rgb(
                        StrokeColor::ThemeDefault,
                        position,
                        Palette::Dark,
                        [15, 20, 32],
                        light,
                        depth,
                    ));
                }
            }
        }
        assert!(
            colors.len()
                <= SPATIAL_THEME_DEFAULT_COLOR_BUCKETS
                    * SPATIAL_SOFTWARE_LIGHT_BUCKETS
                    * SPATIAL_SOFTWARE_DEPTH_BUCKETS
        );

        let invalid = spatial_appearance_rgb(
            StrokeColor::Rgb([200, 30, 80]),
            f64::NAN,
            Palette::Light,
            [242, 245, 249],
            f64::NAN,
            f64::INFINITY,
        );
        assert_eq!(
            invalid,
            spatial_appearance_rgb(
                StrokeColor::Rgb([200, 30, 80]),
                0.5,
                Palette::Light,
                [242, 245, 249],
                1.0,
                0.5,
            )
        );
        assert!(invalid[0] > invalid[1] && invalid[0] > invalid[2]);
    }

    #[test]
    fn spatial_polygon_color_uses_its_centroid_and_widths_use_the_scene_reference() {
        let polygon = WorkerPolygon3d {
            vertices: vec![[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 3.0, 9.0]],
            color: WorkerStrokeColor::ThemeDefault,
            lines_before: 0,
        };
        let centroid = spatial_polygon_centroid(&polygon).expect("non-empty polygon");
        assert!((centroid.x - 2.0).abs() < 1.0e-12);
        assert!((centroid.y - 1.0).abs() < 1.0e-12);
        assert!((centroid.z - 3.0).abs() < 1.0e-12);

        assert_eq!(spatial_width_multiplier(4.0, 2.0), Some(2.0));
        assert_eq!(spatial_width_multiplier(8.0, 2.0), Some(3.0));
        assert_eq!(spatial_width_multiplier(0.0, 2.0), None);
    }
}
