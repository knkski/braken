#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

//! Responsive orchestration for derivation, visualization, and display.
//!
//! These are deliberately separate stages. `braken` derives a
//! backend-neutral generation, `braken_viz` turns it into scene geometry, and
//! Iced displays that geometry through WGPU when available or its software
//! fallback. `DerivationInfo` and cache metadata describe derivation only;
//! result/status summaries label derivation and visualization separately.
//!
//! Turtle iteration controls keep the last completed integer scene as their
//! authority. A new model first displays iteration 0; adjacent worker requests
//! then prepare lineage-based deformations in either direction, including the
//! initial 0-to-1 rewrite. Animation
//! frames update only bounded display previews, and promoting an exact target
//! is the only operation that advances the displayed iteration or cache.
//!
//! Browsing and panel changes are display-only. Mobile grammar edits remain a
//! draft until Apply; closing the editor retains the draft for the same source.

mod camera;
mod canvas_clip;
mod generated_preset_previews;
mod layout;
mod line_shader;
mod spatial_scene;
#[cfg(test)]
mod ui_state_tests;
#[cfg(target_arch = "wasm32")]
mod worker_protocol;

use braken_gui::{
    ir_tooling,
    orientation::OrientationAnchor,
    presets,
    theme_palette::{THEME_DEFAULT_COLOR_BUCKETS, theme_default_bucket, theme_default_color},
};

use camera::{Camera2d, Orbit3d, ScreenPoint, ViewBounds, ViewTransform, ViewportSize, WorldPoint};
use line_shader::{LineProgram, LineScene, LineSceneBuilder, TransitionProgram, TransitionScene};
use spatial_scene::{
    ProjectedPrimitive3d, SpatialProgram, SpatialScene, SpatialSceneBuilder, SpatialSceneError,
};
#[cfg(target_arch = "wasm32")]
use spatial_scene::{RenderBounds3d, SpatialPolygon3d};

#[cfg(target_arch = "wasm32")]
use braken::CalculationPhase;
use braken::{
    AmbiguousRulePolicy, BackendChoice, CalculationLimits, CalculationProgress, CalculationRequest,
    CancellationToken, CompiledGrammar, CpuBackend, DerivationSemantics, FloatWidth, Generation,
    calculate_rewrite_lineage_with_control, calculate_with_control, generate_seed,
};
#[cfg(test)]
use braken_viz::Primitive3d;
#[cfg(target_arch = "wasm32")]
use braken_viz::Text2d;
use braken_viz::targets::{
    Palette, StrokeWidthEstimator, adaptive_scene_stroke_width, svg as svg_target,
    turtle_stroke_rgb,
};
use braken_viz::{
    IndexedLine2d, IndexedModulePosition2d, InspectorConfig, InspectorStreamRequest, Polygon2d,
    Primitive2d, Scene2d, Scene3d, StrokeColor, TextRole, Turtle2dConfig, Turtle2dProgress,
    Turtle2dStreamRequest, Turtle2dStreamer, Turtle3dConfig, Turtle3dProgress,
    Turtle3dStreamRequest, Visualization, VisualizationContext, VisualizeError, VisualizerBackend,
    VisualizerConfig, VisualizerKind, build_line_transition, stream_inspector, stream_turtle_3d,
    visualize_with_backend,
};
use iced::gradient;
use iced::mouse;
use iced::theme;
use iced::widget::canvas::{self, Cache, Canvas, LineCap, LineJoin, Path, Stroke};
use iced::widget::{
    Row, Space, button as button_widget, column, container as container_widget, pick_list,
    responsive, row, scrollable, slider, stack, svg, text, text_input, tooltip,
};
use iced::widget::{button, container, mouse_area, shader, text_editor};
use iced::{
    Alignment, Background, Border, Color, Degrees, Element, Length, Point, Rectangle,
    Renderer as IcedRenderer, Shadow, Size, Task, Theme, Vector,
};
use std::collections::HashMap;
use std::fmt;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use web_time::Instant as RenderInstant;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Condvar;

const MOBILE_BREAKPOINT: f32 = 700.0;
const CANVAS_PADDING: f32 = 16.0;
const COLOR_BUCKETS: usize = THEME_DEFAULT_COLOR_BUCKETS;
const CANVAS_BACKGROUND: Color = Color::from_rgb8(242, 245, 249);
const DARK_CANVAS_BACKGROUND: Color = Color::from_rgb8(15, 20, 32);
const GENERATION_CACHE_BUDGET: usize = 64 * 1024 * 1024;
const ITERATION_TRANSITION_DURATION: Duration = Duration::from_millis(600);
const MIN_ANGLE: f32 = 0.0;
const MAX_ANGLE: f32 = 180.0;
const ANGLE_SLIDER_STEP: f32 = 0.1;
const MIN_LINE_WIDTH_PERCENT: f32 = 5.0;
const MAX_LINE_WIDTH_PERCENT: f32 = 200.0;
const DEFAULT_LINE_WIDTH_PERCENT: f32 = 100.0;
const LINE_WIDTH_SLIDER_STEP: f32 = 5.0;
const BACKGROUND_STATUS_INTERVAL: Duration = Duration::from_millis(100);
const TRANSIENT_STATUS_DELAY: Duration = Duration::from_millis(250);
const SLOW_RENDER_NOTICE_AFTER: Duration = Duration::from_secs(2);
const SOURCE_EDIT_DEBOUNCE: Duration = Duration::from_millis(180);
const VIEW_WHEEL_SETTLE_DELAY: Duration = Duration::from_millis(150);
const VIEW_BUTTON_ZOOM_FACTOR: f64 = 1.25;
const VIEW_WHEEL_LINES_FACTOR: f64 = 1.15;
const VIEW_WHEEL_PIXELS_PER_LINE: f64 = 100.0;
const AUTOROTATE_RADIANS_PER_SECOND: f64 = std::f64::consts::PI / 15.0;
const MAX_AUTOROTATE_FRAME_DELTA: Duration = Duration::from_millis(100);
// This scene is used only by Iced's software renderer. The WGPU shader streams
// the exact line set in bounded batches, so the compatibility Canvas must never
// receive enough geometry to recreate the old monolithic index-buffer failure.
const SOFTWARE_FALLBACK_PREVIEW_LINES: usize = 4 * 1024;

fn should_offer_render_cancel(elapsed: Duration) -> bool {
    elapsed >= SLOW_RENDER_NOTICE_AFTER
}

fn should_show_transient_status(elapsed: Duration) -> bool {
    elapsed >= TRANSIENT_STATUS_DELAY
}

fn application_window_settings() -> iced::window::Settings {
    iced::window::Settings {
        size: Size::new(1280.0, 820.0),
        icon: application_window_icon(),
        ..iced::window::Settings::default()
    }
}

fn application_window_icon() -> Option<iced::window::Icon> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let rgba = include_bytes!("../assets/icons/native/app-icon-256.rgba").to_vec();
        Some(
            iced::window::icon::from_rgba(rgba, 256, 256)
                .expect("the generated native application icon must be valid RGBA"),
        )
    }

    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

fn main() -> iced::Result {
    install_panic_hook();

    let result = iced::application(BrakenGui::new, BrakenGui::update, BrakenGui::view)
        .title(app_title)
        .subscription(app_subscription)
        .theme(app_theme)
        .window(application_window_settings())
        // Iced 0.14's pooled Canvas MSAA resolve can leak stale regions after
        // resizing or scrolling. Custom display shaders do not use this pass;
        // tiny-skia keeps its own path antialiasing independently.
        .antialiasing(false)
        .run();

    #[cfg(target_arch = "wasm32")]
    if let Err(error) = &result {
        show_wasm_failure(
            "The browser application could not start",
            &error.to_string(),
        );
    }

    result
}

fn install_panic_hook() {
    #[cfg(all(target_arch = "wasm32", feature = "console_error_panic_hook"))]
    std::panic::set_hook(Box::new(|info| {
        show_wasm_failure("The application panicked", &info.to_string());
        console_error_panic_hook::hook(info);
    }));
}

#[cfg(target_arch = "wasm32")]
fn show_wasm_failure(title: &str, message: &str) {
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    if let Some(heading) = document.get_element_by_id("wasm-status-title") {
        heading.set_text_content(Some(title));
    }
    if let Some(summary) = document.get_element_by_id("wasm-status-summary") {
        summary.set_text_content(Some(
            "The browser build stopped unexpectedly. Reload the page, or copy the details below when reporting the problem.",
        ));
    }
    if let Some(details) = document.get_element_by_id("wasm-status-details") {
        details.set_text_content(Some(message));
        let _result = details.remove_attribute("hidden");
    }
    if let Some(reload) = document.get_element_by_id("wasm-status-reload") {
        let _result = reload.remove_attribute("hidden");
    }
    if let Some(panel) = document.get_element_by_id("wasm-status") {
        let _result = panel.set_attribute("data-state", "error");
        let _result = panel.set_attribute("data-source", "rust");
        let _result = panel.remove_attribute("hidden");
    }
}

fn app_title(_state: &BrakenGui) -> String {
    String::from("Braken")
}

fn app_theme(state: &BrakenGui) -> Theme {
    match state.effective_theme_mode() {
        theme::Mode::Dark => Theme::TokyoNightStorm,
        theme::Mode::Light | theme::Mode::None => Theme::TokyoNightLight,
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum Message {
    SourceEdited(text_editor::Action),
    DraftEdited(text_editor::Action),
    ToggleEditor(LayoutMode),
    CloseEditor,
    ApplyDraft,
    OpenCatalog,
    CloseCatalog,
    CloseOverlay,
    WindowResized(Size),
    ExploreScrolled(scrollable::AbsoluteOffset),
    ToggleDiagnostics,
    AngleChanged(f32),
    AngleDecrement,
    AngleIncrement,
    AngleScrolled(mouse::ScrollDelta),
    LineWidthChanged(f32),
    IterationsChanged(f32),
    IterationsInputChanged(String),
    IterationsDecrement,
    IterationsIncrement,
    IterationsScrolled(mouse::ScrollDelta),
    SeedChanged(String),
    RandomizeSeed,
    FloatWidthChanged(FloatWidth),
    AmbiguousRulesChanged(AmbiguousRulePolicy),
    PresetSearchChanged(String),
    ClearPresetSearch,
    PresetSelected(PresetChoice),
    Rendered(RenderOutcome),
    CancelRender,
    ToggleSettings,
    DockViewSelected(DockView),
    CopyIr,
    ImportIr,
    IrClipboardRead(Option<String>),
    UseImportedIr,
    ClearImportedIr,
    ExportIr,
    IrExported(Result<String, String>),
    CycleTheme,
    SystemThemeChanged(theme::Mode),
    ExportSvg,
    CancelExport,
    Exported {
        export_id: u64,
        result: Result<String, String>,
    },
    Viewport(ViewportMessage),
    ViewportInvalidated,
    Frame,
}

#[derive(Debug, Clone)]
enum ViewportMessage {
    GestureStarted,
    GestureChanged(Camera2d),
    GestureEnded(Camera2d),
    OrbitGestureStarted,
    OrbitGestureChanged(Orbit3d),
    OrbitGestureEnded(Orbit3d),
    WheelZoom {
        factor: f64,
        anchor: ScreenPoint,
        viewport: ViewportSize,
    },
    Fit,
    ZoomBy {
        factor: f64,
        viewport: ViewportSize,
    },
}

struct BrakenGui {
    source_editor: text_editor::Content,
    editor_draft: text_editor::Content,
    editor_draft_base: String,
    editor_draft_mode: bool,
    editor_open: bool,
    catalog_open: bool,
    layout_mode: LayoutMode,
    explore_scroll_offset: scrollable::AbsoluteOffset,
    diagnostics_open: bool,
    angle: f32,
    line_width_percent: f32,
    iterations: usize,
    iterations_input: String,
    iterations_notice: Option<String>,
    suggested_max_iterations: usize,
    seed_input: String,
    effective_seed: u64,
    seed_placeholder: String,
    seed_notice: Option<String>,
    derivation_semantics: DerivationSemantics,
    visualizer: VisualizerKind,
    turtle_config: Turtle2dConfig,
    orientation_anchor: Option<OrientationAnchor>,
    presets: Vec<presets::Preset>,
    preset_previews: Vec<PresetPreviewHandles>,
    selected_preset: Option<PresetChoice>,
    preset_search: String,
    request_id: u64,
    render: RenderState,
    canvas_cache: Cache,
    panel_expanded: bool,
    dock_view: DockView,
    imported_ir: Option<ImportedIr>,
    active_ir_json: Option<Arc<str>>,
    ir_notice: Option<String>,
    theme_preference: ThemePreference,
    system_theme: theme::Mode,
    generation_cache: GenerationCache,
    export_notice: Option<String>,
    export_sequence: u64,
    export_job: Option<ExportJob>,
    fps: u32,
    frames_since_sample: u32,
    fps_sample_started: RenderInstant,
    last_render_ms: Option<u128>,
    last_render_backend: Option<String>,
    iteration_route: Option<IterationRoute>,
    iteration_animation: Option<IterationAnimation>,
    source_changed_at: Option<RenderInstant>,
    // The live camera follows pointer input immediately. The settled camera is
    // the camera for which progressive display refinement is being rasterized.
    live_camera: Camera2d,
    settled_camera: Camera2d,
    orbit_3d: Orbit3d,
    reduced_motion: bool,
    autorotate_3d: bool,
    autorotate_frame_at: Option<RenderInstant>,
    pending_autorotate_request: Option<u64>,
    view_gesture_active: bool,
    wheel_changed_at: Option<RenderInstant>,
    // System/display epochs control fit-on-successful-source-replacement;
    // gesture epochs invalidate input baselines; raster epochs invalidate only
    // backend display resources, never derivation or visualization work.
    system_view_epoch: u64,
    displayed_view_epoch: u64,
    gesture_epoch: u64,
    raster_epoch: u64,
    render_coordinator: Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemePreference {
    Auto,
    Light,
    Dark,
}

impl ThemePreference {
    fn next(self) -> Self {
        match self {
            Self::Auto => Self::Light,
            Self::Light => Self::Dark,
            Self::Dark => Self::Auto,
        }
    }
}

fn resolve_theme_mode(preference: ThemePreference, system: theme::Mode) -> theme::Mode {
    match preference {
        ThemePreference::Auto => match system {
            theme::Mode::Dark => theme::Mode::Dark,
            theme::Mode::Light | theme::Mode::None => theme::Mode::Light,
        },
        ThemePreference::Light => theme::Mode::Light,
        ThemePreference::Dark => theme::Mode::Dark,
    }
}

#[derive(Debug, Default)]
struct RenderState {
    displayed: Option<RenderResult>,
    active: Option<RenderJob>,
    failure: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderPhase {
    Queued,
    Deriving,
    Visualizing,
    Measuring,
}

impl RenderPhase {
    const fn label(self) -> &'static str {
        match self {
            Self::Queued => "Queued",
            Self::Deriving => "Deriving",
            Self::Visualizing => "Building preview",
            Self::Measuring => "Preparing display",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderProgress {
    phase: RenderPhase,
    derivation: Option<CalculationProgress>,
    visualization: Option<Turtle2dProgress>,
}

#[derive(Debug, Clone)]
struct RenderJob {
    cancelled: Arc<AtomicBool>,
    calculation_cancel: CancellationToken,
    progress: Arc<Mutex<RenderProgress>>,
    started_at: RenderInstant,
}

impl Default for RenderJob {
    fn default() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            calculation_cancel: CancellationToken::new(),
            progress: Arc::new(Mutex::new(RenderProgress {
                phase: RenderPhase::Queued,
                derivation: None,
                visualization: None,
            })),
            started_at: RenderInstant::now(),
        }
    }
}

impl RenderJob {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.calculation_cancel.cancel();
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn set_phase(&self, phase: RenderPhase) {
        if let Ok(mut progress) = self.progress.lock() {
            progress.phase = phase;
        }
    }

    fn set_derivation_progress(&self, derivation: CalculationProgress) {
        if let Ok(mut progress) = self.progress.lock() {
            progress.phase = RenderPhase::Deriving;
            progress.derivation = Some(derivation);
        }
    }

    fn set_visualization_progress(&self, visualization: Turtle2dProgress) {
        if let Ok(mut progress) = self.progress.lock() {
            progress.phase = RenderPhase::Visualizing;
            progress.visualization = Some(visualization);
        }
    }

    fn set_spatial_visualization_progress(&self, visualization: Turtle3dProgress) {
        self.set_visualization_progress(Turtle2dProgress {
            items_processed: visualization.items_processed,
            modules_processed: visualization.modules_processed,
            branches_entered: visualization.branches_entered,
            lines_emitted: visualization.lines_emitted,
            polygons_emitted: visualization.polygons_emitted,
            active_branch_depth: visualization.active_branch_depth,
            max_branch_depth: visualization.max_branch_depth,
        });
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
enum RenderOutcome {
    Ready(RenderResult),
    TransitionReady(PreparedIterationTransition),
    Failed { request_id: u64, message: String },
    Cancelled { request_id: u64 },
}

#[derive(Debug, Clone)]
struct RenderResult {
    request_id: u64,
    view_epoch: u64,
    scene: Option<Arc<Scene2d>>,
    line_scene: Option<Arc<LineScene>>,
    spatial_scene: Option<Arc<SpatialScene>>,
    element_count: usize,
    refinement_started_at: RenderInstant,
    elapsed_ms: u128,
    // Human-readable derivation and visualization stage summary. The Iced
    // renderer remains an independently selected display stage.
    backend: String,
    bounds: RenderBounds,
    cache_insert: Option<(GenerationCacheKey, Arc<Generation>, DerivationInfo, usize)>,
    iteration: usize,
    identity: RenderIdentity,
    ir_tooling: IrToolingSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DockView {
    System,
    Ir,
}

#[derive(Debug, Clone)]
struct IrToolingSnapshot {
    disassembly: Arc<str>,
    json: Option<Arc<str>>,
}

#[derive(Debug, Clone)]
struct ImportedIr {
    json: Arc<str>,
    embedded_source: Option<String>,
    tooling: IrToolingSnapshot,
}

struct PreparedDisplay {
    scene: Option<Arc<Scene2d>>,
    line_scene: Option<Arc<LineScene>>,
    spatial_scene: Option<Arc<SpatialScene>>,
    element_count: usize,
    bounds: RenderBounds,
    visualization_backend: VisualizerBackend,
}

#[derive(Debug, Clone)]
struct ExportJob {
    export_id: u64,
    cancelled: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct RenderIdentity {
    source: Arc<str>,
    ir_json: Option<Arc<str>>,
    angle: f32,
    visualizer: VisualizerKind,
    turtle_config: Turtle2dConfig,
    orientation_anchor: Option<OrientationAnchor>,
    seed: u64,
    semantics: DerivationSemantics,
}

#[derive(Debug, Clone)]
struct PreparedIterationTransition {
    request_id: u64,
    from_iteration: usize,
    to_iteration: usize,
    scene: Arc<TransitionScene>,
    target: RenderResult,
}

#[derive(Debug, Clone)]
struct IterationAnimation {
    started_at: RenderInstant,
    paused_at: Option<RenderInstant>,
    paused_duration: Duration,
    progress: f32,
    prepared: PreparedIterationTransition,
}

#[derive(Debug, Clone)]
struct IterationRoute {
    request_id: u64,
    target_iteration: usize,
    identity: RenderIdentity,
}

impl RenderIdentity {
    fn same_as(&self, other: &Self) -> bool {
        self.source == other.source
            && self.ir_json == other.ir_json
            && self.angle.to_bits() == other.angle.to_bits()
            && self.visualizer == other.visualizer
            && turtle_configs_equal(&self.turtle_config, &other.turtle_config)
            && self.orientation_anchor == other.orientation_anchor
            && self.seed == other.seed
            && self.semantics == other.semantics
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GenerationCacheKey {
    source: Arc<str>,
    ir_json: Option<Arc<str>>,
    iterations: usize,
    seed: u64,
    semantics: DerivationSemantics,
}

struct GenerationCacheEntry {
    generation: Arc<Generation>,
    derivation: DerivationInfo,
    estimated_bytes: usize,
    last_used: u64,
}

#[derive(Debug, Clone)]
struct CachedGeneration {
    generation: Arc<Generation>,
    derivation: DerivationInfo,
}

struct GenerationCache {
    entries: HashMap<GenerationCacheKey, GenerationCacheEntry>,
    budget: usize,
    estimated_bytes: usize,
    clock: u64,
}

impl GenerationCache {
    fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            budget,
            estimated_bytes: 0,
            clock: 0,
        }
    }

    fn get(&mut self, key: &GenerationCacheKey) -> Option<CachedGeneration> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(key)?;
        entry.last_used = self.clock;
        Some(CachedGeneration {
            generation: Arc::clone(&entry.generation),
            derivation: entry.derivation,
        })
    }

    fn insert(
        &mut self,
        key: GenerationCacheKey,
        generation: Arc<Generation>,
        derivation: DerivationInfo,
        estimated_bytes: usize,
    ) {
        if estimated_bytes > self.budget {
            return;
        }

        self.clock = self.clock.wrapping_add(1);
        if let Some(previous) = self.entries.remove(&key) {
            self.estimated_bytes = self
                .estimated_bytes
                .saturating_sub(previous.estimated_bytes);
        }
        self.estimated_bytes += estimated_bytes;
        self.entries.insert(
            key,
            GenerationCacheEntry {
                generation,
                derivation,
                estimated_bytes,
                last_used: self.clock,
            },
        );

        while self.estimated_bytes > self.budget {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.estimated_bytes = self.estimated_bytes.saturating_sub(removed.estimated_bytes);
            }
        }
    }
}

fn estimate_cache_entry_bytes(key: &GenerationCacheKey, items: usize, modules: usize) -> usize {
    key.source
        .len()
        .saturating_add(key.ir_json.as_ref().map_or(0, |json| json.len()))
        .saturating_add(std::mem::size_of::<GenerationCacheKey>())
        .saturating_add(std::mem::size_of::<DerivationInfo>())
        .saturating_add(items.saturating_mul(64))
        .saturating_add(modules.saturating_mul(32))
}

#[derive(Debug, Clone, Copy)]
struct RenderLine {
    start: (f32, f32),
    end: (f32, f32),
    width: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct RenderBounds {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

impl From<RenderBounds> for ViewBounds {
    fn from(bounds: RenderBounds) -> Self {
        Self {
            min_x: f64::from(bounds.min_x),
            max_x: f64::from(bounds.max_x),
            min_y: f64::from(bounds.min_y),
            max_y: f64::from(bounds.max_y),
        }
    }
}

fn render_bounds_extent(bounds: RenderBounds) -> (f64, f64) {
    (
        f64::from((bounds.max_x - bounds.min_x).max(0.0)),
        f64::from((bounds.max_y - bounds.min_y).max(0.0)),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PresetChoice(usize);

struct PresetPreviewHandles {
    light: svg::Handle,
    dark: svg::Handle,
}

impl PresetPreviewHandles {
    fn for_theme(&self, dark: bool) -> &svg::Handle {
        if dark { &self.dark } else { &self.light }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayoutMode {
    Desktop,
    Mobile,
}

fn layout_mode(width: f32) -> LayoutMode {
    if width >= MOBILE_BREAKPOINT {
        LayoutMode::Desktop
    } else {
        LayoutMode::Mobile
    }
}

fn initial_layout_mode() -> LayoutMode {
    #[cfg(target_arch = "wasm32")]
    if let Some(width) = web_sys::window()
        .and_then(|window| window.inner_width().ok())
        .and_then(|value| value.as_f64())
    {
        return layout_mode(width as f32);
    }
    LayoutMode::Desktop
}

impl fmt::Display for PresetChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Preset {}", self.0 + 1)
    }
}

#[cfg(target_arch = "wasm32")]
fn prefers_reduced_motion() -> bool {
    web_sys::window()
        .and_then(|window| {
            window
                .match_media("(prefers-reduced-motion: reduce)")
                .ok()
                .flatten()
        })
        .is_some_and(|query| query.matches())
}

#[cfg(not(target_arch = "wasm32"))]
const fn prefers_reduced_motion() -> bool {
    // Iced 0.14 exposes system color-scheme changes, but no portable native
    // reduced-motion preference. Browser builds use the standard media query.
    false
}

impl BrakenGui {
    fn new() -> (Self, Task<Message>) {
        let presets = presets::get_presets();
        let preset_previews = presets
            .iter()
            .map(|preset| {
                let preview = |palette| {
                    let bytes =
                        generated_preset_previews::preset_preview_svg(&preset.name, palette)
                            .unwrap_or_else(|| {
                                panic!(
                                    "{} is missing its generated {palette:?} preview",
                                    preset.name
                                )
                            });
                    svg::Handle::from_memory(bytes)
                };
                PresetPreviewHandles {
                    light: preview(Palette::Light),
                    dark: preview(Palette::Dark),
                }
            })
            .collect();
        let default = presets[0].clone();

        let mut app = Self {
            source_editor: text_editor::Content::with_text(&default.source),
            editor_draft: text_editor::Content::with_text(&default.source),
            editor_draft_base: default.source.clone(),
            editor_draft_mode: false,
            editor_open: false,
            catalog_open: false,
            layout_mode: initial_layout_mode(),
            explore_scroll_offset: scrollable::AbsoluteOffset::default(),
            diagnostics_open: false,
            angle: default.angle as f32,
            line_width_percent: DEFAULT_LINE_WIDTH_PERCENT,
            iterations: default.iters as usize,
            iterations_input: default.iters.to_string(),
            iterations_notice: None,
            suggested_max_iterations: default.max_iters as usize,
            seed_input: String::new(),
            effective_seed: 0,
            seed_placeholder: String::from("0"),
            seed_notice: None,
            derivation_semantics: DerivationSemantics {
                float_width: FloatWidth::F32,
                ambiguous_rules: AmbiguousRulePolicy::Uniform,
            },
            visualizer: default.visualizer,
            turtle_config: default.turtle_config,
            orientation_anchor: default.orientation_anchor,
            presets,
            preset_previews,
            selected_preset: Some(PresetChoice(0)),
            preset_search: String::new(),
            request_id: 0,
            render: RenderState::default(),
            canvas_cache: Cache::new(),
            panel_expanded: false,
            dock_view: DockView::System,
            imported_ir: None,
            active_ir_json: None,
            ir_notice: None,
            theme_preference: ThemePreference::Auto,
            system_theme: theme::Mode::None,
            generation_cache: GenerationCache::new(GENERATION_CACHE_BUDGET),
            export_notice: None,
            export_sequence: 0,
            export_job: None,
            fps: 0,
            frames_since_sample: 0,
            fps_sample_started: RenderInstant::now(),
            last_render_ms: None,
            last_render_backend: None,
            iteration_route: None,
            iteration_animation: None,
            source_changed_at: None,
            live_camera: Camera2d::default(),
            settled_camera: Camera2d::default(),
            orbit_3d: Orbit3d::canonical(),
            reduced_motion: prefers_reduced_motion(),
            autorotate_3d: false,
            autorotate_frame_at: None,
            pending_autorotate_request: None,
            view_gesture_active: false,
            wheel_changed_at: None,
            system_view_epoch: 1,
            displayed_view_epoch: 0,
            gesture_epoch: 0,
            raster_epoch: 0,
            render_coordinator: Worker::new(),
        };
        app.regenerate_seed();

        let initial_render = app.iteration_render_task();
        if app.visualizer == VisualizerKind::Turtle3d && !app.reduced_motion {
            app.pending_autorotate_request = Some(app.request_id);
        }
        let task = Task::batch([
            initial_render,
            iced::system::theme().map(Message::SystemThemeChanged),
        ]);

        (app, task)
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::DraftEdited(action) => {
                self.editor_draft.perform(action);
                Task::none()
            }
            Message::ToggleEditor(layout) => {
                self.layout_mode = layout;
                let closing = self.editor_open;
                if self.editor_open {
                    self.editor_open = false;
                } else {
                    self.sync_editor_draft();
                    self.editor_draft_mode =
                        layout == LayoutMode::Mobile || self.draft_is_modified();
                    self.editor_open = true;
                    self.catalog_open = false;
                    self.dock_view = DockView::System;
                }
                self.panels_changed();
                if closing {
                    self.restore_explore_scroll()
                } else {
                    Task::none()
                }
            }
            Message::CloseEditor => {
                self.editor_open = false;
                self.panels_changed();
                self.restore_explore_scroll()
            }
            Message::ApplyDraft => {
                self.sync_editor_draft();
                let modified = self.draft_is_modified();
                if modified {
                    let source = self.editor_draft.text();
                    self.source_editor = text_editor::Content::with_text(&source);
                    self.editor_draft_base = source;
                    self.selected_preset = None;
                    self.active_ir_json = None;
                    self.pending_autorotate_request = None;
                    self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                    self.cancel_obsolete_work();
                }
                self.editor_open = false;
                self.panels_changed();
                self.explore_scroll_offset = scrollable::AbsoluteOffset::default();
                let scroll = self.restore_explore_scroll();
                if modified {
                    Task::batch([self.render_task(), scroll])
                } else {
                    scroll
                }
            }
            Message::OpenCatalog => {
                self.catalog_open = true;
                self.editor_open = false;
                self.panels_changed();
                Task::none()
            }
            Message::CloseCatalog => {
                self.catalog_open = false;
                self.panels_changed();
                self.restore_explore_scroll()
            }
            Message::CloseOverlay => {
                if self.editor_open {
                    self.editor_open = false;
                } else if self.catalog_open {
                    self.catalog_open = false;
                } else {
                    return Task::none();
                }
                self.panels_changed();
                self.restore_explore_scroll()
            }
            Message::WindowResized(size) => {
                self.layout_mode = layout_mode(size.width);
                if self.layout_mode == LayoutMode::Mobile
                    && self.editor_open
                    && !self.editor_draft_mode
                {
                    self.sync_editor_draft();
                    self.editor_draft_mode = true;
                }
                self.panels_changed();
                Task::none()
            }
            Message::ExploreScrolled(offset) => {
                if self.canvas_is_visible() {
                    self.explore_scroll_offset = offset;
                }
                Task::none()
            }
            Message::ToggleDiagnostics => {
                self.diagnostics_open = !self.diagnostics_open;
                Task::none()
            }
            Message::SourceEdited(action) => {
                if self.editor_open && self.editor_draft_mode {
                    self.editor_draft.perform(action);
                    return Task::none();
                }
                let previous = self.source_editor.text();
                self.source_editor.perform(action);
                if self.source_editor.text() == previous {
                    return Task::none();
                }
                self.selected_preset = None;
                self.active_ir_json = None;
                self.pending_autorotate_request = None;
                self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                self.cancel_obsolete_work();
                self.source_changed_at = Some(RenderInstant::now());
                Task::none()
            }
            Message::AngleChanged(value) => {
                self.angle = value.clamp(MIN_ANGLE, MAX_ANGLE);
                self.selected_preset = None;
                self.render_task()
            }
            Message::AngleDecrement => {
                self.angle = adjacent_integer_angle(self.angle, -1.0);
                self.selected_preset = None;
                self.render_task()
            }
            Message::AngleIncrement => {
                self.angle = adjacent_integer_angle(self.angle, 1.0);
                self.selected_preset = None;
                self.render_task()
            }
            Message::AngleScrolled(delta) => {
                self.angle = stepped_fractional_value(self.angle, delta, MIN_ANGLE, MAX_ANGLE);
                self.selected_preset = None;
                self.render_task()
            }
            Message::LineWidthChanged(value) => {
                let value = if value.is_finite() {
                    value.clamp(MIN_LINE_WIDTH_PERCENT, MAX_LINE_WIDTH_PERCENT)
                } else {
                    DEFAULT_LINE_WIDTH_PERCENT
                };
                if self.line_width_percent == value {
                    return Task::none();
                }
                self.line_width_percent = value;
                if let Some(export) = self.export_job.take() {
                    export.cancelled.store(true, Ordering::Release);
                    self.export_notice = None;
                }
                self.invalidate_view_raster();
                Task::none()
            }
            Message::IterationsChanged(value) => {
                self.set_iterations(value.round().max(0.0) as usize);
                self.selected_preset = None;
                self.iteration_render_task()
            }
            Message::IterationsInputChanged(value) => {
                self.iterations_input = value;
                if self.iterations_input.is_empty() {
                    self.iterations_notice = None;
                    return Task::none();
                }

                match self.iterations_input.parse::<usize>() {
                    Ok(iterations) => {
                        self.iterations = iterations;
                        self.iterations_notice = None;
                        self.selected_preset = None;
                        self.render_task()
                    }
                    Err(_) => {
                        self.iterations_notice = Some(String::from(
                            "Iterations must be a non-negative whole number",
                        ));
                        Task::none()
                    }
                }
            }
            Message::IterationsDecrement => {
                self.set_iterations(self.iterations.saturating_sub(1));
                self.selected_preset = None;
                self.iteration_render_task()
            }
            Message::IterationsIncrement => {
                self.set_iterations(self.iterations.saturating_add(1));
                self.selected_preset = None;
                self.iteration_render_task()
            }
            Message::IterationsScrolled(delta) => {
                let amount = scroll_amount(delta);
                if amount != 0.0 {
                    self.set_iterations(if amount.is_sign_positive() {
                        self.iterations.saturating_add(1)
                    } else {
                        self.iterations.saturating_sub(1)
                    });
                }
                self.selected_preset = None;
                self.iteration_render_task()
            }
            Message::SeedChanged(value) => {
                self.seed_input = value;
                if self.seed_input.is_empty() {
                    if let Ok(seed) = self.seed_placeholder.parse::<u64>() {
                        self.effective_seed = seed;
                    }
                    self.seed_notice = None;
                    return self.render_task();
                }

                match self.seed_input.parse::<u64>() {
                    Ok(seed) => {
                        self.effective_seed = seed;
                        self.seed_notice = None;
                        self.render_task()
                    }
                    Err(_) => {
                        self.seed_notice = Some(String::from(
                            "Seed must be a decimal integer from 0 to 18446744073709551615",
                        ));
                        Task::none()
                    }
                }
            }
            Message::RandomizeSeed => {
                self.regenerate_seed();
                self.render_task()
            }
            Message::FloatWidthChanged(float_width) => {
                if self.derivation_semantics.float_width == float_width {
                    Task::none()
                } else {
                    self.derivation_semantics.float_width = float_width;
                    self.render_task()
                }
            }
            Message::AmbiguousRulesChanged(ambiguous_rules) => {
                if self.derivation_semantics.ambiguous_rules == ambiguous_rules {
                    Task::none()
                } else {
                    self.derivation_semantics.ambiguous_rules = ambiguous_rules;
                    self.render_task()
                }
            }
            Message::PresetSearchChanged(query) => {
                self.preset_search = query;
                Task::none()
            }
            Message::ClearPresetSearch => {
                self.preset_search.clear();
                Task::none()
            }
            Message::PresetSelected(choice) => {
                if let Some(preset) = self.presets.get(choice.0).cloned() {
                    self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                    self.source_editor = text_editor::Content::with_text(&preset.source);
                    self.reset_editor_draft();
                    self.catalog_open = false;
                    self.panels_changed();
                    self.set_iterations(preset.iters as usize);
                    self.suggested_max_iterations = preset.max_iters as usize;
                    self.visualizer = preset.visualizer;
                    self.turtle_config = preset.turtle_config;
                    self.orientation_anchor = preset.orientation_anchor;
                    self.selected_preset = Some(choice);
                    self.imported_ir = None;
                    self.active_ir_json = None;
                    self.ir_notice = None;
                    self.regenerate_seed();
                    self.angle = preset.angle as f32;
                    let autorotate =
                        preset.visualizer == VisualizerKind::Turtle3d && !self.reduced_motion;
                    let task = self.iteration_render_task();
                    self.pending_autorotate_request = autorotate.then_some(self.request_id);
                    self.explore_scroll_offset = scrollable::AbsoluteOffset::default();
                    return Task::batch([task, self.restore_explore_scroll()]);
                }

                Task::none()
            }
            Message::Rendered(RenderOutcome::Ready(mut result)) => {
                if result.request_id == self.request_id {
                    if result.view_epoch != self.displayed_view_epoch {
                        self.reset_view(
                            result.spatial_scene.is_some(),
                            result.request_id,
                            RenderInstant::now(),
                        );
                        self.displayed_view_epoch = result.view_epoch;
                    } else {
                        // A replacement scene may have different bounds. End any
                        // in-flight gesture at the current normalized camera so
                        // its old baseline cannot jump against the new scene.
                        self.settled_camera = self.live_camera;
                        self.view_gesture_active = false;
                        self.wheel_changed_at = None;
                        self.gesture_epoch = self.gesture_epoch.wrapping_add(1);
                    }
                    result.refinement_started_at = RenderInstant::now();
                    if let Some((key, generation, derivation, estimated_bytes)) =
                        result.cache_insert.take()
                    {
                        self.generation_cache
                            .insert(key, generation, derivation, estimated_bytes);
                    }
                    self.last_render_ms = Some(result.elapsed_ms);
                    self.last_render_backend = Some(result.backend.clone());
                    self.render.displayed = Some(result);
                    self.render.active = None;
                    self.render.failure = None;
                    self.canvas_cache.clear();
                    return self.queue_next_iteration_transition();
                }
                Task::none()
            }
            Message::Rendered(RenderOutcome::TransitionReady(prepared)) => {
                if prepared.request_id == self.request_id
                    && self
                        .iteration_route
                        .as_ref()
                        .is_some_and(|route| route.request_id == prepared.request_id)
                {
                    self.render.active = None;
                    self.render.failure = None;
                    self.iteration_animation = Some(IterationAnimation {
                        started_at: RenderInstant::now(),
                        paused_at: (!self.canvas_is_visible()).then(RenderInstant::now),
                        paused_duration: Duration::ZERO,
                        progress: 0.0,
                        prepared,
                    });
                    self.canvas_cache.clear();
                }
                Task::none()
            }
            Message::Rendered(RenderOutcome::Failed {
                request_id,
                message,
            }) => {
                if request_id == self.request_id {
                    if self.pending_autorotate_request == Some(request_id) {
                        self.pending_autorotate_request = None;
                    }
                    self.render.active = None;
                    self.render.failure = Some(message);
                    self.iteration_route = None;
                    self.iteration_animation = None;
                }
                Task::none()
            }
            Message::Rendered(RenderOutcome::Cancelled { request_id }) => {
                if request_id == self.request_id {
                    if self.pending_autorotate_request == Some(request_id) {
                        self.pending_autorotate_request = None;
                    }
                    self.render.active = None;
                    self.render.failure = None;
                    self.iteration_route = None;
                    self.iteration_animation = None;
                }
                Task::none()
            }
            Message::CancelRender => {
                self.pending_autorotate_request = None;
                if self.render.active.take().is_some() {
                    self.render_coordinator.cancel_current();
                    self.request_id = self.request_id.wrapping_add(1);
                    self.iteration_route = None;
                    self.iteration_animation = None;
                    self.render.failure = None;
                } else if self.iteration_animation.take().is_some() {
                    self.iteration_route = None;
                    self.request_id = self.request_id.wrapping_add(1);
                    self.canvas_cache.clear();
                } else if let Some(lines) = self
                    .render
                    .displayed
                    .as_ref()
                    .and_then(|result| result.line_scene.as_ref())
                    .filter(|lines| lines.is_refining())
                {
                    lines.cancel_refinement();
                } else if let Some(spatial) = self
                    .render
                    .displayed
                    .as_ref()
                    .and_then(|result| result.spatial_scene.as_ref())
                    .filter(|scene| scene.is_refining())
                {
                    spatial.cancel_refinement();
                }
                Task::none()
            }
            Message::ToggleSettings => {
                self.panel_expanded = !self.panel_expanded;
                self.panels_changed();
                Task::none()
            }
            Message::DockViewSelected(view) => {
                self.dock_view = view;
                Task::none()
            }
            Message::CopyIr => self
                .current_ir_tooling()
                .map_or_else(Task::none, |tooling| {
                    iced::clipboard::write(tooling.disassembly.to_string())
                }),
            Message::ImportIr => iced::clipboard::read().map(Message::IrClipboardRead),
            Message::IrClipboardRead(contents) => {
                let Some(json) = contents else {
                    self.ir_notice = Some(String::from("The clipboard does not contain text"));
                    return Task::none();
                };
                let imported = ir_tooling::import_json(&json)
                    .map(|imported| ImportedIr {
                        json: Arc::from(json),
                        embedded_source: imported.embedded_source,
                        tooling: imported.tooling.into(),
                    })
                    .map_err(|error| error.to_string());
                match imported {
                    Ok(imported) => {
                        self.imported_ir = Some(imported);
                        self.ir_notice = Some(String::from("Imported IR is valid"));
                    }
                    Err(error) => self.ir_notice = Some(error),
                }
                Task::none()
            }
            Message::UseImportedIr => {
                let Some(imported) = self.imported_ir.clone() else {
                    return Task::none();
                };
                self.active_ir_json = Some(imported.json);
                if let Some(source) = imported.embedded_source {
                    self.source_editor = text_editor::Content::with_text(&source);
                }
                self.reset_editor_draft();
                self.selected_preset = None;
                self.pending_autorotate_request = None;
                self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                self.cancel_obsolete_work();
                self.render_task()
            }
            Message::ClearImportedIr => {
                let was_active = self.active_ir_json.take().is_some();
                self.imported_ir = None;
                self.ir_notice = None;
                if was_active {
                    self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                    self.render_task()
                } else {
                    Task::none()
                }
            }
            Message::ExportIr => {
                let Some(json) = self
                    .current_ir_tooling()
                    .and_then(|tooling| tooling.json.as_ref())
                    .map(Arc::clone)
                else {
                    self.export_notice =
                        Some(String::from("IR export is unavailable for this result"));
                    return Task::none();
                };
                self.export_notice = Some(String::from("Preparing IR document…"));
                Task::perform(export_ir_json(json.to_string()), Message::IrExported)
            }
            Message::IrExported(result) => {
                self.export_notice = Some(match result {
                    Ok(message) => message,
                    Err(error) => format!("IR export failed: {error}"),
                });
                Task::none()
            }
            Message::CycleTheme => {
                let was_dark = self.effective_dark();
                self.theme_preference = self.theme_preference.next();
                if was_dark != self.effective_dark() {
                    self.restart_spatial_refinement();
                    self.invalidate_view_raster();
                }
                Task::none()
            }
            Message::SystemThemeChanged(mode) => {
                let was_dark = self.effective_dark();
                self.system_theme = mode;
                if was_dark != self.effective_dark() {
                    self.restart_spatial_refinement();
                    self.invalidate_view_raster();
                }
                Task::none()
            }
            Message::ExportSvg => {
                let Some(result) = &self.render.displayed else {
                    return Task::none();
                };
                let palette = if self.effective_dark() {
                    Palette::Dark
                } else {
                    Palette::Light
                };
                let scene = result.scene.as_ref().map(Arc::clone);
                let line_scene = result.line_scene.as_ref().map(Arc::clone);
                let spatial_scene = result.spatial_scene.as_ref().map(Arc::clone);
                let orbit = self.orbit_3d;
                let line_width_scale = self.line_width_scale();
                if let Some(previous) = self.export_job.take() {
                    previous.cancelled.store(true, Ordering::Release);
                }
                self.export_sequence = self.export_sequence.wrapping_add(1);
                let export_id = self.export_sequence;
                let cancelled = Arc::new(AtomicBool::new(false));
                self.export_job = Some(ExportJob {
                    export_id,
                    cancelled: Arc::clone(&cancelled),
                });
                self.export_notice = Some(String::from("Preparing SVG…"));
                Task::perform(
                    prepare_and_export_svg(
                        line_scene,
                        scene,
                        spatial_scene,
                        orbit,
                        palette,
                        line_width_scale,
                        cancelled,
                    ),
                    move |result| Message::Exported { export_id, result },
                )
            }
            Message::CancelExport => {
                if let Some(export) = self.export_job.take() {
                    export.cancelled.store(true, Ordering::Release);
                    self.export_notice = Some(String::from("SVG export cancelled"));
                }
                Task::none()
            }
            Message::Exported { export_id, result } => {
                if self
                    .export_job
                    .as_ref()
                    .is_some_and(|job| job.export_id == export_id)
                {
                    self.export_job = None;
                    self.export_notice = Some(match result {
                        Ok(message) => message,
                        Err(error) => format!("SVG export failed: {error}"),
                    });
                }
                Task::none()
            }
            Message::Viewport(message) => {
                self.update_viewport(message);
                Task::none()
            }
            Message::ViewportInvalidated => {
                self.invalidate_view_raster();
                Task::none()
            }
            Message::Frame => {
                self.frames_since_sample = self.frames_since_sample.saturating_add(1);
                let now = RenderInstant::now();
                if self.autorotate_3d
                    && self.canvas_is_visible()
                    && !self.view_gesture_active
                    && self
                        .render
                        .displayed
                        .as_ref()
                        .is_some_and(|result| result.spatial_scene.is_some())
                {
                    let delta = self
                        .autorotate_frame_at
                        .map_or(Duration::ZERO, |previous| now.duration_since(previous))
                        .min(MAX_AUTOROTATE_FRAME_DELTA);
                    self.orbit_3d = self
                        .orbit_3d
                        .autorotated(delta.as_secs_f64() * AUTOROTATE_RADIANS_PER_SECOND);
                    self.autorotate_frame_at = Some(now);
                    self.canvas_cache.clear();
                }
                let elapsed = now.duration_since(self.fps_sample_started);
                if elapsed.as_secs_f64() >= 1.0 {
                    self.fps =
                        (self.frames_since_sample as f64 / elapsed.as_secs_f64()).round() as u32;
                    self.frames_since_sample = 0;
                    self.fps_sample_started = now;
                }

                if self.wheel_changed_at.is_some_and(|changed_at| {
                    now.duration_since(changed_at) >= VIEW_WHEEL_SETTLE_DELAY
                }) {
                    self.wheel_changed_at = None;
                    self.settle_view();
                }

                if self.source_changed_at.is_some_and(|changed_at| {
                    now.duration_since(changed_at) >= SOURCE_EDIT_DEBOUNCE
                }) {
                    self.source_changed_at = None;
                    return self.render_task();
                }

                if let Some(mut animation) = self.iteration_animation.take() {
                    if self.view_is_interacting() || !self.canvas_is_visible() {
                        animation.paused_at.get_or_insert(now);
                        self.iteration_animation = Some(animation);
                        return Task::none();
                    }
                    if let Some(paused_at) = animation.paused_at.take() {
                        animation.paused_duration += now.duration_since(paused_at);
                    }
                    let elapsed = now
                        .duration_since(animation.started_at)
                        .saturating_sub(animation.paused_duration);
                    let linear = (elapsed.as_secs_f32()
                        / ITERATION_TRANSITION_DURATION.as_secs_f32())
                    .clamp(0.0, 1.0);
                    animation.progress = iteration_ease(linear);
                    self.canvas_cache.clear();
                    if linear < 1.0 {
                        self.iteration_animation = Some(animation);
                        return Task::none();
                    }

                    let mut result = animation.prepared.target;
                    result.refinement_started_at = now;
                    if let Some((key, generation, derivation, estimated_bytes)) =
                        result.cache_insert.take()
                    {
                        self.generation_cache
                            .insert(key, generation, derivation, estimated_bytes);
                    }
                    self.last_render_ms = Some(result.elapsed_ms);
                    self.last_render_backend = Some(result.backend.clone());
                    self.render.displayed = Some(result);
                    return self.queue_next_iteration_transition();
                }
                Task::none()
            }
        }
    }

    fn restore_explore_scroll(&self) -> Task<Message> {
        if self.layout_mode == LayoutMode::Mobile {
            iced::widget::operation::scroll_to("explore-scroll", self.explore_scroll_offset)
        } else {
            Task::none()
        }
    }

    fn draft_is_modified(&self) -> bool {
        self.editor_draft.text() != self.editor_draft_base
    }

    fn reset_editor_draft(&mut self) {
        self.editor_draft_base = self.source_editor.text();
        self.editor_draft = text_editor::Content::with_text(&self.editor_draft_base);
    }

    fn sync_editor_draft(&mut self) {
        if self.editor_draft_base != self.source_editor.text() {
            self.reset_editor_draft();
        }
    }

    fn canvas_is_visible(&self) -> bool {
        self.layout_mode == LayoutMode::Desktop || !(self.editor_open || self.catalog_open)
    }

    /// Panel transitions invalidate display resources without replacing the
    /// accepted scene, camera, cache, or pending derivation. Hidden mobile views
    /// pause animations instead of consuming their duration offscreen.
    fn panels_changed(&mut self) {
        let visible = self.canvas_is_visible();
        if !visible {
            // The input widget is removed in full-screen mobile panels. A
            // visible canvas instead retains ownership until the last release,
            // including when a resize rebases the controller's gesture.
            self.view_gesture_active = false;
            self.wheel_changed_at = None;
        }
        let can_animate = visible && !self.view_is_interacting();
        let now = RenderInstant::now();
        if let Some(animation) = &mut self.iteration_animation {
            if can_animate {
                if let Some(paused_at) = animation.paused_at.take() {
                    animation.paused_duration += now.duration_since(paused_at);
                }
            } else {
                animation.paused_at.get_or_insert(now);
            }
        }
        self.autorotate_frame_at = None;
        self.gesture_epoch = self.gesture_epoch.wrapping_add(1);
        if !self.view_is_interacting() {
            self.settle_view();
        }
        self.invalidate_view_raster();
    }

    fn effective_theme_mode(&self) -> theme::Mode {
        resolve_theme_mode(self.theme_preference, self.system_theme)
    }

    fn effective_dark(&self) -> bool {
        self.effective_theme_mode() == theme::Mode::Dark
    }

    fn view_bounds(&self) -> ViewBounds {
        self.render
            .displayed
            .as_ref()
            .map_or_else(ViewBounds::default, |result| result.bounds.into())
    }

    fn view_is_interacting(&self) -> bool {
        self.view_gesture_active || self.wheel_changed_at.is_some()
    }

    fn reset_view(&mut self, spatial: bool, request_id: u64, now: RenderInstant) {
        self.live_camera = Camera2d::fit();
        self.settled_camera = self.live_camera;
        self.orbit_3d = Orbit3d::canonical();
        self.autorotate_3d =
            spatial && !self.reduced_motion && self.pending_autorotate_request == Some(request_id);
        self.autorotate_frame_at = self.autorotate_3d.then_some(now);
        self.pending_autorotate_request = None;
        self.view_gesture_active = false;
        self.wheel_changed_at = None;
        self.gesture_epoch = self.gesture_epoch.wrapping_add(1);
        self.canvas_cache.clear();
    }

    /// Commits the live camera used for input to the raster camera and restarts
    /// only display refinement. Navigation never regenerates or revisualizes the
    /// L-system.
    fn settle_view(&mut self) {
        let changed = self.settled_camera != self.live_camera;
        self.settled_camera = self.live_camera;
        if changed
            && let Some(result) = self.render.displayed.as_mut()
            && let Some(lines) = result.line_scene.as_ref()
        {
            lines.restart_view_refinement();
            result.refinement_started_at = RenderInstant::now();
        }
    }

    fn invalidate_view_raster(&mut self) {
        self.raster_epoch = self.raster_epoch.wrapping_add(1);
        self.canvas_cache.clear();
        if let Some(result) = self.render.displayed.as_mut()
            && let Some(lines) = result.line_scene.as_ref()
        {
            lines.restart_view_refinement();
            result.refinement_started_at = RenderInstant::now();
        }
    }

    fn restart_spatial_refinement(&mut self) {
        if let Some(result) = self.render.displayed.as_mut()
            && let Some(spatial) = result.spatial_scene.as_ref()
        {
            spatial.restart_refinement();
            result.refinement_started_at = RenderInstant::now();
        }
    }

    fn line_width_scale(&self) -> f32 {
        self.line_width_percent / 100.0
    }

    fn update_viewport(&mut self, message: ViewportMessage) {
        match message {
            ViewportMessage::GestureStarted => {
                self.pending_autorotate_request = None;
                self.view_gesture_active = true;
                self.wheel_changed_at = None;
            }
            ViewportMessage::OrbitGestureStarted => {
                self.pending_autorotate_request = None;
                self.autorotate_3d = false;
                self.autorotate_frame_at = None;
                self.view_gesture_active = true;
                self.wheel_changed_at = None;
            }
            ViewportMessage::OrbitGestureChanged(orbit) => {
                self.pending_autorotate_request = None;
                self.orbit_3d = orbit;
                self.autorotate_3d = false;
                self.autorotate_frame_at = None;
                self.view_gesture_active = true;
                self.canvas_cache.clear();
            }
            ViewportMessage::OrbitGestureEnded(orbit) => {
                self.pending_autorotate_request = None;
                self.orbit_3d = orbit;
                self.autorotate_3d = false;
                self.autorotate_frame_at = None;
                self.view_gesture_active = false;
                self.canvas_cache.clear();
                if let Some(result) = self.render.displayed.as_mut()
                    && result
                        .spatial_scene
                        .as_ref()
                        .is_some_and(|scene| scene.is_refining())
                {
                    result.refinement_started_at = RenderInstant::now();
                }
            }
            ViewportMessage::GestureChanged(camera) => {
                self.live_camera = camera;
                self.view_gesture_active = true;
                self.canvas_cache.clear();
            }
            ViewportMessage::GestureEnded(camera) => {
                self.live_camera = camera;
                self.view_gesture_active = false;
                self.wheel_changed_at = None;
                self.canvas_cache.clear();
                self.settle_view();
            }
            ViewportMessage::WheelZoom {
                factor,
                anchor,
                viewport,
            } => {
                self.pending_autorotate_request = None;
                self.live_camera =
                    self.live_camera
                        .zoom_about(factor, anchor, self.view_bounds(), viewport);
                self.wheel_changed_at = Some(RenderInstant::now());
                self.canvas_cache.clear();
            }
            ViewportMessage::Fit => {
                self.live_camera = Camera2d::fit();
                self.view_gesture_active = false;
                self.wheel_changed_at = None;
                self.gesture_epoch = self.gesture_epoch.wrapping_add(1);
                self.canvas_cache.clear();
                self.settle_view();
            }
            ViewportMessage::ZoomBy { factor, viewport } => {
                self.live_camera =
                    self.live_camera
                        .zoom_by_in(factor, self.view_bounds(), viewport);
                self.view_gesture_active = false;
                self.wheel_changed_at = None;
                self.gesture_epoch = self.gesture_epoch.wrapping_add(1);
                self.canvas_cache.clear();
                self.settle_view();
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        container_widget(responsive(|size| self.responsive_layout(size)))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(app_background)
            .into()
    }

    fn render_task(&mut self) -> Task<Message> {
        self.pending_autorotate_request = None;
        self.source_changed_at = None;
        self.iteration_route = None;
        self.iteration_animation = None;
        self.request_id = self.request_id.wrapping_add(1);
        self.queue_exact_iteration(self.request_id, self.iterations)
    }

    fn iteration_render_task(&mut self) -> Task<Message> {
        if self.visualizer != VisualizerKind::Turtle2d {
            return self.render_task();
        }
        self.source_changed_at = None;
        self.iteration_animation = None;
        self.render_coordinator.cancel_current();
        self.request_id = self.request_id.wrapping_add(1);
        let identity = self.current_render_identity();
        self.iteration_route = Some(IterationRoute {
            request_id: self.request_id,
            target_iteration: self.iterations,
            identity: identity.clone(),
        });

        let can_continue = self
            .render
            .displayed
            .as_ref()
            .is_some_and(|result| result.identity.same_as(&identity));
        if can_continue {
            return self.queue_next_iteration_transition();
        }
        self.queue_exact_iteration(self.request_id, 0)
    }

    fn queue_exact_iteration(&mut self, request_id: u64, iteration: usize) -> Task<Message> {
        let key = GenerationCacheKey {
            source: Arc::from(self.source_editor.text()),
            ir_json: self.active_ir_json.clone(),
            iterations: iteration,
            seed: self.effective_seed,
            semantics: self.derivation_semantics,
        };
        let cached_generation = self.generation_cache.get(&key);
        let job = RenderJob::default();
        let request = AppVisualizeRequest {
            request_id,
            view_epoch: self.system_view_epoch,
            key,
            angle: self.angle,
            visualizer: self.visualizer,
            turtle_config: self.turtle_config.clone(),
            orientation_anchor: self.orientation_anchor,
            seed: self.effective_seed,
            cached_generation,
            transition: None,
            transition_source: None,
            job: job.clone(),
        };

        self.queue_render_request(request, job)
    }

    fn queue_next_iteration_transition(&mut self) -> Task<Message> {
        let Some(route) = self.iteration_route.clone() else {
            return Task::none();
        };
        if route.request_id != self.request_id {
            self.iteration_route = None;
            return Task::none();
        }
        let Some(displayed) = self.render.displayed.as_ref() else {
            return Task::none();
        };
        if !displayed.identity.same_as(&route.identity) {
            self.iteration_route = None;
            return Task::none();
        }
        let from_iteration = displayed.iteration;
        if from_iteration == route.target_iteration {
            self.iteration_route = None;
            return Task::none();
        }
        let to_iteration = if from_iteration < route.target_iteration {
            from_iteration.saturating_add(1)
        } else {
            from_iteration.saturating_sub(1)
        };
        let source_key = GenerationCacheKey {
            source: Arc::clone(&route.identity.source),
            ir_json: route.identity.ir_json.clone(),
            iterations: from_iteration,
            seed: route.identity.seed,
            semantics: route.identity.semantics,
        };
        let target_key = GenerationCacheKey {
            source: Arc::clone(&route.identity.source),
            ir_json: route.identity.ir_json.clone(),
            iterations: to_iteration,
            seed: route.identity.seed,
            semantics: route.identity.semantics,
        };
        let transition_source = self.generation_cache.get(&source_key);
        let cached_generation = self.generation_cache.get(&target_key);
        let animate = !source_uses_filled_turtle_polygons(&route.identity.source);
        let job = RenderJob::default();
        let request = AppVisualizeRequest {
            request_id: route.request_id,
            view_epoch: self.system_view_epoch,
            key: target_key,
            angle: route.identity.angle,
            visualizer: route.identity.visualizer,
            turtle_config: route.identity.turtle_config,
            orientation_anchor: route.identity.orientation_anchor,
            seed: route.identity.seed,
            cached_generation,
            transition: animate.then_some(IterationTransitionRequest {
                from_iteration,
                to_iteration,
            }),
            transition_source: animate.then_some(transition_source).flatten(),
            job: job.clone(),
        };

        self.queue_render_request(request, job)
    }

    fn current_render_identity(&self) -> RenderIdentity {
        RenderIdentity {
            source: Arc::from(self.source_editor.text()),
            ir_json: self.active_ir_json.clone(),
            angle: self.angle,
            visualizer: self.visualizer,
            turtle_config: self.turtle_config.clone(),
            orientation_anchor: self.orientation_anchor,
            seed: self.effective_seed,
            semantics: self.derivation_semantics,
        }
    }

    fn queue_render_request(
        &mut self,
        request: AppVisualizeRequest,
        job: RenderJob,
    ) -> Task<Message> {
        self.cancel_export_for_replacement();
        if let Some(lines) = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.line_scene.as_ref())
        {
            lines.cancel_refinement();
        }
        if let Some(spatial) = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.spatial_scene.as_ref())
        {
            spatial.cancel_refinement();
        }
        self.render.active = Some(job);
        self.render.failure = None;
        self.render_coordinator.submit(request)
    }

    fn set_iterations(&mut self, iterations: usize) {
        self.iterations = iterations;
        self.iterations_input = iterations.to_string();
        self.iterations_notice = None;
    }

    fn cancel_obsolete_work(&mut self) {
        self.pending_autorotate_request = None;
        self.cancel_export_for_replacement();
        self.iteration_route = None;
        self.iteration_animation = None;
        self.request_id = self.request_id.wrapping_add(1);
        self.render.active = None;
        self.render_coordinator.cancel_current();
        if let Some(lines) = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.line_scene.as_ref())
        {
            lines.cancel_refinement();
        }
        if let Some(spatial) = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.spatial_scene.as_ref())
        {
            spatial.cancel_refinement();
        }
        self.render.failure = None;
    }

    fn cancel_export_for_replacement(&mut self) {
        if let Some(export) = self.export_job.take() {
            export.cancelled.store(true, Ordering::Release);
            self.export_notice = None;
        }
    }

    fn regenerate_seed(&mut self) {
        match generate_seed() {
            Ok(seed) => {
                self.effective_seed = seed;
                self.seed_input.clear();
                self.seed_placeholder = seed.to_string();
                self.seed_notice = None;
            }
            Err(error) => {
                self.seed_notice = Some(format!(
                    "Could not generate a random seed; using {}: {error}",
                    self.effective_seed
                ));
            }
        }
    }

    fn ir_panel(&self) -> Element<'_, Message> {
        let Some(tooling) = self.current_ir_tooling() else {
            return column![
                button_widget("Import JSON from clipboard").on_press(Message::ImportIr),
                text("Compile a valid system or import IR to inspect it.").size(13),
            ]
            .spacing(8)
            .width(Length::Fill)
            .height(Length::Fill)
            .into();
        };
        let status = if self.imported_ir.is_some() {
            "Validated imported IR"
        } else if self.render.displayed.as_ref().is_some_and(|result| {
            result.identity.source.as_ref() == self.source_editor.text().as_str()
                && result.identity.ir_json == self.active_ir_json
        }) {
            "Validated IR for the current input"
        } else {
            "IR for the last completed input"
        };
        let export: Element<'_, Message> = if tooling.json.is_some() {
            button_widget("Export JSON")
                .on_press(Message::ExportIr)
                .into()
        } else {
            button_widget("Export unavailable").into()
        };
        let import_actions: Element<'_, Message> = if self.imported_ir.is_some() {
            row![
                button_widget("Use imported IR").on_press(Message::UseImportedIr),
                button_widget("Close import").on_press(Message::ClearImportedIr),
            ]
            .spacing(8)
            .into()
        } else {
            button_widget("Import JSON from clipboard")
                .on_press(Message::ImportIr)
                .into()
        };
        column![
            text(status).size(12),
            row![
                button_widget("Copy disassembly").on_press(Message::CopyIr),
                export,
            ]
            .spacing(8),
            import_actions,
            if let Some(notice) = self.ir_notice.as_deref() {
                text(notice).size(12)
            } else {
                text("").size(1)
            },
            scrollable(
                container_widget(text(tooling.disassembly.as_ref()).size(12))
                    .padding(8)
                    .width(Length::Fill)
            )
            .height(Length::Fill)
            .width(Length::Fill),
        ]
        .spacing(8)
        .height(Length::Fill)
        .width(Length::Fill)
        .into()
    }

    fn current_ir_tooling(&self) -> Option<&IrToolingSnapshot> {
        self.imported_ir.as_ref().map_or_else(
            || {
                self.render
                    .displayed
                    .as_ref()
                    .map(|result| &result.ir_tooling)
            },
            |imported| Some(&imported.tooling),
        )
    }

    fn render_scene_and_status(&self) -> (Option<&Scene2d>, String) {
        self.render_scene_and_status_at(RenderInstant::now())
    }

    fn render_scene_and_status_at(&self, now: RenderInstant) -> (Option<&Scene2d>, String) {
        let scene = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.scene.as_deref());
        if let Some(animation) = &self.iteration_animation {
            return (
                scene,
                format!(
                    "Animating iteration {} → {}",
                    animation.prepared.from_iteration, animation.prepared.to_iteration,
                ),
            );
        }
        if let Some(job) = &self.render.active {
            if !should_show_transient_status(now.duration_since(job.started_at)) {
                return (scene, self.resting_status());
            }
            let progress = job.progress.lock().ok().map(|value| *value);
            let status = progress.map_or_else(
                || String::from("Preparing…"),
                |progress| {
                    if progress.phase == RenderPhase::Visualizing
                        && let Some(visualization) = progress.visualization
                    {
                        return format!(
                            "Visualizing · {} items · {} lines",
                            visualization.items_processed, visualization.lines_emitted,
                        );
                    }
                    (progress.phase == RenderPhase::Deriving)
                        .then_some(progress.derivation)
                        .flatten()
                        .map_or_else(
                            || format!("{}…", progress.phase.label()),
                            |derivation| {
                                format!(
                                    "Deriving {}/{} · {} modules · {} items",
                                    derivation.completed_iterations,
                                    derivation.total_iterations,
                                    derivation.modules,
                                    derivation.items,
                                )
                            },
                        )
                },
            );
            return (scene, status);
        }
        if self.render.failure.is_some() {
            return (
                scene,
                String::from("Error — showing the last completed result"),
            );
        }
        match &self.render.displayed {
            Some(result) => {
                if let Some(lines) = &result.line_scene {
                    let (completed, total) = lines.refinement_progress();
                    if lines.is_refining()
                        && should_show_transient_status(
                            now.duration_since(result.refinement_started_at),
                        )
                    {
                        return (
                            scene,
                            format!("Showing preview · refining display {completed}/{total} lines"),
                        );
                    }
                    if lines.refinement_was_cancelled() && completed < total {
                        return (
                            scene,
                            format!(
                                "Showing preview · display refinement stopped at {completed}/{total}"
                            ),
                        );
                    }
                }
                if let Some(spatial) = &result.spatial_scene {
                    if let Some(error) = spatial.gpu_refinement_failure() {
                        return (
                            scene,
                            format!(
                                "Showing software 3D preview · GPU display unavailable: {error}"
                            ),
                        );
                    }
                    let (completed, total) = spatial.refinement_progress();
                    if spatial.is_refining()
                        && !self.autorotate_3d
                        && !self.view_gesture_active
                        && should_show_transient_status(
                            now.duration_since(result.refinement_started_at),
                        )
                    {
                        return (
                            scene,
                            format!(
                                "Showing 3D preview · refining display {completed}/{total} primitives"
                            ),
                        );
                    }
                    if spatial.refinement_was_cancelled() && completed < total {
                        return (
                            scene,
                            format!(
                                "Showing 3D preview · display refinement stopped at {completed}/{total} primitives"
                            ),
                        );
                    }
                }
                (scene, format!("{} elements", result.element_count))
            }
            None => (None, String::from("Ready")),
        }
    }

    fn resting_status(&self) -> String {
        self.render.displayed.as_ref().map_or_else(
            || String::from("Ready"),
            |result| format!("{} elements", result.element_count),
        )
    }

    fn canvas(&self, layout: LayoutMode) -> Element<'_, Message> {
        responsive(move |size| self.canvas_at_size(layout, size)).into()
    }

    fn canvas_at_size(&self, layout: LayoutMode, size: Size) -> Element<'_, Message> {
        let viewport = ViewportSize::new(f64::from(size.width), f64::from(size.height));
        let (scene, _) = self.render_scene_and_status();
        let transition = self.iteration_animation.as_ref();
        let line_scene = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.line_scene.as_ref());
        let spatial_scene = self
            .render
            .displayed
            .as_ref()
            .and_then(|result| result.spatial_scene.as_ref());
        let line_bounds = self
            .render
            .displayed
            .as_ref()
            .map_or_else(RenderBounds::default, |result| result.bounds);
        let planar_navigation = transition.is_some()
            || line_scene.is_some()
            || scene.is_some_and(scene_supports_planar_navigation);
        let fallback: Element<'_, Message> = if let Some(spatial_scene) = spatial_scene {
            Canvas::new(SpatialCanvas {
                scene: spatial_scene,
                orbit: self.orbit_3d,
                dark: self.effective_dark(),
                line_width_scale: self.line_width_scale(),
                cache: &self.canvas_cache,
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        } else {
            Canvas::new(LsystemCanvas {
                scene: transition.is_none().then_some(scene).flatten(),
                transition: transition
                    .map(|animation| (animation.prepared.scene.as_ref(), animation.progress)),
                line_bounds,
                exact_total_line_length: line_scene.and_then(|scene| scene.total_line_length()),
                line_scene: line_scene.map(Arc::as_ref),
                camera: self.live_camera,
                dark: self.effective_dark(),
                line_width_scale: self.line_width_scale(),
                cache: &self.canvas_cache,
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        };

        let accelerated: Element<'_, Message> = if let Some(spatial_scene) = spatial_scene {
            shader::Shader::new(SpatialProgram::new(
                Arc::clone(spatial_scene),
                self.orbit_3d,
                self.effective_dark(),
                self.view_gesture_active || self.autorotate_3d,
                self.line_width_scale(),
            ))
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        } else if let Some(animation) = transition {
            shader::Shader::new(TransitionProgram::new(
                Arc::clone(&animation.prepared.scene),
                animation.progress,
                self.effective_dark(),
                self.live_camera,
                self.line_width_scale(),
            ))
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        } else {
            line_scene.map_or_else(
                || Space::new().width(Length::Fill).height(Length::Fill).into(),
                |line_scene| {
                    shader::Shader::new(LineProgram::new(
                        Arc::clone(line_scene),
                        self.effective_dark(),
                        self.live_camera,
                        self.settled_camera,
                        self.view_is_interacting(),
                        self.raster_epoch,
                        self.line_width_scale(),
                    ))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .into()
                },
            )
        };

        let gestures: Element<'_, Message> = if spatial_scene.is_some() {
            Canvas::new(SpatialViewportController {
                orbit: self.orbit_3d,
                epoch: self.gesture_epoch,
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        } else {
            Canvas::new(ViewportController {
                camera: self.live_camera,
                line_bounds,
                mobile: layout == LayoutMode::Mobile,
                enabled: planar_navigation,
                epoch: self.gesture_epoch,
            })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        };

        let controls: Element<'_, Message> =
            if layout == LayoutMode::Desktop && planar_navigation && spatial_scene.is_none() {
                container_widget(
                    row![
                        button_widget("−").on_press(Message::Viewport(ViewportMessage::ZoomBy {
                            factor: 1.0 / VIEW_BUTTON_ZOOM_FACTOR,
                            viewport,
                        })),
                        button_widget("Fit").on_press(Message::Viewport(ViewportMessage::Fit)),
                        button_widget("+").on_press(Message::Viewport(ViewportMessage::ZoomBy {
                            factor: VIEW_BUTTON_ZOOM_FACTOR,
                            viewport,
                        })),
                        text(format!("{:.0}%", self.live_camera.zoom * 100.0)).size(12),
                    ]
                    .spacing(6)
                    .align_y(Alignment::Center),
                )
                .padding(8)
                .align_right(Length::Fill)
                .align_top(Length::Fill)
                .into()
            } else {
                Space::new().width(Length::Fill).height(Length::Fill).into()
            };

        // Iced's fallback renderer ignores custom shader primitives. Keeping a
        // bounded Canvas underneath makes that path useful; on WGPU, the opaque
        // shader result replaces it without duplicate sampled lines. The input
        // layer is transparent and shared by both renderers.
        // Stack's first child is not enclosed in a clipping layer. A spacer
        // makes every drawing layer honor the canvas bounds, including inside
        // scrolling layouts.
        stack![
            Space::new().width(Length::Fill).height(Length::Fill),
            fallback,
            accelerated,
            gestures,
            controls,
        ]
        .clip(true)
        .into()
    }

    fn matching_preset_indices(&self) -> Vec<usize> {
        let terms = normalized_search_terms(&self.preset_search);
        self.presets
            .iter()
            .enumerate()
            .filter(|(_, preset)| preset.matches_search_terms(&terms))
            .map(|(index, _)| index)
            .collect()
    }
}

fn source_uses_filled_turtle_polygons(source: &str) -> bool {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .flat_map(|line| {
            line.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        })
        .any(|word| matches!(word, "PolygonBegin" | "PolygonEnd" | "Vertex"))
}

fn normalized_search_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|term| term.to_lowercase())
        .collect()
}

fn app_subscription(state: &BrakenGui) -> iced::Subscription<Message> {
    let theme_changes = iced::system::theme_changes().map(Message::SystemThemeChanged);
    let viewport_changes = iced::event::listen_with(|event, _status, _window| match event {
        iced::Event::Window(
            iced::window::Event::Opened { size, .. } | iced::window::Event::Resized(size),
        ) => Some(Message::WindowResized(size)),
        iced::Event::Window(iced::window::Event::Rescaled(_)) => Some(Message::ViewportInvalidated),
        iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
            key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape),
            ..
        }) => Some(Message::CloseOverlay),
        _ => None,
    });
    let refining = state.render.displayed.as_ref().is_some_and(|result| {
        result
            .line_scene
            .as_ref()
            .is_some_and(|scene| scene.is_refining())
            || result
                .spatial_scene
                .as_ref()
                .is_some_and(|scene| scene.is_refining())
    });
    let activity = if state.canvas_is_visible()
        && (state.iteration_animation.is_some()
            || refining
            || state.wheel_changed_at.is_some()
            || state.view_gesture_active
            || state.autorotate_3d)
    {
        // Animation and progressive GPU upload need one bounded slice per
        // display frame.
        iced::window::frames().map(|_| Message::Frame)
    } else if state.source_changed_at.is_some() || state.render.active.is_some() {
        // Calculation progress and the delayed cancel affordance do not need
        // a full-rate rebuild of the (large) preset list.
        iced::time::every(BACKGROUND_STATUS_INTERVAL).map(|_| Message::Frame)
    } else {
        iced::Subscription::none()
    };

    iced::Subscription::batch([theme_changes, viewport_changes, activity])
}

fn stepped_fractional_value(current: f32, delta: mouse::ScrollDelta, min: f32, max: f32) -> f32 {
    let amount = scroll_amount(delta);
    if amount == 0.0 {
        return current.clamp(min, max);
    }

    let direction = if amount.is_sign_positive() { 1.0 } else { -1.0 };
    (current + direction).clamp(min, max)
}

fn adjacent_integer_angle(current: f32, direction: f32) -> f32 {
    let nearest = current.round();
    let normalized = if (current - nearest).abs() < 1.0e-4 {
        nearest
    } else {
        current
    };
    let value = if direction.is_sign_positive() {
        normalized.floor() + 1.0
    } else {
        normalized.ceil() - 1.0
    };
    value.clamp(MIN_ANGLE, MAX_ANGLE)
}

fn formatted_angle(angle: f32) -> String {
    let rounded = angle.round();
    if (angle - rounded).abs() < 0.05 {
        format!("{rounded:.0}")
    } else {
        format!("{angle:.1}")
    }
}

fn scroll_amount(delta: mouse::ScrollDelta) -> f32 {
    match delta {
        mouse::ScrollDelta::Lines { y, .. } => y,
        mouse::ScrollDelta::Pixels { y, .. } => y,
    }
}

fn scene_supports_planar_navigation(scene: &Scene2d) -> bool {
    let has_text = scene
        .primitives
        .iter()
        .any(|primitive| matches!(primitive, Primitive2d::Text(_)));
    !has_text
        && scene
            .primitives
            .iter()
            .any(|primitive| matches!(primitive, Primitive2d::Line(_) | Primitive2d::Polygon(_)))
}

fn preset_button<'a>(
    index: usize,
    preset: &'a presets::Preset,
    preview_handle: &'a svg::Handle,
    selected: bool,
) -> Element<'a, Message> {
    let is_3d = preset.visualizer == VisualizerKind::Turtle3d;
    let preview = container_widget(
        svg(preview_handle.clone())
            .width(Length::Fixed(64.0))
            .height(Length::Fixed(46.0)),
    )
    .width(Length::Fixed(64.0))
    .height(Length::Fixed(46.0))
    .style(move |theme| preview_style(theme, selected, is_3d));

    let mut metadata: Row<'_, Message> = Row::new().spacing(10).align_y(Alignment::Center);
    if let Some(label) = preset.dimension_badge_label() {
        metadata = metadata.push(
            container_widget(text(label).size(10))
                .padding([2, 6])
                .style(move |theme| dimension_badge_style(theme, is_3d)),
        );
    }
    let content = row![
        preview,
        column![
            text(&preset.name).size(15),
            text(&preset.summary).size(12),
            metadata
        ]
        .spacing(4)
        .width(Length::Fill),
    ]
    .align_y(Alignment::Center)
    .spacing(12);

    button_widget(content)
        .width(Length::Fill)
        .padding(10)
        .style(move |theme, status| preset_button_style(theme, status, selected))
        .on_press(Message::PresetSelected(PresetChoice(index)))
        .into()
}

fn theme_is_dark(theme: &Theme) -> bool {
    theme.extended_palette().is_dark
}

fn dimension_badge_style(theme: &Theme, is_3d: bool) -> container::Style {
    let dark = theme_is_dark(theme);
    let (background, text, border) = match (dark, is_3d) {
        (true, true) => (
            Color::from_rgb8(52, 69, 95),
            Color::from_rgb8(225, 236, 255),
            Color::from_rgb8(95, 165, 255),
        ),
        (false, true) => (
            Color::from_rgb8(220, 232, 249),
            Color::from_rgb8(25, 68, 112),
            Color::from_rgb8(74, 126, 196),
        ),
        (true, false) => (
            Color::from_rgb8(34, 45, 66),
            Color::from_rgb8(174, 198, 232),
            Color::from_rgb8(66, 82, 108),
        ),
        (false, false) => (
            Color::from_rgb8(239, 243, 248),
            Color::from_rgb8(78, 99, 132),
            Color::from_rgb8(177, 195, 217),
        ),
    };
    container::Style::default()
        .background(background)
        .color(text)
        .border(Border::default().rounded(5).width(1).color(border))
}

fn app_background(theme: &Theme) -> container::Style {
    if theme_is_dark(theme) {
        container::Style::default()
            .background(DARK_CANVAS_BACKGROUND)
            .color(Color::from_rgb8(232, 238, 247))
    } else {
        container::Style::default()
            .background(CANVAS_BACKGROUND)
            .color(Color::from_rgb8(24, 31, 46))
    }
}

fn panel_style(theme: &Theme) -> container::Style {
    let (background, border, text) = if theme_is_dark(theme) {
        (
            Color::from_rgb8(23, 30, 46),
            Color::from_rgb8(53, 65, 88),
            Color::from_rgb8(235, 240, 248),
        )
    } else {
        (
            Color::from_rgb8(250, 251, 253),
            Color::from_rgb8(205, 213, 224),
            Color::from_rgb8(24, 31, 46),
        )
    };
    container::Style::default()
        .background(background)
        .border(Border::default().rounded(8).width(1).color(border))
        .color(text)
}

fn header_icon_style(theme: &Theme, status: svg::Status) -> svg::Style {
    let color = match (theme_is_dark(theme), status) {
        (true, svg::Status::Hovered) => Color::from_rgb8(255, 255, 255),
        (true, svg::Status::Idle) => Color::from_rgb8(210, 230, 255),
        (false, svg::Status::Hovered) => Color::from_rgb8(18, 66, 112),
        (false, svg::Status::Idle) => Color::from_rgb8(31, 78, 121),
    };
    svg::Style { color: Some(color) }
}

fn disabled_icon_style(theme: &Theme, _status: svg::Status) -> svg::Style {
    let color = if theme_is_dark(theme) {
        Color::from_rgb8(112, 124, 143)
    } else {
        Color::from_rgb8(145, 154, 168)
    };
    svg::Style { color: Some(color) }
}

fn header_icon_button_style(theme: &Theme, status: button::Status) -> button::Style {
    let dark = theme_is_dark(theme);
    let background = match (dark, status) {
        (true, button::Status::Hovered) => Color::from_rgb8(52, 72, 101),
        (true, button::Status::Pressed) => Color::from_rgb8(30, 43, 64),
        (true, button::Status::Disabled) => Color::from_rgb8(31, 39, 55),
        (true, _) => Color::from_rgb8(38, 51, 74),
        (false, button::Status::Hovered) => Color::from_rgb8(216, 230, 247),
        (false, button::Status::Pressed) => Color::from_rgb8(197, 218, 241),
        (false, button::Status::Disabled) => Color::from_rgb8(239, 242, 246),
        (false, _) => Color::from_rgb8(232, 239, 248),
    };
    let border = if status == button::Status::Disabled && dark {
        Color::from_rgb8(58, 68, 84)
    } else if status == button::Status::Disabled {
        Color::from_rgb8(210, 216, 225)
    } else if dark {
        Color::from_rgb8(78, 99, 132)
    } else {
        Color::from_rgb8(177, 195, 217)
    };

    button::Style {
        background: Some(Background::Color(background)),
        text_color: if dark {
            Color::from_rgb8(210, 230, 255)
        } else {
            Color::from_rgb8(31, 78, 121)
        },
        border: Border::default().rounded(7).width(1).color(border),
        ..button::Style::default()
    }
}

fn preview_style(theme: &Theme, selected: bool, is_3d: bool) -> container::Style {
    let dark = theme_is_dark(theme);
    let flat_background = if selected && dark {
        Color::from_rgb8(64, 82, 116)
    } else if selected {
        Color::from_rgb8(218, 229, 246)
    } else if dark {
        Color::from_rgb8(34, 45, 66)
    } else {
        Color::from_rgb8(239, 243, 248)
    };
    let background = if is_3d {
        let (start, middle, end) = match (dark, selected) {
            (true, true) => (
                Color::from_rgb8(64, 82, 116),
                Color::from_rgb8(52, 72, 101),
                Color::from_rgb8(52, 69, 95),
            ),
            (true, false) => (
                Color::from_rgb8(52, 72, 101),
                Color::from_rgb8(34, 45, 66),
                Color::from_rgb8(52, 69, 95),
            ),
            (false, true) => (
                Color::WHITE,
                Color::from_rgb8(218, 229, 246),
                Color::from_rgb8(220, 232, 249),
            ),
            (false, false) => (
                Color::WHITE,
                Color::from_rgb8(239, 243, 248),
                Color::from_rgb8(220, 232, 249),
            ),
        };
        gradient::Linear::new(Degrees(145.0))
            .add_stop(0.0, start)
            .add_stop(0.55, middle)
            .add_stop(1.0, end)
            .into()
    } else {
        Background::Color(flat_background)
    };
    let border_color = if selected {
        Color::from_rgb8(111, 181, 255)
    } else if is_3d && dark {
        Color::from_rgb8(78, 99, 132)
    } else if is_3d {
        Color::from_rgb8(117, 150, 194)
    } else {
        Color::from_rgb8(66, 82, 108)
    };
    let shadow = if is_3d {
        Shadow {
            color: if dark {
                Color::from_rgba8(0, 0, 0, 0.3)
            } else {
                Color::from_rgba8(31, 78, 121, 0.18)
            },
            offset: Vector::new(0.0, 2.0),
            blur_radius: 4.0,
        }
    } else {
        Shadow::default()
    };
    container::Style::default()
        .background(background)
        .border(Border::default().rounded(6).width(1).color(border_color))
        .shadow(shadow)
}

fn preset_button_style(theme: &Theme, status: button::Status, selected: bool) -> button::Style {
    let dark = theme_is_dark(theme);
    let background = match (dark, selected, status) {
        (true, true, _) => Color::from_rgb8(52, 69, 95),
        (true, false, button::Status::Hovered) => Color::from_rgb8(38, 49, 70),
        (true, false, _) => Color::from_rgb8(29, 37, 54),
        (false, true, _) => Color::from_rgb8(220, 232, 249),
        (false, false, button::Status::Hovered) => Color::from_rgb8(235, 240, 247),
        (false, false, _) => Color::from_rgb8(247, 249, 252),
    };

    button::Style {
        background: Some(Background::Color(background)),
        text_color: if dark {
            Color::from_rgb8(234, 240, 248)
        } else {
            Color::from_rgb8(28, 36, 51)
        },
        border: Border::default()
            .rounded(8)
            .width(if selected { 2 } else { 1 })
            .color(if selected && dark {
                Color::from_rgb8(95, 165, 255)
            } else if selected {
                Color::from_rgb8(74, 126, 196)
            } else if dark {
                Color::from_rgb8(54, 66, 88)
            } else {
                Color::from_rgb8(205, 213, 224)
            }),
        ..button::Style::default()
    }
}

fn status_style(theme: &Theme) -> container::Style {
    let (background, border) = if theme_is_dark(theme) {
        (Color::from_rgb8(29, 38, 57), Color::from_rgb8(57, 70, 94))
    } else {
        (
            Color::from_rgb8(235, 240, 247),
            Color::from_rgb8(202, 211, 224),
        )
    };
    container::Style::default()
        .background(background)
        .border(Border::default().rounded(6).width(1).color(border))
}

fn tooltip_style(theme: &Theme) -> container::Style {
    let (background, text) = if theme_is_dark(theme) {
        (
            Color::from_rgb8(238, 242, 248),
            Color::from_rgb8(20, 27, 40),
        )
    } else {
        (Color::from_rgb8(25, 32, 46), Color::WHITE)
    };
    container::Style::default()
        .background(background)
        .color(text)
        .border(Border::default().rounded(5))
}

#[derive(Debug, Clone)]
struct AppVisualizeRequest {
    request_id: u64,
    view_epoch: u64,
    key: GenerationCacheKey,
    angle: f32,
    visualizer: VisualizerKind,
    turtle_config: Turtle2dConfig,
    orientation_anchor: Option<OrientationAnchor>,
    seed: u64,
    cached_generation: Option<CachedGeneration>,
    transition: Option<IterationTransitionRequest>,
    transition_source: Option<CachedGeneration>,
    job: RenderJob,
}

#[derive(Debug, Clone, Copy)]
struct IterationTransitionRequest {
    from_iteration: usize,
    to_iteration: usize,
}

/// Runs the derive-then-visualize pipeline away from the UI thread/event loop.
/// Display preparation remains backend-neutral and the previous completed
/// result stays visible until this function returns a current successful result.
fn render_system(
    request: AppVisualizeRequest,
    turtle_streamer: &mut Turtle2dStreamer,
) -> RenderOutcome {
    if let Some(transition) = request.transition {
        return render_iteration_transition(request, transition, turtle_streamer);
    }
    let start = RenderInstant::now();
    if request.job.is_cancelled() {
        return RenderOutcome::Cancelled {
            request_id: request.request_id,
        };
    }

    let result = generation_for_request(&request).and_then(
        |(generation, cache_insert, info, ir_tooling)| {
            if request.job.is_cancelled() {
                return Err(String::from("calculation cancelled"));
            }
            request.job.set_phase(RenderPhase::Visualizing);
            visualize_generation(&generation, &request, info, turtle_streamer)
                .map(|display| (display, cache_insert, info.label(), ir_tooling))
        },
    );

    match result {
        Ok((display, cache_insert, derivation_backend, ir_tooling)) => {
            if request.job.is_cancelled() {
                return RenderOutcome::Cancelled {
                    request_id: request.request_id,
                };
            }
            request.job.set_phase(RenderPhase::Measuring);
            if request.job.is_cancelled() {
                return RenderOutcome::Cancelled {
                    request_id: request.request_id,
                };
            }
            RenderOutcome::Ready(RenderResult {
                request_id: request.request_id,
                view_epoch: request.view_epoch,
                scene: display.scene,
                line_scene: display.line_scene,
                spatial_scene: display.spatial_scene,
                element_count: display.element_count,
                refinement_started_at: RenderInstant::now(),
                elapsed_ms: start.elapsed().as_millis(),
                backend: format!(
                    "{} derive · {} visualize",
                    derivation_backend,
                    visualizer_backend_name(display.visualization_backend),
                ),
                bounds: display.bounds,
                cache_insert,
                iteration: request.key.iterations,
                identity: request_identity(&request),
                ir_tooling,
            })
        }
        Err(_) if request.job.is_cancelled() => RenderOutcome::Cancelled {
            request_id: request.request_id,
        },
        Err(message) => RenderOutcome::Failed {
            request_id: request.request_id,
            message,
        },
    }
}

fn render_iteration_transition(
    request: AppVisualizeRequest,
    transition: IterationTransitionRequest,
    turtle_streamer: &mut Turtle2dStreamer,
) -> RenderOutcome {
    let started = RenderInstant::now();
    let request_id = request.request_id;
    let result = (|| {
        if request.visualizer != VisualizerKind::Turtle2d
            || transition.from_iteration.abs_diff(transition.to_iteration) != 1
        {
            return Err(String::from(
                "iteration deformation requires adjacent Turtle 2D generations",
            ));
        }

        let mut source_request = request.clone();
        source_request.key.iterations = transition.from_iteration;
        source_request.cached_generation = request.transition_source.clone();
        source_request.transition = None;
        source_request.transition_source = None;
        let (source_generation, _, source_info, _) = generation_for_request(&source_request)?;
        let (target_generation, cache_insert, target_info, ir_tooling) =
            generation_for_request(&request)?;
        if request.job.is_cancelled() {
            return Err(String::from("calculation cancelled"));
        }

        let (lower_generation, higher_generation, lower_iteration, higher_info, reverse) =
            if transition.from_iteration < transition.to_iteration {
                (
                    source_generation.as_ref(),
                    target_generation.as_ref(),
                    transition.from_iteration,
                    target_info,
                    false,
                )
            } else {
                (
                    target_generation.as_ref(),
                    source_generation.as_ref(),
                    transition.to_iteration,
                    source_info,
                    true,
                )
            };
        let grammar = compiled_grammar_for_key(&request.key)?;
        let lineage = if higher_info.backend == "WGPU" {
            calculate_rewrite_lineage_with_control(
                CalculationRequest {
                    grammar: grammar.clone(),
                    iterations: lower_iteration.saturating_add(1),
                    backend: BackendChoice::Wgpu,
                    seed: request.seed,
                    semantics: request.key.semantics,
                    limits: CalculationLimits::default(),
                },
                &request.job.calculation_cancel,
            )
            .map_err(|error| error.to_string())?
            .0
        } else {
            let trace_program = CpuBackend::new()
                .float_width(request.key.semantics.float_width)
                .ambiguous_rules(request.key.semantics.ambiguous_rules)
                .limits(CalculationLimits::default().into())
                .compile_ir(grammar.derivation_ir())
                .map_err(|error| error.to_string())?;
            let (traced_higher, _, lineage) = trace_program
                .trace_rewrite_with_control(
                    lower_generation,
                    u64::try_from(lower_iteration).unwrap_or(u64::MAX),
                    request.seed,
                    || request.job.is_cancelled(),
                    |_| {},
                )
                .map_err(|error| error.to_string())?;
            if &traced_higher != higher_generation {
                return Err(String::from(
                    "selected derivation backend disagreed with CPU rewrite lineage",
                ));
            }
            lineage
        };

        request.job.set_phase(RenderPhase::Visualizing);
        let source_display = visualize_indexed_turtle(
            &source_generation,
            &request,
            source_info,
            !reverse,
            turtle_streamer,
        )?;
        let target_display = visualize_indexed_turtle(
            &target_generation,
            &request,
            target_info,
            reverse,
            turtle_streamer,
        )?;
        let morphs = if reverse {
            build_line_transition(
                &target_display.indexed_lines,
                &target_display.module_positions,
                &source_display.indexed_lines,
                &lineage,
                true,
            )
        } else {
            build_line_transition(
                &source_display.indexed_lines,
                &source_display.module_positions,
                &target_display.indexed_lines,
                &lineage,
                false,
            )
        }
        .map_err(|error| error.to_string())?;
        if request.job.is_cancelled() {
            return Err(String::from("visualization cancelled"));
        }
        let transition_scene = TransitionScene::new(
            morphs,
            source_display.display.bounds,
            target_display.display.bounds,
            (
                source_display.display.element_count,
                source_display
                    .display
                    .line_scene
                    .as_ref()
                    .and_then(|scene| scene.total_line_length()),
            ),
            (
                target_display.display.element_count,
                target_display
                    .display
                    .line_scene
                    .as_ref()
                    .and_then(|scene| scene.total_line_length()),
            ),
            request.turtle_config.background,
        );
        let target_backend = format!(
            "{} derive · {} visualize",
            target_info.label(),
            visualizer_backend_name(target_display.display.visualization_backend),
        );
        let target = RenderResult {
            request_id,
            view_epoch: request.view_epoch,
            scene: target_display.display.scene,
            line_scene: target_display.display.line_scene,
            spatial_scene: target_display.display.spatial_scene,
            element_count: target_display.display.element_count,
            refinement_started_at: RenderInstant::now(),
            elapsed_ms: started.elapsed().as_millis(),
            backend: target_backend,
            bounds: target_display.display.bounds,
            cache_insert,
            iteration: transition.to_iteration,
            identity: request_identity(&request),
            ir_tooling,
        };
        Ok(PreparedIterationTransition {
            request_id,
            from_iteration: transition.from_iteration,
            to_iteration: transition.to_iteration,
            scene: transition_scene,
            target,
        })
    })();

    match result {
        Ok(transition) => RenderOutcome::TransitionReady(transition),
        Err(_) if request.job.is_cancelled() => RenderOutcome::Cancelled { request_id },
        Err(message) => RenderOutcome::Failed {
            request_id,
            message,
        },
    }
}

struct IndexedPreparedDisplay {
    display: PreparedDisplay,
    indexed_lines: Vec<IndexedLine2d>,
    module_positions: Vec<IndexedModulePosition2d>,
}

fn visualize_indexed_turtle(
    generation: &Generation,
    request: &AppVisualizeRequest,
    _info: DerivationInfo,
    capture_module_positions: bool,
    turtle_streamer: &mut Turtle2dStreamer,
) -> Result<IndexedPreparedDisplay, String> {
    let mut config = request.turtle_config.clone();
    config.turn_angle = (request.angle as f64).to_radians();
    let orientation_reference = config.initial_angle;
    let mut builder = LineSceneBuilder::default();
    builder.set_background(config.background);
    let mut indexed_lines = Vec::new();
    let mut module_positions = Vec::new();
    let mut polygons = Vec::new();
    let is_cancelled = || request.job.is_cancelled();
    let summary = turtle_streamer
        .stream(
            VisualizerBackend::Auto,
            Turtle2dStreamRequest {
                generation,
                config,
                batch_size: 16 * 1024,
                is_cancelled: &is_cancelled,
            },
            |batch| {
                request.job.set_visualization_progress(batch.progress);
                if batch.lines.len() != batch.module_indices.len() {
                    return Err(VisualizeError::InvalidConfiguration(String::from(
                        "turtle line/module-index batch length mismatch",
                    )));
                }
                indexed_lines.try_reserve(batch.lines.len()).map_err(|_| {
                    VisualizeError::ResourceExhausted {
                        resource: "GUI indexed turtle scene",
                        requested: Some(indexed_lines.len().saturating_add(batch.lines.len())),
                    }
                })?;
                indexed_lines.extend(
                    batch
                        .lines
                        .iter()
                        .copied()
                        .zip(batch.module_indices)
                        .map(|(line, module_index)| IndexedLine2d { module_index, line }),
                );
                if capture_module_positions {
                    module_positions
                        .try_reserve(batch.module_positions.len())
                        .map_err(|_| VisualizeError::ResourceExhausted {
                            resource: "GUI indexed turtle module positions",
                            requested: Some(
                                module_positions
                                    .len()
                                    .saturating_add(batch.module_positions.len()),
                            ),
                        })?;
                    module_positions.extend(batch.module_positions);
                }
                polygons.try_reserve(batch.polygons.len()).map_err(|_| {
                    VisualizeError::ResourceExhausted {
                        resource: "GUI turtle polygons",
                        requested: Some(polygons.len().saturating_add(batch.polygons.len())),
                    }
                })?;
                polygons.extend(batch.polygons);
                builder
                    .extend(batch.lines)
                    .map_err(|error| VisualizeError::ResourceExhausted {
                        resource: "GUI GPU line scene",
                        requested: match error {
                            line_shader::LineSceneError::ResourceExhausted { requested_lines } => {
                                Some(requested_lines)
                            }
                            line_shader::LineSceneError::Cancelled => None,
                            line_shader::LineSceneError::InvalidTransfer { .. } => None,
                        },
                    })
            },
        )
        .map_err(|error| error.to_string())?;
    if !polygons.is_empty() {
        return Err(String::from(
            "iteration deformation is unavailable for turtle scenes containing filled polygons",
        ));
    }
    let bounds = summary
        .bounds
        .map_or_else(RenderBounds::default, |bounds| RenderBounds {
            min_x: bounds.min.0 as f32,
            max_x: bounds.max.0 as f32,
            min_y: bounds.min.1 as f32,
            max_y: bounds.max.1 as f32,
        });
    let (line_scene, bounds, orientation) =
        builder.finish_oriented(bounds, request.orientation_anchor, orientation_reference);
    if !orientation.is_identity() {
        for line in &mut indexed_lines {
            line.line.line.0 = orientation.apply(line.line.line.0);
            line.line.line.1 = orientation.apply(line.line.line.1);
        }
        for position in &mut module_positions {
            position.position = orientation.apply(position.position);
        }
    }
    let scene = line_scene
        .fallback_preview(SOFTWARE_FALLBACK_PREVIEW_LINES)
        .map(Arc::new)
        .map_err(|error| error.to_string())?;
    Ok(IndexedPreparedDisplay {
        display: PreparedDisplay {
            scene: Some(scene),
            line_scene: Some(line_scene),
            spatial_scene: None,
            element_count: summary.progress.lines_emitted,
            bounds,
            visualization_backend: summary.backend_used,
        },
        indexed_lines,
        module_positions,
    })
}

fn request_identity(request: &AppVisualizeRequest) -> RenderIdentity {
    RenderIdentity {
        source: Arc::clone(&request.key.source),
        ir_json: request.key.ir_json.clone(),
        angle: request.angle,
        visualizer: request.visualizer,
        turtle_config: request.turtle_config.clone(),
        orientation_anchor: request.orientation_anchor,
        seed: request.seed,
        semantics: request.key.semantics,
    }
}

/// A unit of latest-wins work. The shared queue owns only the newest waiting
/// item; the platform transport owns the active item after `Run` is sent.
trait WorkerWork {
    fn request_id(&self) -> u64;
    fn cancel(&self);
}

/// Commands emitted by the platform-independent latest-wins policy. Every
/// platform receives the same `Run`/`Cancel`/`Shutdown` vocabulary even though
/// native cancellation flips thread-safe tokens and browser cancellation is a
/// structured-clone message.
enum WorkerCommand<T> {
    Run(T),
    Cancel { request_id: u64 },
    Shutdown,
}

/// Platform-independent latest-wins state. One request may be active and one
/// newer request may wait; further submissions replace the waiting request.
/// Results remain transport-owned and are accepted by the GUI only by ID.
struct WorkerQueue<T> {
    active: Option<u64>,
    cancelling: bool,
    pending: Option<T>,
}

impl<T> Default for WorkerQueue<T> {
    fn default() -> Self {
        Self {
            active: None,
            cancelling: false,
            pending: None,
        }
    }
}

impl<T: WorkerWork> WorkerQueue<T> {
    fn submit(&mut self, item: T) -> Option<WorkerCommand<T>> {
        if self.active.is_none() {
            self.active = Some(item.request_id());
            return Some(WorkerCommand::Run(item));
        }

        if let Some(replaced) = self.pending.replace(item) {
            replaced.cancel();
        }
        if self.cancelling {
            None
        } else {
            self.cancelling = true;
            Some(WorkerCommand::Cancel {
                request_id: self.active.expect("an active request was checked above"),
            })
        }
    }

    fn finish(&mut self, request_id: u64) -> Option<WorkerCommand<T>> {
        if self.active != Some(request_id) {
            return None;
        }
        self.active = None;
        self.cancelling = false;
        self.pending.take().map(|next| {
            self.active = Some(next.request_id());
            WorkerCommand::Run(next)
        })
    }

    fn cancel(&mut self) -> Option<WorkerCommand<T>> {
        if let Some(pending) = self.pending.take() {
            pending.cancel();
        }
        let request_id = self.active?;
        if self.cancelling {
            None
        } else {
            self.cancelling = true;
            Some(WorkerCommand::Cancel { request_id })
        }
    }

    fn shutdown(&mut self) -> WorkerCommand<T> {
        if let Some(pending) = self.pending.take() {
            pending.cancel();
        }
        self.active = None;
        self.cancelling = false;
        WorkerCommand::Shutdown
    }
}

/// GUI-facing worker. Latest-wins coordination is shared; only command
/// delivery and cooperative-cancellation signaling differ by platform.
struct Worker {
    #[cfg(not(target_arch = "wasm32"))]
    platform: NativeWorkerThread,
    #[cfg(target_arch = "wasm32")]
    platform: BrowserWorker,
}

impl Worker {
    fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            platform: NativeWorkerThread::new(),
            #[cfg(target_arch = "wasm32")]
            platform: BrowserWorker::new(),
        }
    }

    fn submit(&self, request: AppVisualizeRequest) -> Task<Message> {
        self.platform.submit(request)
    }

    fn cancel_current(&self) {
        self.platform.cancel_current();
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeWorkerThread {
    shared: Arc<NativeWorkerShared>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeWorkerShared {
    state: Mutex<NativeWorkerState>,
    ready: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct NativeWorkerState {
    queue: WorkerQueue<NativeWorkItem>,
    ready: Option<NativeWorkItem>,
    active: Option<(u64, RenderJob)>,
    stopped: bool,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeWorkItem {
    request: AppVisualizeRequest,
    outcome: iced::futures::channel::oneshot::Sender<RenderOutcome>,
}

#[cfg(not(target_arch = "wasm32"))]
impl WorkerWork for NativeWorkItem {
    fn request_id(&self) -> u64 {
        self.request.request_id
    }

    fn cancel(&self) {
        self.request.job.cancel();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_native_command(state: &mut NativeWorkerState, command: WorkerCommand<NativeWorkItem>) {
    match command {
        WorkerCommand::Run(item) => state.ready = Some(item),
        WorkerCommand::Cancel { request_id } => {
            if let Some(item) = state
                .ready
                .as_ref()
                .filter(|item| item.request_id() == request_id)
            {
                item.cancel();
            }
            if let Some((_, job)) = state
                .active
                .as_ref()
                .filter(|(active_id, _)| *active_id == request_id)
            {
                // The native listener translates the cancellation command to
                // the atomics observed by CPU, WGPU, CUDA, and visualization.
                job.cancel();
            }
        }
        WorkerCommand::Shutdown => {
            state.stopped = true;
            if let Some(item) = state.ready.take() {
                item.cancel();
            }
            if let Some((_, job)) = &state.active {
                job.cancel();
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeWorkerThread {
    fn new() -> Self {
        let shared = Arc::new(NativeWorkerShared {
            state: Mutex::new(NativeWorkerState::default()),
            ready: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name(String::from("lsystem-render"))
            .spawn(move || native_render_worker(worker_shared))
            .expect("failed to start render worker");
        Self {
            shared,
            handle: Some(handle),
        }
    }

    fn submit(&self, request: AppVisualizeRequest) -> Task<Message> {
        let request_id = request.request_id;
        let (sender, receiver) = iced::futures::channel::oneshot::channel();
        let item = NativeWorkItem {
            request,
            outcome: sender,
        };
        let stopped = {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.stopped {
                item.cancel();
                true
            } else {
                if let Some(command) = state.queue.submit(item) {
                    apply_native_command(&mut state, command);
                }
                self.shared.ready.notify_one();
                false
            }
        };

        if stopped {
            return Task::done(Message::Rendered(RenderOutcome::Cancelled { request_id }));
        }
        Task::perform(
            async move {
                receiver
                    .await
                    .unwrap_or(RenderOutcome::Cancelled { request_id })
            },
            Message::Rendered,
        )
    }

    fn cancel_current(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(command) = state.queue.cancel() {
            apply_native_command(&mut state, command);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NativeWorkerThread {
    fn drop(&mut self) {
        {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let command = state.queue.shutdown();
            apply_native_command(&mut state, command);
            self.shared.ready.notify_one();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_render_worker(shared: Arc<NativeWorkerShared>) {
    // This worker owns reusable visualization device state. Keeping it here
    // avoids rebuilding CUDA modules for angle-only or cached-generation work.
    let mut turtle_streamer = Turtle2dStreamer::new();
    loop {
        let item = {
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            loop {
                if let Some(item) = state.ready.take() {
                    state.active = Some((item.request_id(), item.request.job.clone()));
                    break item;
                }
                if state.stopped {
                    return;
                }
                state = shared
                    .ready
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
            }
        };

        let request_id = item.request.request_id;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if item.request.job.is_cancelled() {
                RenderOutcome::Cancelled { request_id }
            } else {
                render_system(item.request, &mut turtle_streamer)
            }
        }));
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(panic) => {
                // A panic may leave reusable backend scratch state partially
                // updated. Recreate it before this worker accepts more work.
                turtle_streamer = Turtle2dStreamer::new();
                RenderOutcome::Failed {
                    request_id,
                    message: format!(
                        "render worker recovered from a panic: {}",
                        panic_message(panic)
                    ),
                }
            }
        };
        let _result = item.outcome.send(outcome);

        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.active = None;
        if !state.stopped
            && let Some(command) = state.queue.finish(request_id)
        {
            apply_native_command(&mut state, command);
            shared.ready.notify_one();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| String::from("unknown panic payload"))
}

#[cfg(target_arch = "wasm32")]
/// Browser transport for [`Worker`]. Routine replacement sends `Cancel` to a
/// persistent Web Worker. A watchdog recreates only a worker that fails or does
/// not acknowledge cancellation at a cooperative boundary.
struct BrowserWorker {
    state: std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct BrowserWorkerState {
    queue: WorkerQueue<WebWorkerRequest>,
    run: Option<WebWorkerRun>,
    next_worker_id: u64,
    stopped: bool,
}

#[cfg(target_arch = "wasm32")]
struct WebWorkerRun {
    worker_id: u64,
    ready: bool,
    worker: web_sys::Worker,
    active: Option<WebWorkerRequest>,
    _onmessage: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>,
    _onerror: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::ErrorEvent)>,
}

#[cfg(target_arch = "wasm32")]
struct WebWorkerRequest {
    request_id: u64,
    view_epoch: u64,
    request_json: String,
    outcome: Option<iced::futures::channel::oneshot::Sender<RenderOutcome>>,
    job: RenderJob,
    identity: RenderIdentity,
    iteration: usize,
    sent: bool,
}

#[cfg(target_arch = "wasm32")]
impl WorkerWork for WebWorkerRequest {
    fn request_id(&self) -> u64 {
        self.request_id
    }

    fn cancel(&self) {
        self.job.cancel();
    }
}

#[cfg(target_arch = "wasm32")]
impl BrowserWorker {
    fn new() -> Self {
        Self {
            state: std::rc::Rc::new(std::cell::RefCell::new(BrowserWorkerState::default())),
        }
    }

    fn submit(&self, request: AppVisualizeRequest) -> Task<Message> {
        use worker_protocol::{
            WorkerAmbiguousRules, WorkerFloatWidth, WorkerIterationTransition, WorkerRenderRequest,
            WorkerTurtleConfig,
        };

        let request_id = request.request_id;
        let identity = request_identity(&request);
        let iteration = request.key.iterations;
        let wire_request = WorkerRenderRequest {
            request_id,
            source: request.key.source.to_string(),
            ir_json: request.key.ir_json.as_deref().map(str::to_owned),
            iterations: request.key.iterations,
            angle: request.angle,
            visualizer: request.visualizer.to_string(),
            turtle: WorkerTurtleConfig {
                initial_angle: request.turtle_config.initial_angle,
                default_step: request.turtle_config.default_step,
                scale_multiplier: request.turtle_config.scale_multiplier,
                initial_width: request.turtle_config.initial_width,
                width_increment: request.turtle_config.width_increment,
                turn_angle_increment: request.turtle_config.turn_angle_increment,
                initial_color: request.turtle_config.initial_color.into(),
                color_increment: request.turtle_config.color_increment,
                palette: request.turtle_config.palette,
                background: request.turtle_config.background,
                draw_modules: request.turtle_config.draw_modules,
                move_modules: request.turtle_config.move_modules,
                module_aliases: request.turtle_config.module_aliases,
            },
            orientation_anchor: request
                .orientation_anchor
                .map(OrientationAnchor::as_str)
                .map(str::to_owned),
            seed: request.seed,
            float_width: match request.key.semantics.float_width {
                FloatWidth::F32 => WorkerFloatWidth::F32,
                FloatWidth::F64 => WorkerFloatWidth::F64,
            },
            ambiguous_rules: match request.key.semantics.ambiguous_rules {
                AmbiguousRulePolicy::First => WorkerAmbiguousRules::First,
                AmbiguousRulePolicy::Error => WorkerAmbiguousRules::Error,
                AmbiguousRulePolicy::Uniform => WorkerAmbiguousRules::Uniform,
            },
            transition: request
                .transition
                .map(|transition| WorkerIterationTransition {
                    from_iteration: transition.from_iteration,
                    to_iteration: transition.to_iteration,
                }),
        };
        let request_json =
            match serde_json::to_string(&worker_protocol::WorkerCommand::Run(wire_request)) {
                Ok(json) => json,
                Err(error) => {
                    self.cancel_current();
                    return Task::done(Message::Rendered(RenderOutcome::Failed {
                        request_id,
                        message: format!("could not serialize browser render request: {error}"),
                    }));
                }
            };
        let (sender, receiver) = iced::futures::channel::oneshot::channel();
        let task = Task::perform(
            async move {
                receiver
                    .await
                    .unwrap_or(RenderOutcome::Cancelled { request_id })
            },
            Message::Rendered,
        );
        let work = WebWorkerRequest {
            request_id,
            view_epoch: request.view_epoch,
            request_json,
            outcome: Some(sender),
            job: request.job,
            identity,
            iteration,
            sent: false,
        };

        let command = {
            let mut state = self.state.borrow_mut();
            if state.stopped {
                work.cancel();
                None
            } else {
                state.queue.submit(work)
            }
        };
        if let Some(command) = command {
            apply_browser_command(&self.state, command);
        }

        task
    }

    fn cancel_current(&self) {
        let command = self.state.borrow_mut().queue.cancel();
        if let Some(command) = command {
            apply_browser_command(&self.state, command);
        }
    }

    fn shutdown(&self) {
        let command = {
            let mut state = self.state.borrow_mut();
            state.stopped = true;
            state.queue.shutdown()
        };
        apply_browser_command(&self.state, command);
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for BrowserWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(target_arch = "wasm32")]
fn spawn_coordinated_web_worker(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
    worker_id: u64,
) -> Result<WebWorkerRun, String> {
    use wasm_bindgen::JsCast;

    let worker = spawn_web_worker()?;
    let message_state = std::rc::Rc::downgrade(state);
    let onmessage =
        wasm_bindgen::closure::Closure::wrap(Box::new(move |message: web_sys::MessageEvent| {
            let Some(message_state) = message_state.upgrade() else {
                return;
            };
            if message_state
                .borrow()
                .run
                .as_ref()
                .is_none_or(|run| run.worker_id != worker_id)
            {
                return;
            }
            let data = message.data();
            let (json, transferred_lines, transferred_morphs) = if let Some(json) = data.as_string()
            {
                (json, None, None)
            } else {
                let metadata =
                    js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("metadata"))
                        .ok()
                        .and_then(|value| value.as_string());
                let lines = js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("lines"))
                    .ok()
                    .and_then(|value| value.dyn_into::<js_sys::Array>().ok());
                let morphs =
                    js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str("morphs"))
                        .ok()
                        .and_then(|value| value.dyn_into::<js_sys::Array>().ok());
                let Some(metadata) = metadata else {
                    recover_browser_worker(
                        &message_state,
                        None,
                        String::from("browser render worker returned an invalid payload"),
                    );
                    return;
                };
                (metadata, lines, morphs)
            };
            let event = match serde_json::from_str::<worker_protocol::WorkerEvent>(&json) {
                Ok(event) => event,
                Err(error) => {
                    recover_browser_worker(
                        &message_state,
                        None,
                        format!("invalid browser render response: {error}"),
                    );
                    return;
                }
            };

            match event {
                worker_protocol::WorkerEvent::Ready => {
                    if let Some(run) = message_state.borrow_mut().run.as_mut() {
                        run.ready = true;
                    }
                    if let Err((request_id, message)) = dispatch_active_web_worker(&message_state) {
                        recover_browser_worker(&message_state, Some(request_id), message);
                    }
                }
                worker_protocol::WorkerEvent::Fatal { message } => {
                    recover_browser_worker(&message_state, None, message);
                }
                worker_protocol::WorkerEvent::Progress {
                    request_id,
                    phase,
                    phase_completed,
                    phase_total,
                    completed_iterations,
                    total_iterations,
                    modules,
                    items,
                    elapsed_millis,
                } => {
                    let job = message_state
                        .borrow()
                        .run
                        .as_ref()
                        .and_then(|run| run.active.as_ref())
                        .filter(|active| {
                            active.request_id == request_id && active.outcome.is_some()
                        })
                        .map(|active| active.job.clone());
                    let Some(job) = job else {
                        return;
                    };
                    if phase == "Visualizing" {
                        job.set_phase(RenderPhase::Visualizing);
                    } else {
                        job.set_derivation_progress(CalculationProgress {
                            phase: calculation_phase_from_worker(&phase),
                            phase_completed,
                            phase_total,
                            completed_iterations,
                            total_iterations,
                            modules,
                            items,
                            elapsed: Duration::from_millis(elapsed_millis),
                        });
                    }
                }
                worker_protocol::WorkerEvent::Cancelled { request_id } => {
                    complete_web_worker(
                        &message_state,
                        request_id,
                        RenderOutcome::Cancelled { request_id },
                    );
                }
                worker_protocol::WorkerEvent::Finished { request_id, result } => {
                    if !web_worker_request_is_current(&message_state, request_id) {
                        return;
                    }
                    let (view_epoch, identity, iteration) = message_state
                        .borrow()
                        .run
                        .as_ref()
                        .and_then(|run| run.active.as_ref())
                        .map_or((0, None, 0), |active| {
                            (
                                active.view_epoch,
                                Some(active.identity.clone()),
                                active.iteration,
                            )
                        });
                    let Some(identity) = identity else {
                        return;
                    };
                    let outcome = worker_result_to_outcome(
                        request_id,
                        view_epoch,
                        result,
                        transferred_lines,
                        transferred_morphs,
                        identity,
                        iteration,
                    );
                    complete_web_worker(&message_state, request_id, outcome);
                }
            }
        }) as Box<dyn FnMut(web_sys::MessageEvent)>);

    let error_state = std::rc::Rc::downgrade(state);
    let onerror =
        wasm_bindgen::closure::Closure::wrap(Box::new(move |error: web_sys::ErrorEvent| {
            let Some(error_state) = error_state.upgrade() else {
                return;
            };
            if error_state
                .borrow()
                .run
                .as_ref()
                .is_none_or(|run| run.worker_id != worker_id)
            {
                return;
            }
            let details = if error.message().is_empty() {
                String::from("browser render worker failed to load")
            } else {
                format!("browser render worker failed: {}", error.message())
            };
            recover_browser_worker(&error_state, None, details);
        }) as Box<dyn FnMut(web_sys::ErrorEvent)>);

    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
    Ok(WebWorkerRun {
        worker_id,
        ready: false,
        worker,
        active: None,
        _onmessage: onmessage,
        _onerror: onerror,
    })
}

#[cfg(target_arch = "wasm32")]
fn dispatch_active_web_worker(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
) -> Result<(), (u64, String)> {
    let request = {
        let mut state = state.borrow_mut();
        let Some(run) = state.run.as_mut() else {
            return Ok(());
        };
        if !run.ready {
            return Ok(());
        }
        let Some(active) = run.active.as_mut() else {
            return Ok(());
        };
        if active.sent {
            return Ok(());
        }
        active.sent = true;
        active.job.set_phase(RenderPhase::Deriving);
        (
            run.worker.clone(),
            active.request_id,
            active.request_json.clone(),
        )
    };

    request
        .0
        .post_message(&wasm_bindgen::JsValue::from_str(&request.2))
        .map_err(|error| {
            (
                request.1,
                format!("could not send work to browser render worker: {error:?}"),
            )
        })
}

#[cfg(target_arch = "wasm32")]
fn web_worker_request_is_current(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
    request_id: u64,
) -> bool {
    state
        .borrow()
        .run
        .as_ref()
        .and_then(|run| run.active.as_ref())
        .is_some_and(|active| active.request_id == request_id)
}

#[cfg(target_arch = "wasm32")]
fn complete_web_worker(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
    request_id: u64,
    outcome: RenderOutcome,
) {
    let (sender, command) = {
        let mut state = state.borrow_mut();
        let Some(run) = state.run.as_mut() else {
            return;
        };
        if run.active.as_ref().map(|active| active.request_id) != Some(request_id) {
            return;
        }
        let sender = run
            .active
            .take()
            .and_then(|mut active| active.outcome.take());
        let command = state.queue.finish(request_id);
        (sender, command)
    };
    if let Some(sender) = sender {
        let _result = sender.send(outcome);
    }
    if let Some(command) = command {
        apply_browser_command(state, command);
    }
}

#[cfg(target_arch = "wasm32")]
fn recover_browser_worker(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
    expected_request_id: Option<u64>,
    message: String,
) {
    let should_fail = state.borrow().run.as_ref().is_some_and(|run| {
        expected_request_id.is_none_or(|expected| {
            run.active.as_ref().map(|active| active.request_id) == Some(expected)
        })
    });
    if !should_fail {
        return;
    }
    let (run, active_was_cancelled) = {
        let mut state = state.borrow_mut();
        let active_was_cancelled = state.queue.cancelling;
        (state.run.take(), active_was_cancelled)
    };
    let Some(mut run) = run else {
        return;
    };
    let active = run.active.take();
    terminate_web_worker(run);

    let Some(mut active) = active else {
        return;
    };
    active.job.cancel();
    let request_id = active.request_id;
    if let Some(sender) = active.outcome.take() {
        let outcome = if active_was_cancelled {
            RenderOutcome::Cancelled { request_id }
        } else {
            RenderOutcome::Failed {
                request_id,
                message,
            }
        };
        let _result = sender.send(outcome);
    }
    let command = state.borrow_mut().queue.finish(request_id);
    if let Some(command) = command {
        apply_browser_command(state, command);
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_browser_command(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
    command: WorkerCommand<WebWorkerRequest>,
) {
    match command {
        WorkerCommand::Run(item) => {
            let spawn = {
                let mut state = state.borrow_mut();
                if state.stopped {
                    item.cancel();
                    return;
                }
                if state.run.is_none() {
                    state.next_worker_id = state.next_worker_id.wrapping_add(1);
                    Some(state.next_worker_id)
                } else {
                    None
                }
            };
            if let Some(worker_id) = spawn {
                match spawn_coordinated_web_worker(state, worker_id) {
                    Ok(run) => state.borrow_mut().run = Some(run),
                    Err(message) => {
                        let request_id = item.request_id;
                        if let Some(sender) = item.outcome {
                            let _result = sender.send(RenderOutcome::Failed {
                                request_id,
                                message,
                            });
                        }
                        let command = state.borrow_mut().queue.finish(request_id);
                        if let Some(command) = command {
                            apply_browser_command(state, command);
                        }
                        return;
                    }
                }
            }
            state
                .borrow_mut()
                .run
                .as_mut()
                .expect("browser worker was initialized")
                .active = Some(item);
            if let Err((request_id, message)) = dispatch_active_web_worker(state) {
                recover_browser_worker(state, Some(request_id), message);
            }
        }
        WorkerCommand::Cancel { request_id } => {
            let dispatch = {
                let mut state = state.borrow_mut();
                let Some(run) = state.run.as_mut() else {
                    return;
                };
                let Some(active) = run
                    .active
                    .as_ref()
                    .filter(|active| active.request_id == request_id)
                else {
                    return;
                };
                active.cancel();
                if active.sent {
                    Some((run.worker.clone(), run.worker_id))
                } else {
                    None
                }
            };
            let Some((worker, worker_id)) = dispatch else {
                complete_web_worker(state, request_id, RenderOutcome::Cancelled { request_id });
                return;
            };
            let command = worker_protocol::WorkerCommand::Cancel { request_id };
            let sent = serde_json::to_string(&command)
                .map_err(|error| error.to_string())
                .and_then(|json| {
                    worker
                        .post_message(&wasm_bindgen::JsValue::from_str(&json))
                        .map_err(|error| format!("could not cancel browser render work: {error:?}"))
                });
            if let Err(message) = sent {
                recover_browser_worker(state, Some(request_id), message);
            } else {
                arm_browser_cancel_watchdog(state, worker_id, request_id);
            }
        }
        WorkerCommand::Shutdown => {
            let run = state.borrow_mut().run.take();
            if let Some(mut run) = run {
                if let Some(active) = run.active.take() {
                    active.cancel();
                }
                let command = worker_protocol::WorkerCommand::Shutdown;
                if let Ok(json) = serde_json::to_string(&command) {
                    let _result = run
                        .worker
                        .post_message(&wasm_bindgen::JsValue::from_str(&json));
                }
                terminate_web_worker(run);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn arm_browser_cancel_watchdog(
    state: &std::rc::Rc<std::cell::RefCell<BrowserWorkerState>>,
    worker_id: u64,
    request_id: u64,
) {
    use wasm_bindgen::JsCast;

    // Cooperative browser work normally acknowledges cancellation first. This
    // timeout is only a recovery path for a panic or a long synchronous browser
    // call that prevents the Worker event loop from observing its message.
    const CANCEL_WATCHDOG_MILLIS: i32 = 1_000;
    let weak = std::rc::Rc::downgrade(state);
    let callback = wasm_bindgen::closure::Closure::once_into_js(move || {
        let Some(state) = weak.upgrade() else {
            return;
        };
        let still_unresponsive = {
            let state = state.borrow();
            state.queue.active == Some(request_id)
                && state.queue.cancelling
                && state
                    .run
                    .as_ref()
                    .is_some_and(|run| run.worker_id == worker_id)
        };
        if still_unresponsive {
            recover_browser_worker(
                &state,
                Some(request_id),
                String::from("browser render worker did not acknowledge cancellation"),
            );
        }
    });
    if let Some(window) = web_sys::window() {
        let _result = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.unchecked_ref(),
            CANCEL_WATCHDOG_MILLIS,
        );
    }
}

#[cfg(target_arch = "wasm32")]
fn terminate_web_worker(run: WebWorkerRun) {
    run.worker.set_onmessage(None);
    run.worker.set_onerror(None);
    run.worker.terminate();
}

#[cfg(target_arch = "wasm32")]
fn spawn_web_worker() -> Result<web_sys::Worker, String> {
    use wasm_bindgen::JsValue;

    let window = web_sys::window().ok_or_else(|| String::from("browser window is unavailable"))?;
    let base = window
        .location()
        .href()
        .map_err(|error| format!("browser URL is unavailable: {error:?}"))?;
    let script_url = web_sys::Url::new_with_base("braken-render-worker.js", &base)
        .map_err(|error| format!("could not resolve render worker script: {error:?}"))?
        .href();
    let wasm_url = web_sys::Url::new_with_base("braken-render-worker_bg.wasm", &base)
        .map_err(|error| format!("could not resolve render worker WASM: {error:?}"))?
        .href();
    let script_literal = serde_json::to_string(&script_url).map_err(|error| error.to_string())?;
    let wasm_literal = serde_json::to_string(&wasm_url).map_err(|error| error.to_string())?;
    let bootstrap = format!(
        "importScripts({script_literal});wasm_bindgen({wasm_literal}).catch(error=>{{setTimeout(()=>{{throw error;}},0);}});"
    );
    let parts = js_sys::Array::new();
    parts.push(&JsValue::from_str(&bootstrap));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("text/javascript");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|error| format!("could not create render worker bootstrap: {error:?}"))?;
    let object_url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|error| format!("could not create render worker URL: {error:?}"))?;
    let worker = web_sys::Worker::new(&object_url)
        .map_err(|error| format!("could not start browser render worker: {error:?}"));
    let _result = web_sys::Url::revoke_object_url(&object_url);
    worker
}

#[cfg(target_arch = "wasm32")]
fn worker_result_to_outcome(
    request_id: u64,
    view_epoch: u64,
    result: Result<worker_protocol::WorkerRenderResult, String>,
    transferred_lines: Option<js_sys::Array>,
    transferred_morphs: Option<js_sys::Array>,
    identity: RenderIdentity,
    iteration: usize,
) -> RenderOutcome {
    use worker_protocol::{WorkerScene, WorkerTextRole};

    let result = result.and_then(|result| {
        let _layout = result.transfer_layout()?;
        let line_values = transferred_lines
            .ok_or_else(|| String::from("browser-worker result omitted its line buffer"))?;
        let backend = format!(
            "{} derive · {} visualize · Web Worker",
            result.derivation_backend, result.visualization_backend,
        );
        let ir_tooling = IrToolingSnapshot {
            disassembly: Arc::from(result.ir_disassembly.clone()),
            json: result.ir_json.clone().map(Arc::from),
        };
        let (polygons, texts, bounds) = match result.scene {
            WorkerScene::TwoD {
                polygons,
                texts,
                bounds,
            } => (polygons, texts, bounds),
            WorkerScene::ThreeD { polygons, bounds } => {
                let width_reference = result
                    .width_reference
                    .ok_or_else(|| String::from("browser worker omitted its 3D width reference"))?;
                if result.transition.is_some() || transferred_morphs.is_some() {
                    return Err(String::from(
                        "browser worker returned transition data for a 3D scene",
                    ));
                }
                let element_count = result
                    .line_count
                    .checked_add(polygons.len())
                    .ok_or_else(|| String::from("browser scene contains too many primitives"))?;
                let mut spatial_polygons = Vec::new();
                spatial_polygons
                    .try_reserve_exact(polygons.len())
                    .map_err(|_| String::from("not enough browser memory for spatial polygons"))?;
                for polygon in polygons {
                    let mut vertices = Vec::new();
                    vertices
                        .try_reserve_exact(polygon.vertices.len())
                        .map_err(|_| {
                            String::from("not enough browser memory for spatial polygon vertices")
                        })?;
                    vertices.extend(polygon.vertices.into_iter().map(|[x, y, z]| (x, y, z)));
                    spatial_polygons.push(SpatialPolygon3d::new(
                        braken_viz::Polygon3d {
                            vertices,
                            color: polygon.color.into(),
                        },
                        polygon.lines_before,
                    ));
                }
                let bounds = bounds.map_or_else(RenderBounds3d::default, |bounds| RenderBounds3d {
                    min_x: bounds[0],
                    max_x: bounds[1],
                    min_y: bounds[2],
                    max_y: bounds[3],
                    min_z: bounds[4],
                    max_z: bounds[5],
                });
                let spatial_scene = SpatialScene::from_transferred(
                    line_values,
                    result.line_count,
                    spatial_polygons,
                    bounds,
                    result.total_line_length,
                    width_reference,
                    result.background,
                )
                .map_err(|error| error.to_string())?;
                return Ok(RenderOutcome::Ready(RenderResult {
                    request_id,
                    view_epoch,
                    scene: None,
                    line_scene: None,
                    spatial_scene: Some(spatial_scene),
                    element_count,
                    refinement_started_at: RenderInstant::now(),
                    elapsed_ms: result.elapsed_millis as u128,
                    backend,
                    bounds: RenderBounds::default(),
                    cache_insert: None,
                    iteration,
                    identity,
                    ir_tooling,
                }));
            }
        };
        let element_count = result
            .line_count
            .checked_add(texts.len())
            .and_then(|count| count.checked_add(polygons.len()))
            .ok_or_else(|| String::from("browser scene contains too many primitives"))?;
        let has_polygons = !polygons.is_empty();
        let mut primitives = Vec::new();
        primitives
            .try_reserve_exact(texts.len().saturating_add(polygons.len()))
            .map_err(|_| String::from("not enough UI memory for browser-worker primitives"))?;
        primitives.extend(polygons.into_iter().map(|item| {
            Primitive2d::Polygon(Polygon2d {
                vertices: item.vertices.into_iter().map(|[x, y]| (x, y)).collect(),
                color: item.color.into(),
            })
        }));
        primitives.extend(texts.into_iter().map(|item| {
            Primitive2d::Text(Text2d {
                position: (item.x, item.y),
                content: item.content,
                size: item.size,
                role: match item.role {
                    WorkerTextRole::Title => TextRole::Title,
                    WorkerTextRole::Heading => TextRole::Heading,
                    WorkerTextRole::Body => TextRole::Body,
                    WorkerTextRole::Muted => TextRole::Muted,
                    WorkerTextRole::Error => TextRole::Error,
                },
            })
        }));
        let bounds = bounds.map_or_else(RenderBounds::default, |bounds| RenderBounds {
            min_x: bounds[0],
            max_x: bounds[1],
            min_y: bounds[2],
            max_y: bounds[3],
        });
        let mut line_scene = (result.line_count > 0)
            .then(|| {
                LineScene::from_transferred(
                    line_values,
                    result.line_count,
                    bounds,
                    result.total_line_length,
                    result.background,
                )
            })
            .transpose()
            .map_err(|error| error.to_string())?;
        if let Some(lines) = &line_scene {
            let preview = lines
                .fallback_preview(if has_polygons {
                    usize::MAX
                } else {
                    SOFTWARE_FALLBACK_PREVIEW_LINES
                })
                .map_err(|error| error.to_string())?;
            primitives
                .try_reserve(preview.primitives.len())
                .map_err(|_| String::from("not enough UI memory for the software preview"))?;
            primitives.extend(preview.primitives);
        }
        if has_polygons {
            line_scene = None;
        }
        let transition = result.transition.clone();
        let target = RenderResult {
            request_id,
            view_epoch,
            scene: (!primitives.is_empty()).then(|| {
                Arc::new(Scene2d {
                    primitives,
                    background: result.background,
                })
            }),
            line_scene,
            spatial_scene: None,
            element_count,
            refinement_started_at: RenderInstant::now(),
            elapsed_ms: result.elapsed_millis as u128,
            backend,
            bounds,
            cache_insert: None,
            iteration,
            identity,
            ir_tooling,
        };
        if let Some(transition) = transition {
            let morph_values = transferred_morphs.ok_or_else(|| {
                String::from("browser-worker transition omitted its morph buffer")
            })?;
            let source_bounds =
                transition
                    .source_bounds
                    .map_or_else(RenderBounds::default, |bounds| RenderBounds {
                        min_x: bounds[0],
                        max_x: bounds[1],
                        min_y: bounds[2],
                        max_y: bounds[3],
                    });
            let scene = TransitionScene::from_transferred(
                morph_values,
                transition.morph_count,
                source_bounds,
                bounds,
                (
                    transition.source_line_count,
                    transition.source_total_line_length,
                ),
                (result.line_count, result.total_line_length),
                result.background,
            )
            .map_err(|error| error.to_string())?;
            Ok(RenderOutcome::TransitionReady(
                PreparedIterationTransition {
                    request_id,
                    from_iteration: transition.from_iteration,
                    to_iteration: transition.to_iteration,
                    scene,
                    target,
                },
            ))
        } else {
            Ok(RenderOutcome::Ready(target))
        }
    });

    match result {
        Ok(outcome) => outcome,
        Err(message) => RenderOutcome::Failed {
            request_id,
            message,
        },
    }
}

#[cfg(target_arch = "wasm32")]
fn calculation_phase_from_worker(phase: &str) -> CalculationPhase {
    match phase {
        "Preparing" => CalculationPhase::Preparing,
        "InspectingInput" => CalculationPhase::InspectingInput,
        "Indexing" => CalculationPhase::Indexing,
        "SelectingProductions" => CalculationPhase::SelectingProductions,
        "Rewriting" => CalculationPhase::Rewriting,
        "ValidatingOutput" => CalculationPhase::ValidatingOutput,
        "Transferring" => CalculationPhase::Transferring,
        "Complete" => CalculationPhase::Complete,
        _ => CalculationPhase::Rewriting,
    }
}

type GenerationForRequest = (
    Arc<Generation>,
    Option<(GenerationCacheKey, Arc<Generation>, DerivationInfo, usize)>,
    DerivationInfo,
    IrToolingSnapshot,
);

fn generation_for_request(request: &AppVisualizeRequest) -> Result<GenerationForRequest, String> {
    let grammar = compiled_grammar_for_key(&request.key)?;
    let ir_tooling = ir_tooling_snapshot(&grammar, request.key.semantics);
    if let Some(cached) = &request.cached_generation {
        return Ok((
            Arc::clone(&cached.generation),
            None,
            cached.derivation.as_cache_hit(),
            ir_tooling,
        ));
    }

    request.job.set_phase(RenderPhase::Deriving);
    let result = calculate_with_control(
        CalculationRequest {
            grammar,
            iterations: request.key.iterations,
            backend: BackendChoice::Auto,
            seed: request.seed,
            semantics: request.key.semantics,
            limits: CalculationLimits::default(),
        },
        &request.job.calculation_cancel,
        |progress| {
            request.job.set_derivation_progress(progress);
            !request.job.is_cancelled()
        },
    )
    .map_err(|err| err.to_string())?;
    let estimated_bytes =
        estimate_cache_entry_bytes(&request.key, result.stats.items, result.stats.modules);
    let info = DerivationInfo {
        backend: match result.backend_used {
            BackendChoice::Cpu => "CPU",
            BackendChoice::Cuda => "CUDA",
            BackendChoice::Wgpu => "WGPU",
            BackendChoice::Auto => "auto",
            BackendChoice::Named(_) => "custom",
        },
        elapsed: Some(result.stats.elapsed),
        cached: false,
    };
    let generation = Arc::new(result.generation);
    let cache_insert = Some((
        request.key.clone(),
        Arc::clone(&generation),
        info,
        estimated_bytes,
    ));

    Ok((generation, cache_insert, info, ir_tooling))
}

fn compiled_grammar_for_key(key: &GenerationCacheKey) -> Result<CompiledGrammar, String> {
    if let Some(json) = key.ir_json.as_deref() {
        ir_tooling::compile_json(json).map_err(|error| error.to_string())
    } else {
        CompiledGrammar::parse(&key.source).map_err(|error| error.to_string())
    }
}

fn ir_tooling_snapshot(
    grammar: &CompiledGrammar,
    semantics: DerivationSemantics,
) -> IrToolingSnapshot {
    ir_tooling::snapshot_with_semantics(grammar, semantics).into()
}

impl From<ir_tooling::IrToolingSnapshot> for IrToolingSnapshot {
    fn from(snapshot: ir_tooling::IrToolingSnapshot) -> Self {
        Self {
            disassembly: Arc::from(snapshot.disassembly),
            json: snapshot.json.map(Arc::from),
        }
    }
}

fn visualize_generation(
    generation: &Generation,
    request: &AppVisualizeRequest,
    info: DerivationInfo,
    turtle_streamer: &mut Turtle2dStreamer,
) -> Result<PreparedDisplay, String> {
    // This stage consumes a completed Generation. Its backend policy is
    // independent of the derivation backend recorded in `info`, and the output
    // remains independent of the WGPU/Canvas display choice.
    if request.visualizer == VisualizerKind::Turtle2d {
        let mut config = request.turtle_config.clone();
        config.turn_angle = (request.angle as f64).to_radians();
        let orientation_reference = config.initial_angle;
        let mut builder = LineSceneBuilder::default();
        builder.set_background(config.background);
        let mut polygons = Vec::<Polygon2d>::new();
        let is_cancelled = || request.job.is_cancelled();
        let summary = turtle_streamer
            .stream(
                VisualizerBackend::Auto,
                Turtle2dStreamRequest {
                    generation,
                    config,
                    batch_size: 16 * 1024,
                    is_cancelled: &is_cancelled,
                },
                |batch| {
                    request.job.set_visualization_progress(batch.progress);
                    polygons.try_reserve(batch.polygons.len()).map_err(|_| {
                        VisualizeError::ResourceExhausted {
                            resource: "GUI turtle polygons",
                            requested: Some(polygons.len().saturating_add(batch.polygons.len())),
                        }
                    })?;
                    polygons.extend(batch.polygons);
                    builder
                        .extend(batch.lines)
                        .map_err(|error| VisualizeError::ResourceExhausted {
                            resource: "GUI GPU line scene",
                            requested: match error {
                                line_shader::LineSceneError::ResourceExhausted {
                                    requested_lines,
                                } => Some(requested_lines),
                                line_shader::LineSceneError::Cancelled => None,
                                line_shader::LineSceneError::InvalidTransfer { .. } => None,
                            },
                        })
                },
            )
            .map_err(|error| error.to_string())?;
        let bounds = summary
            .bounds
            .map_or_else(RenderBounds::default, |bounds| RenderBounds {
                min_x: bounds.min.0 as f32,
                max_x: bounds.max.0 as f32,
                min_y: bounds.min.1 as f32,
                max_y: bounds.max.1 as f32,
            });
        let element_count = summary.progress.lines_emitted;
        let orientation_anchor = polygons
            .is_empty()
            .then_some(request.orientation_anchor)
            .flatten();
        let (line_scene, bounds, _) =
            builder.finish_oriented(bounds, orientation_anchor, orientation_reference);
        if !polygons.is_empty() {
            let mut scene = line_scene
                .fallback_preview(usize::MAX)
                .map_err(|error| error.to_string())?;
            scene
                .primitives
                .try_reserve(polygons.len())
                .map_err(|_| String::from("not enough memory for GUI turtle polygons"))?;
            scene
                .primitives
                .extend(polygons.into_iter().map(Primitive2d::Polygon));
            return Ok(PreparedDisplay {
                scene: Some(Arc::new(scene)),
                line_scene: None,
                spatial_scene: None,
                element_count: element_count.saturating_add(summary.progress.polygons_emitted),
                bounds,
                visualization_backend: summary.backend_used,
            });
        }
        let scene = line_scene
            .fallback_preview(SOFTWARE_FALLBACK_PREVIEW_LINES)
            .map(Arc::new)
            .map_err(|error| error.to_string())?;
        return Ok(PreparedDisplay {
            scene: Some(scene),
            line_scene: Some(line_scene),
            spatial_scene: None,
            element_count,
            bounds,
            visualization_backend: summary.backend_used,
        });
    }

    if request.visualizer == VisualizerKind::Turtle3d {
        let mut turtle = request.turtle_config.clone();
        turtle.turn_angle = (request.angle as f64).to_radians();
        let config = Turtle3dConfig::from(turtle);
        let mut builder = SpatialSceneBuilder::new(config.turtle.background);
        let is_cancelled = || request.job.is_cancelled();
        let summary = stream_turtle_3d(
            Turtle3dStreamRequest {
                generation,
                config,
                batch_size: 16 * 1024,
                is_cancelled: &is_cancelled,
            },
            |batch| {
                request
                    .job
                    .set_spatial_visualization_progress(batch.progress);
                builder
                    .extend(batch.lines, batch.polygons, batch.primitive_order)
                    .map_err(spatial_scene_visualize_error)
            },
        )
        .map_err(|error| error.to_string())?;
        return Ok(PreparedDisplay {
            scene: None,
            line_scene: None,
            spatial_scene: Some(builder.finish().map_err(|error| error.to_string())?),
            element_count: summary
                .progress
                .lines_emitted
                .saturating_add(summary.progress.polygons_emitted),
            bounds: RenderBounds::default(),
            visualization_backend: summary.backend_used,
        });
    }

    let context = VisualizationContext {
        iterations: request.key.iterations,
        seed: request.seed,
        derivation_backend: Some(info.label()),
        elapsed: info.elapsed,
    };
    if request.visualizer == VisualizerKind::Inspector {
        return prepare_scene_display(
            inspect_generation(generation, context, &request.job)?,
            VisualizerBackend::Cpu,
        );
    }
    let config = VisualizerConfig::for_kind(request.visualizer);
    let first = visualize_with_backend(braken_viz::VisualizeRequest {
        generation,
        backend: VisualizerBackend::Auto,
        config: config.clone(),
        context,
    });
    let output = match first {
        Err(VisualizeError::Unimplemented { .. }) => {
            return prepare_scene_display(
                inspect_generation(generation, context, &request.job)?,
                VisualizerBackend::Cpu,
            );
        }
        result => result,
    }
    .map_err(|error| error.to_string())?;
    match output.visualization {
        Visualization::Scene2d(scene) => prepare_scene_display(scene, output.backend_used),
        Visualization::Scene3d(scene) => prepare_spatial_display(scene, output.backend_used),
    }
}

fn inspect_generation(
    generation: &Generation,
    context: VisualizationContext,
    job: &RenderJob,
) -> Result<Scene2d, String> {
    let is_cancelled = || job.is_cancelled();
    stream_inspector(
        InspectorStreamRequest {
            generation,
            config: InspectorConfig,
            context,
            work_quantum: 16 * 1024,
            is_cancelled: &is_cancelled,
        },
        |_| {
            if job.is_cancelled() {
                Err(VisualizeError::Cancelled)
            } else {
                Ok(())
            }
        },
    )
    .map_err(|error| error.to_string())
}

fn prepare_scene_display(
    scene: Scene2d,
    visualization_backend: VisualizerBackend,
) -> Result<PreparedDisplay, String> {
    let bounds = scene_bounds(&scene);
    let element_count = scene.primitives.len();
    let line_scene = LineScene::from_scene(&scene, bounds).map_err(|error| error.to_string())?;
    let scene = match &line_scene {
        Some(lines) => Arc::new(
            lines
                .fallback_preview(SOFTWARE_FALLBACK_PREVIEW_LINES)
                .map_err(|error| error.to_string())?,
        ),
        None => Arc::new(scene),
    };
    Ok(PreparedDisplay {
        scene: Some(scene),
        line_scene,
        spatial_scene: None,
        element_count,
        bounds,
        visualization_backend,
    })
}

fn prepare_spatial_display(
    scene: Scene3d,
    visualization_backend: VisualizerBackend,
) -> Result<PreparedDisplay, String> {
    let element_count = scene.primitives.len();
    let spatial_scene = SpatialScene::from_scene(scene).map_err(|error| error.to_string())?;
    Ok(PreparedDisplay {
        scene: None,
        line_scene: None,
        spatial_scene: Some(spatial_scene),
        element_count,
        bounds: RenderBounds::default(),
        visualization_backend,
    })
}

fn spatial_scene_visualize_error(error: SpatialSceneError) -> VisualizeError {
    match error {
        SpatialSceneError::ResourceExhausted { requested } => VisualizeError::ResourceExhausted {
            resource: "GUI spatial scene",
            requested: Some(requested),
        },
        error => VisualizeError::InvalidConfiguration(error.to_string()),
    }
}

fn visualizer_backend_name(backend: VisualizerBackend) -> &'static str {
    match backend {
        VisualizerBackend::Auto => "auto",
        VisualizerBackend::Cpu => "CPU",
        VisualizerBackend::Cuda => "CUDA",
    }
}

fn cubic_bezier_ease(progress: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    if progress <= 0.0 {
        return 0.0;
    }
    if progress >= 1.0 {
        return 1.0;
    }

    fn coordinate(t: f32, first: f32, second: f32) -> f32 {
        let inverse = 1.0 - t;
        3.0 * inverse * inverse * t * first + 3.0 * inverse * t * t * second + t * t * t
    }

    fn derivative(t: f32, first: f32, second: f32) -> f32 {
        3.0 * (1.0 - t) * (1.0 - t) * first
            + 6.0 * (1.0 - t) * t * (second - first)
            + 3.0 * t * t * (1.0 - second)
    }

    let mut parameter = progress;
    for _ in 0..8 {
        let difference = coordinate(parameter, x1, x2) - progress;
        let slope = derivative(parameter, x1, x2);
        if difference.abs() < 1.0e-5 || slope.abs() < 1.0e-6 {
            break;
        }
        parameter = (parameter - difference / slope).clamp(0.0, 1.0);
    }

    if (coordinate(parameter, x1, x2) - progress).abs() >= 1.0e-5 {
        let (mut low, mut high) = (0.0, 1.0);
        for _ in 0..12 {
            parameter = (low + high) * 0.5;
            if coordinate(parameter, x1, x2) < progress {
                low = parameter;
            } else {
                high = parameter;
            }
        }
    }

    coordinate(parameter, y1, y2)
}

fn iteration_ease(progress: f32) -> f32 {
    cubic_bezier_ease(progress, 0.4, 0.0, 0.2, 1.0)
}

fn turtle_configs_equal(left: &Turtle2dConfig, right: &Turtle2dConfig) -> bool {
    left.initial_angle.to_bits() == right.initial_angle.to_bits()
        && left.default_step.to_bits() == right.default_step.to_bits()
        && left.scale_multiplier.to_bits() == right.scale_multiplier.to_bits()
        && left.initial_width.to_bits() == right.initial_width.to_bits()
        && left.width_increment.to_bits() == right.width_increment.to_bits()
        && left.turn_angle_increment.to_bits() == right.turn_angle_increment.to_bits()
        && left.initial_color == right.initial_color
        && left.color_increment == right.color_increment
        && left.palette == right.palette
        && left.background == right.background
        && left.draw_modules == right.draw_modules
        && left.move_modules == right.move_modules
        && left.module_aliases == right.module_aliases
}

#[derive(Debug, Clone, Copy)]
struct DerivationInfo {
    /// Informational derivation backend; never a visualizer or display backend.
    backend: &'static str,
    elapsed: Option<std::time::Duration>,
    cached: bool,
}

impl DerivationInfo {
    fn as_cache_hit(mut self) -> Self {
        self.cached = true;
        self
    }

    fn label(self) -> &'static str {
        match (self.backend, self.cached) {
            ("CPU", true) => "CPU (cached)",
            ("CUDA", true) => "CUDA (cached)",
            ("WGPU", true) => "WGPU (cached)",
            ("auto", true) => "auto (cached)",
            ("custom", true) => "custom (cached)",
            (backend, false) => backend,
            (_, true) => "cached",
        }
    }
}

fn scene_bounds(scene: &Scene2d) -> RenderBounds {
    let bounds = scene
        .primitives
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive2d::Line(line) => Some(RenderLine {
                start: (line.line.0.0 as f32, line.line.0.1 as f32),
                end: (line.line.1.0 as f32, line.line.1.1 as f32),
                width: line.width as f32,
            }),
            Primitive2d::Polygon(polygon) => polygon.vertices.first().map(|first| {
                polygon.vertices.iter().fold(
                    RenderLine {
                        start: (first.0 as f32, first.1 as f32),
                        end: (first.0 as f32, first.1 as f32),
                        width: 0.0,
                    },
                    |bounds, point| RenderLine {
                        start: (
                            bounds.start.0.min(point.0 as f32),
                            bounds.start.1.min(point.1 as f32),
                        ),
                        end: (
                            bounds.end.0.max(point.0 as f32),
                            bounds.end.1.max(point.1 as f32),
                        ),
                        width: 0.0,
                    },
                )
            }),
            Primitive2d::Text(_) => None,
        })
        .fold(
            RenderBounds {
                min_x: f32::INFINITY,
                max_x: f32::NEG_INFINITY,
                min_y: f32::INFINITY,
                max_y: f32::NEG_INFINITY,
            },
            include_render_line,
        );
    if bounds.min_x.is_finite()
        && bounds.max_x.is_finite()
        && bounds.min_y.is_finite()
        && bounds.max_y.is_finite()
    {
        bounds
    } else {
        RenderBounds::default()
    }
}

#[cfg(test)]
fn render_bounds(lines: &[RenderLine]) -> RenderBounds {
    lines.iter().copied().fold(
        RenderBounds {
            min_x: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            min_y: f32::INFINITY,
            max_y: f32::NEG_INFINITY,
        },
        include_render_line,
    )
}

fn include_render_line(bounds: RenderBounds, line: RenderLine) -> RenderBounds {
    RenderBounds {
        min_x: bounds.min_x.min(line.start.0).min(line.end.0),
        max_x: bounds.max_x.max(line.start.0).max(line.end.0),
        min_y: bounds.min_y.min(line.start.1).min(line.end.1),
        max_y: bounds.max_y.max(line.start.1).max(line.end.1),
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Encodes the complete world-space scene. Planar pan/zoom never crops the
/// export; spatial geometry is projected at the orbit captured by the caller.
async fn prepare_and_export_svg(
    line_scene: Option<Arc<LineScene>>,
    scene: Option<Arc<Scene2d>>,
    spatial_scene: Option<Arc<SpatialScene>>,
    orbit: Orbit3d,
    palette: Palette,
    line_width_scale: f32,
    cancelled: Arc<AtomicBool>,
) -> Result<String, String> {
    let data = match (spatial_scene, line_scene, scene) {
        (Some(spatial), _, _) => {
            let projected = spatial
                .project_for_svg(orbit, palette, &|| cancelled.load(Ordering::Acquire))
                .map_err(|error| format!("could not project spatial SVG data: {error}"))?;
            svg_target::encode_with_stroke_scale(&projected, palette, f64::from(line_width_scale))
        }
        (None, Some(lines), _) => lines
            .encode_svg_with_cancel(palette, line_width_scale, &|| {
                cancelled.load(Ordering::Acquire)
            })
            .map_err(|error| format!("could not prepare SVG data: {error}"))?,
        (None, None, Some(scene)) => {
            svg_target::encode_with_stroke_scale(&scene, palette, f64::from(line_width_scale))
        }
        (None, None, None) => svg_target::encode_with_stroke_scale(
            &Scene2d::default(),
            palette,
            f64::from(line_width_scale),
        ),
    };
    if cancelled.load(Ordering::Acquire) {
        return Err(String::from("SVG export cancelled"));
    }
    export_svg(data).await
}

#[cfg(target_arch = "wasm32")]
/// Encodes the complete world-space scene. Planar line encoding yields in
/// bounded batches; spatial geometry uses the captured orbit and remains
/// cooperatively cancellable while it is projected.
async fn prepare_and_export_svg(
    line_scene: Option<Arc<LineScene>>,
    scene: Option<Arc<Scene2d>>,
    spatial_scene: Option<Arc<SpatialScene>>,
    orbit: Orbit3d,
    palette: Palette,
    line_width_scale: f32,
    cancelled: Arc<AtomicBool>,
) -> Result<String, String> {
    let data = match (spatial_scene, line_scene, scene) {
        (Some(spatial), _, _) => spatial
            .encode_svg_yielding(orbit, palette, line_width_scale, &|| {
                cancelled.load(Ordering::Acquire)
            })
            .await
            .map_err(|error| format!("could not prepare spatial SVG data: {error}"))?,
        (None, Some(lines), _) => lines
            .encode_svg_yielding(palette, line_width_scale, &|| {
                cancelled.load(Ordering::Acquire)
            })
            .await
            .map_err(|error| format!("could not prepare SVG data: {error}"))?,
        (None, None, Some(scene)) => {
            svg_target::encode_with_stroke_scale(&scene, palette, f64::from(line_width_scale))
        }
        (None, None, None) => svg_target::encode_with_stroke_scale(
            &Scene2d::default(),
            palette,
            f64::from(line_width_scale),
        ),
    };
    if cancelled.load(Ordering::Acquire) {
        return Err(String::from("SVG export cancelled"));
    }
    export_svg(data).await
}

#[cfg(not(target_arch = "wasm32"))]
async fn export_svg(data: String) -> Result<String, String> {
    let path = std::path::Path::new("lsystem.svg");
    std::fs::write(path, data).map_err(|error| error.to_string())?;
    Ok(format!("Saved {}", path.display()))
}

#[cfg(target_arch = "wasm32")]
async fn export_svg(data: String) -> Result<String, String> {
    use wasm_bindgen::JsCast;
    let parts = js_sys::Array::new();
    parts.push(&wasm_bindgen::JsValue::from_str(&data));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("image/svg+xml");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|error| format!("{error:?}"))?;
    let url =
        web_sys::Url::create_object_url_with_blob(&blob).map_err(|error| format!("{error:?}"))?;
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| String::from("browser document is unavailable"))?;
    let anchor = document
        .create_element("a")
        .map_err(|error| format!("{error:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|error| format!("{error:?}"))?;
    anchor.set_href(&url);
    anchor.set_download("lsystem.svg");
    anchor.click();
    web_sys::Url::revoke_object_url(&url).map_err(|error| format!("{error:?}"))?;
    Ok(String::from("Downloaded lsystem.svg"))
}

#[cfg(not(target_arch = "wasm32"))]
async fn export_ir_json(data: String) -> Result<String, String> {
    let path = std::path::Path::new("lsystem.ir.json");
    std::fs::write(path, data).map_err(|error| error.to_string())?;
    Ok(format!("Saved {}", path.display()))
}

#[cfg(target_arch = "wasm32")]
async fn export_ir_json(data: String) -> Result<String, String> {
    use wasm_bindgen::JsCast;
    let parts = js_sys::Array::new();
    parts.push(&wasm_bindgen::JsValue::from_str(&data));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/json");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|error| format!("{error:?}"))?;
    let url =
        web_sys::Url::create_object_url_with_blob(&blob).map_err(|error| format!("{error:?}"))?;
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| String::from("browser document is unavailable"))?;
    let anchor = document
        .create_element("a")
        .map_err(|error| format!("{error:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|error| format!("{error:?}"))?;
    anchor.set_href(&url);
    anchor.set_download("lsystem.ir.json");
    anchor.click();
    web_sys::Url::revoke_object_url(&url).map_err(|error| format!("{error:?}"))?;
    Ok(String::from("Downloaded lsystem.ir.json"))
}

/// Rotation-only input shared by desktop and mobile spatial views.
///
/// The first touch is owned immediately: tapping stops preset autorotation and
/// dragging rotates. Capture remains sticky until every participating finger
/// lifts, including fingers added outside the canvas during an owned gesture.
struct SpatialViewportController {
    orbit: Orbit3d,
    epoch: u64,
}

#[derive(Debug, Default)]
struct SpatialViewportControllerState {
    epoch: u64,
    last_bounds: Option<Rectangle>,
    mouse_drag: Option<SpatialDragGesture>,
    touches: Vec<ActiveTouch>,
    touch_gesture: Option<SpatialDragGesture>,
    touch_captured: bool,
    working_orbit: Option<Orbit3d>,
}

#[derive(Debug, Clone, Copy)]
struct SpatialDragGesture {
    baseline: Orbit3d,
    initial_position: ScreenPoint,
}

impl SpatialViewportController {
    fn viewport(bounds: Rectangle) -> ViewportSize {
        ViewportSize::new(f64::from(bounds.width), f64::from(bounds.height))
    }

    fn local_position(position: Point, bounds: Rectangle) -> ScreenPoint {
        ScreenPoint::new(
            f64::from(position.x - bounds.x),
            f64::from(position.y - bounds.y),
        )
    }

    fn update_touch_position(
        state: &mut SpatialViewportControllerState,
        id: iced::touch::Finger,
        position: Point,
    ) -> bool {
        let Some(touch) = state.touches.iter_mut().find(|touch| touch.id == id) else {
            return false;
        };
        touch.position = position;
        true
    }

    fn touch_centroid(
        state: &SpatialViewportControllerState,
        bounds: Rectangle,
    ) -> Option<ScreenPoint> {
        let first = state.touches.first()?;
        let first = Self::local_position(first.position, bounds);
        let Some(second) = state.touches.get(1) else {
            return Some(first);
        };
        let second = Self::local_position(second.position, bounds);
        Some(ScreenPoint::new(
            (first.x + second.x) * 0.5,
            (first.y + second.y) * 0.5,
        ))
    }

    fn rebase_touch(&self, state: &mut SpatialViewportControllerState, bounds: Rectangle) {
        let Some(centroid) = Self::touch_centroid(state, bounds) else {
            state.touch_gesture = None;
            return;
        };
        let baseline = state.working_orbit.unwrap_or(self.orbit);
        state.working_orbit = Some(baseline);
        state.touch_gesture = Some(SpatialDragGesture {
            baseline,
            initial_position: centroid,
        });
    }

    fn touch_orbit(
        &self,
        state: &SpatialViewportControllerState,
        bounds: Rectangle,
    ) -> Option<Orbit3d> {
        let gesture = state.touch_gesture?;
        let centroid = Self::touch_centroid(state, bounds)?;
        Some(gesture.baseline.arcball_drag(
            gesture.initial_position,
            centroid,
            Self::viewport(bounds),
        ))
    }

    fn reset_if_obsolete(
        &self,
        state: &mut SpatialViewportControllerState,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) {
        let previous_bounds = state.last_bounds.replace(bounds);
        let bounds_changed = previous_bounds.is_some_and(|previous| previous != bounds);
        if state.epoch == self.epoch && !bounds_changed {
            return;
        }
        state.epoch = self.epoch;
        if let Some(drag) = state.mouse_drag.as_mut() {
            let position = cursor
                .position_from(bounds.position())
                .map(|point| ScreenPoint::new(f64::from(point.x), f64::from(point.y)))
                .unwrap_or(drag.initial_position);
            *drag = SpatialDragGesture {
                baseline: self.orbit,
                initial_position: position,
            };
            state.working_orbit = Some(self.orbit);
        } else if state.touch_captured {
            state.working_orbit = Some(self.orbit);
            self.rebase_touch(state, bounds);
        } else {
            state.touch_gesture = None;
            state.working_orbit = None;
        }
    }

    fn finish_touch(
        &self,
        state: &mut SpatialViewportControllerState,
        id: iced::touch::Finger,
        orbit: Orbit3d,
        bounds: Rectangle,
    ) -> Option<canvas::Action<Message>> {
        state.touches.retain(|touch| touch.id != id);
        if !state.touch_captured {
            state.working_orbit = None;
            state.touch_gesture = None;
            return None;
        }
        if state.touches.is_empty() {
            state.touch_captured = false;
            state.touch_gesture = None;
            state.working_orbit = None;
            return Some(
                canvas::Action::publish(Message::Viewport(ViewportMessage::OrbitGestureEnded(
                    orbit,
                )))
                .and_capture(),
            );
        }
        state.working_orbit = Some(orbit);
        self.rebase_touch(state, bounds);
        Some(
            canvas::Action::publish(Message::Viewport(ViewportMessage::OrbitGestureChanged(
                orbit,
            )))
            .and_capture(),
        )
    }
}

impl canvas::Program<Message> for SpatialViewportController {
    type State = SpatialViewportControllerState;

    fn update(
        &self,
        state: &mut Self::State,
        event: &canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        self.reset_if_obsolete(state, bounds, cursor);
        match event {
            canvas::Event::Mouse(mouse::Event::WheelScrolled { .. }) => {
                // Spatial scenes have no zoom. Let ordinary wheel scrolling
                // continue to the surrounding page/scrollable.
                (state.mouse_drag.is_some() || state.touch_captured).then(canvas::Action::capture)
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if !state.touches.is_empty() {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let position = cursor.position_in(bounds)?;
                let position = ScreenPoint::new(f64::from(position.x), f64::from(position.y));
                state.mouse_drag = Some(SpatialDragGesture {
                    baseline: self.orbit,
                    initial_position: position,
                });
                state.working_orbit = Some(self.orbit);
                Some(
                    canvas::Action::publish(Message::Viewport(
                        ViewportMessage::OrbitGestureStarted,
                    ))
                    .and_capture(),
                )
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                if !state.touches.is_empty() {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let gesture = state.mouse_drag?;
                let orbit = gesture.baseline.arcball_drag(
                    gesture.initial_position,
                    Self::local_position(*position, bounds),
                    Self::viewport(bounds),
                );
                state.working_orbit = Some(orbit);
                Some(
                    canvas::Action::publish(Message::Viewport(
                        ViewportMessage::OrbitGestureChanged(orbit),
                    ))
                    .and_capture(),
                )
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                if !state.touches.is_empty() {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let gesture = state.mouse_drag.take()?;
                let orbit = cursor.position_from(bounds.position()).map_or_else(
                    || state.working_orbit.unwrap_or(self.orbit),
                    |position| {
                        gesture.baseline.arcball_drag(
                            gesture.initial_position,
                            ScreenPoint::new(f64::from(position.x), f64::from(position.y)),
                            Self::viewport(bounds),
                        )
                    },
                );
                state.working_orbit = None;
                Some(
                    canvas::Action::publish(Message::Viewport(ViewportMessage::OrbitGestureEnded(
                        orbit,
                    )))
                    .and_capture(),
                )
            }
            canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }) => {
                if state.touches.iter().any(|touch| touch.id == *id) {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                if state.touch_captured {
                    state.touches.push(ActiveTouch {
                        id: *id,
                        position: *position,
                    });
                    self.rebase_touch(state, bounds);
                    return Some(canvas::Action::capture());
                }
                if state.mouse_drag.is_some() {
                    return Some(canvas::Action::capture());
                }
                if !bounds.contains(*position) {
                    return None;
                }
                state.touches.push(ActiveTouch {
                    id: *id,
                    position: *position,
                });
                state.touch_captured = true;
                state.working_orbit = Some(self.orbit);
                self.rebase_touch(state, bounds);
                Some(
                    canvas::Action::publish(Message::Viewport(
                        ViewportMessage::OrbitGestureStarted,
                    ))
                    .and_capture(),
                )
            }
            canvas::Event::Touch(iced::touch::Event::FingerMoved { id, position }) => {
                if !Self::update_touch_position(state, *id, *position) {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                if !state.touch_captured {
                    return None;
                }
                let orbit = self.touch_orbit(state, bounds)?;
                state.working_orbit = Some(orbit);
                Some(
                    canvas::Action::publish(Message::Viewport(
                        ViewportMessage::OrbitGestureChanged(orbit),
                    ))
                    .and_capture(),
                )
            }
            canvas::Event::Touch(iced::touch::Event::FingerLifted { id, position }) => {
                if !Self::update_touch_position(state, *id, *position) {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let orbit = if state.touch_captured {
                    self.touch_orbit(state, bounds)
                        .or(state.working_orbit)
                        .unwrap_or(self.orbit)
                } else {
                    self.orbit
                };
                self.finish_touch(state, *id, orbit, bounds)
            }
            canvas::Event::Touch(iced::touch::Event::FingerLost { id, .. }) => {
                if !state.touches.iter().any(|touch| touch.id == *id) {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let orbit = state.working_orbit.unwrap_or(self.orbit);
                self.finish_touch(state, *id, orbit, bounds)
            }
            canvas::Event::Window(iced::window::Event::Unfocused) => {
                let captured = state.mouse_drag.is_some() || state.touch_captured;
                let orbit = state.working_orbit.unwrap_or(self.orbit);
                let epoch = state.epoch;
                *state = SpatialViewportControllerState {
                    epoch,
                    ..SpatialViewportControllerState::default()
                };
                captured.then(|| {
                    canvas::Action::publish(Message::Viewport(ViewportMessage::OrbitGestureEnded(
                        orbit,
                    )))
                    .and_capture()
                })
            }
            _ => {
                if state.touch_captured {
                    match event {
                        canvas::Event::Mouse(_) | canvas::Event::Touch(_) => {
                            Some(canvas::Action::capture())
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        _renderer: &IcedRenderer,
        _theme: &Theme,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        Vec::new()
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.mouse_drag.is_some() {
            mouse::Interaction::Grabbing
        } else if state.touches.is_empty() && cursor.is_over(bounds) {
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::None
        }
    }
}

struct ViewportController {
    camera: Camera2d,
    line_bounds: RenderBounds,
    mobile: bool,
    enabled: bool,
    epoch: u64,
}

#[derive(Debug, Default)]
struct ViewportControllerState {
    epoch: u64,
    last_bounds: Option<Rectangle>,
    mouse_drag: Option<MouseViewGesture>,
    touches: Vec<ActiveTouch>,
    touch_gesture: Option<TouchViewGesture>,
    touch_captured: bool,
    working_camera: Option<Camera2d>,
}

#[derive(Debug, Clone, Copy)]
struct MouseViewGesture {
    baseline: Camera2d,
    initial_position: ScreenPoint,
    last_position: ScreenPoint,
}

#[derive(Debug, Clone, Copy)]
struct ActiveTouch {
    id: iced::touch::Finger,
    position: Point,
}

#[derive(Debug, Clone, Copy)]
struct TouchViewGesture {
    baseline: Camera2d,
    initial_centroid: ScreenPoint,
    initial_distance: f64,
}

impl ViewportController {
    fn viewport(bounds: Rectangle) -> ViewportSize {
        ViewportSize::new(f64::from(bounds.width), f64::from(bounds.height))
    }

    fn local_position(position: Point, bounds: Rectangle) -> ScreenPoint {
        ScreenPoint::new(
            f64::from(position.x - bounds.x),
            f64::from(position.y - bounds.y),
        )
    }

    fn update_touch_position(
        state: &mut ViewportControllerState,
        id: iced::touch::Finger,
        position: Point,
    ) -> bool {
        let Some(touch) = state.touches.iter_mut().find(|touch| touch.id == id) else {
            return false;
        };
        touch.position = position;
        true
    }

    fn touch_measurement(
        state: &ViewportControllerState,
        bounds: Rectangle,
    ) -> Option<(ScreenPoint, f64)> {
        let first = state.touches.first()?;
        let first = Self::local_position(first.position, bounds);
        let Some(second) = state.touches.get(1) else {
            return Some((first, 0.0));
        };
        let second = Self::local_position(second.position, bounds);
        let centroid = ScreenPoint::new((first.x + second.x) * 0.5, (first.y + second.y) * 0.5);
        let distance = (second.x - first.x).hypot(second.y - first.y);
        Some((centroid, distance))
    }

    fn rebase_touch(&self, state: &mut ViewportControllerState, bounds: Rectangle) {
        let Some((centroid, distance)) = Self::touch_measurement(state, bounds) else {
            state.touch_gesture = None;
            return;
        };
        let baseline = state.working_camera.unwrap_or(self.camera);
        state.working_camera = Some(baseline);
        state.touch_gesture = Some(TouchViewGesture {
            baseline,
            initial_centroid: centroid,
            initial_distance: distance,
        });
    }

    fn touch_camera(&self, state: &ViewportControllerState, bounds: Rectangle) -> Option<Camera2d> {
        let gesture = state.touch_gesture?;
        let (centroid, distance) = Self::touch_measurement(state, bounds)?;
        let scale_factor = if gesture.initial_distance > f64::EPSILON && distance.is_finite() {
            distance / gesture.initial_distance
        } else {
            1.0
        };
        Some(gesture.baseline.pinch(
            gesture.initial_centroid,
            centroid,
            scale_factor,
            self.line_bounds.into(),
            Self::viewport(bounds),
        ))
    }

    fn finish_touch(
        &self,
        state: &mut ViewportControllerState,
        id: iced::touch::Finger,
        camera: Camera2d,
        bounds: Rectangle,
    ) -> Option<canvas::Action<Message>> {
        state.touches.retain(|touch| touch.id != id);
        if !state.touch_captured {
            state.working_camera = None;
            state.touch_gesture = None;
            return None;
        }
        if state.touches.is_empty() {
            state.touch_captured = false;
            state.touch_gesture = None;
            state.working_camera = None;
            return Some(
                canvas::Action::publish(Message::Viewport(ViewportMessage::GestureEnded(camera)))
                    .and_capture(),
            );
        }
        state.working_camera = Some(camera);
        self.rebase_touch(state, bounds);
        Some(
            canvas::Action::publish(Message::Viewport(ViewportMessage::GestureChanged(camera)))
                .and_capture(),
        )
    }

    fn update_disabled_capture(
        &self,
        state: &mut ViewportControllerState,
        event: &canvas::Event,
    ) -> Option<canvas::Action<Message>> {
        if state.touch_captured {
            match event {
                canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }) => {
                    if !state.touches.iter().any(|touch| touch.id == *id) {
                        state.touches.push(ActiveTouch {
                            id: *id,
                            position: *position,
                        });
                    }
                    return Some(canvas::Action::capture());
                }
                canvas::Event::Touch(iced::touch::Event::FingerMoved { id, position }) => {
                    let _updated = Self::update_touch_position(state, *id, *position);
                    return Some(canvas::Action::capture());
                }
                canvas::Event::Touch(
                    iced::touch::Event::FingerLifted { id, .. }
                    | iced::touch::Event::FingerLost { id, .. },
                ) => {
                    state.touches.retain(|touch| touch.id != *id);
                    if state.touches.is_empty() {
                        state.touch_captured = false;
                        state.touch_gesture = None;
                        state.working_camera = None;
                    }
                    return Some(canvas::Action::capture());
                }
                canvas::Event::Window(iced::window::Event::Unfocused) => {
                    let epoch = state.epoch;
                    let last_bounds = state.last_bounds;
                    *state = ViewportControllerState {
                        epoch,
                        last_bounds,
                        ..ViewportControllerState::default()
                    };
                    return Some(canvas::Action::capture());
                }
                canvas::Event::Mouse(_) => {
                    return Some(canvas::Action::capture());
                }
                _ => return None,
            }
        }

        if state.mouse_drag.is_some() {
            match event {
                canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                    state.mouse_drag = None;
                    state.working_camera = None;
                    return Some(canvas::Action::capture());
                }
                canvas::Event::Window(iced::window::Event::Unfocused) => {
                    state.mouse_drag = None;
                    state.working_camera = None;
                    return Some(canvas::Action::capture());
                }
                canvas::Event::Mouse(_) | canvas::Event::Touch(_) => {
                    return Some(canvas::Action::capture());
                }
                _ => return None,
            }
        }

        // An uncaptured first mobile finger still belongs to the surrounding
        // Scrollable. Track it only so its eventual release leaves clean state.
        match event {
            canvas::Event::Touch(iced::touch::Event::FingerMoved { id, position }) => {
                let _updated = Self::update_touch_position(state, *id, *position);
            }
            canvas::Event::Touch(
                iced::touch::Event::FingerLifted { id, .. }
                | iced::touch::Event::FingerLost { id, .. },
            ) => state.touches.retain(|touch| touch.id != *id),
            canvas::Event::Window(iced::window::Event::Unfocused) => state.touches.clear(),
            _ => {}
        }
        None
    }

    fn reset_if_obsolete(
        &self,
        state: &mut ViewportControllerState,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) {
        let previous_bounds = state.last_bounds.replace(bounds);
        let bounds_changed = previous_bounds.is_some_and(|previous| previous != bounds);
        if state.epoch != self.epoch || bounds_changed {
            state.epoch = self.epoch;

            // The app may replace a scene or apply a view button while a
            // pointer is still down. Keep ownership of active pointers so a
            // subsequent move/lift cannot bubble into the mobile Scrollable,
            // but rebase against the new canonical camera to avoid a jump.
            if let Some(mouse_drag) = state.mouse_drag.as_mut() {
                let position = cursor
                    .position_from(bounds.position())
                    .map(|position| ScreenPoint::new(f64::from(position.x), f64::from(position.y)))
                    .unwrap_or_else(|| {
                        previous_bounds.map_or(mouse_drag.last_position, |previous| {
                            ScreenPoint::new(
                                mouse_drag.last_position.x + f64::from(previous.x - bounds.x),
                                mouse_drag.last_position.y + f64::from(previous.y - bounds.y),
                            )
                        })
                    });
                *mouse_drag = MouseViewGesture {
                    baseline: self.camera,
                    initial_position: position,
                    last_position: position,
                };
                state.working_camera = Some(self.camera);
            } else if state.touch_captured {
                state.working_camera = Some(self.camera);
                self.rebase_touch(state, bounds);
            } else {
                // Preserve an uncaptured first mobile finger so a second finger
                // can still promote it into a canvas gesture after the update.
                state.touch_gesture = None;
                state.working_camera = None;
            }
        }
    }
}

impl canvas::Program<Message> for ViewportController {
    type State = ViewportControllerState;

    fn update(
        &self,
        state: &mut Self::State,
        event: &canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        self.reset_if_obsolete(state, bounds, cursor);

        if !self.enabled {
            return self.update_disabled_capture(state, event);
        }

        match event {
            canvas::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                if !state.touches.is_empty() || state.mouse_drag.is_some() {
                    return (state.touch_captured || state.mouse_drag.is_some())
                        .then(canvas::Action::capture);
                }
                let anchor = cursor.position_in(bounds)?;
                let amount = match *delta {
                    mouse::ScrollDelta::Lines { y, .. } => f64::from(y),
                    mouse::ScrollDelta::Pixels { y, .. } => {
                        f64::from(y) / VIEW_WHEEL_PIXELS_PER_LINE
                    }
                };
                if amount == 0.0 || !amount.is_finite() {
                    return None;
                }
                let factor = VIEW_WHEEL_LINES_FACTOR.powf(amount);
                Some(
                    canvas::Action::publish(Message::Viewport(ViewportMessage::WheelZoom {
                        factor,
                        anchor: ScreenPoint::new(f64::from(anchor.x), f64::from(anchor.y)),
                        viewport: Self::viewport(bounds),
                    }))
                    .and_capture(),
                )
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if !state.touches.is_empty() {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let position = cursor.position_in(bounds)?;
                state.working_camera = Some(self.camera);
                state.mouse_drag = Some(MouseViewGesture {
                    baseline: self.camera,
                    initial_position: ScreenPoint::new(
                        f64::from(position.x),
                        f64::from(position.y),
                    ),
                    last_position: ScreenPoint::new(f64::from(position.x), f64::from(position.y)),
                });
                Some(
                    canvas::Action::publish(Message::Viewport(ViewportMessage::GestureStarted))
                        .and_capture(),
                )
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                if !state.touches.is_empty() {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let gesture = state.mouse_drag.as_mut()?;
                let position = Self::local_position(*position, bounds);
                let camera = gesture.baseline.pan_by_pixels(
                    ScreenPoint::new(
                        position.x - gesture.initial_position.x,
                        position.y - gesture.initial_position.y,
                    ),
                    self.line_bounds.into(),
                    Self::viewport(bounds),
                );
                gesture.last_position = position;
                state.working_camera = Some(camera);
                Some(
                    canvas::Action::publish(Message::Viewport(ViewportMessage::GestureChanged(
                        camera,
                    )))
                    .and_capture(),
                )
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                if !state.touches.is_empty() {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                let gesture = state.mouse_drag.take()?;
                let camera = cursor.position_from(bounds.position()).map_or_else(
                    || state.working_camera.unwrap_or(self.camera),
                    |position| {
                        gesture.baseline.pan_by_pixels(
                            ScreenPoint::new(
                                f64::from(position.x) - gesture.initial_position.x,
                                f64::from(position.y) - gesture.initial_position.y,
                            ),
                            self.line_bounds.into(),
                            Self::viewport(bounds),
                        )
                    },
                );
                state.working_camera = None;
                Some(
                    canvas::Action::publish(Message::Viewport(ViewportMessage::GestureEnded(
                        camera,
                    )))
                    .and_capture(),
                )
            }
            canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }) => {
                if state.touches.iter().any(|touch| touch.id == *id) {
                    return state.touch_captured.then(canvas::Action::capture);
                }
                if state.touch_captured {
                    state.touches.push(ActiveTouch {
                        id: *id,
                        position: *position,
                    });
                    self.rebase_touch(state, bounds);
                    return Some(canvas::Action::capture());
                }
                if state.mouse_drag.is_some() {
                    return Some(canvas::Action::capture());
                }
                if !bounds.contains(*position) {
                    return None;
                }
                state.touches.push(ActiveTouch {
                    id: *id,
                    position: *position,
                });
                if !state.touch_captured && self.mobile && state.touches.len() == 1 {
                    return None;
                }
                if !state.touch_captured {
                    state.touch_captured = true;
                    state.working_camera = Some(self.camera);
                    self.rebase_touch(state, bounds);
                    return Some(
                        canvas::Action::publish(Message::Viewport(ViewportMessage::GestureStarted))
                            .and_capture(),
                    );
                }
                self.rebase_touch(state, bounds);
                Some(canvas::Action::capture())
            }
            canvas::Event::Touch(iced::touch::Event::FingerMoved { id, position }) => {
                if !Self::update_touch_position(state, *id, *position) {
                    return (state.touch_captured || state.mouse_drag.is_some())
                        .then(canvas::Action::capture);
                }
                if !state.touch_captured {
                    return None;
                }
                let camera = self.touch_camera(state, bounds)?;
                state.working_camera = Some(camera);
                Some(
                    canvas::Action::publish(Message::Viewport(ViewportMessage::GestureChanged(
                        camera,
                    )))
                    .and_capture(),
                )
            }
            canvas::Event::Touch(iced::touch::Event::FingerLifted { id, position }) => {
                if !Self::update_touch_position(state, *id, *position) {
                    return (state.touch_captured || state.mouse_drag.is_some())
                        .then(canvas::Action::capture);
                }
                let camera = if state.touch_captured {
                    self.touch_camera(state, bounds)
                        .or(state.working_camera)
                        .unwrap_or(self.camera)
                } else {
                    self.camera
                };
                self.finish_touch(state, *id, camera, bounds)
            }
            canvas::Event::Touch(iced::touch::Event::FingerLost { id, .. }) => {
                if !state.touches.iter().any(|touch| touch.id == *id) {
                    return (state.touch_captured || state.mouse_drag.is_some())
                        .then(canvas::Action::capture);
                }
                // A lost finger's reported coordinate is not guaranteed to be
                // meaningful. Keep the last successfully applied camera and
                // simply rebase any fingers that remain.
                let camera = state.working_camera.unwrap_or(self.camera);
                self.finish_touch(state, *id, camera, bounds)
            }
            canvas::Event::Window(iced::window::Event::Unfocused) => {
                let captured = state.mouse_drag.is_some() || state.touch_captured;
                let camera = state.working_camera.unwrap_or(self.camera);
                let epoch = state.epoch;
                *state = ViewportControllerState {
                    epoch,
                    ..ViewportControllerState::default()
                };
                captured.then(|| {
                    canvas::Action::publish(Message::Viewport(ViewportMessage::GestureEnded(
                        camera,
                    )))
                    .and_capture()
                })
            }
            _ => {
                if state.touch_captured {
                    match event {
                        canvas::Event::Mouse(_) | canvas::Event::Touch(_) => {
                            Some(canvas::Action::capture())
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        _renderer: &IcedRenderer,
        _theme: &Theme,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        Vec::new()
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.mouse_drag.is_some() {
            mouse::Interaction::Grabbing
        } else if self.enabled && state.touches.is_empty() && cursor.is_over(bounds) {
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::None
        }
    }
}

struct SpatialCanvas<'a> {
    scene: &'a SpatialScene,
    orbit: Orbit3d,
    dark: bool,
    line_width_scale: f32,
    cache: &'a Cache,
}

impl<Message> canvas::Program<Message> for SpatialCanvas<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &IcedRenderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            let background = Path::rectangle(Point::ORIGIN, bounds.size());
            frame.fill(
                &background,
                self.scene
                    .background()
                    .map(|[red, green, blue]| Color::from_rgb8(red, green, blue))
                    .unwrap_or(if self.dark {
                        DARK_CANVAS_BACKGROUND
                    } else {
                        CANVAS_BACKGROUND
                    }),
            );
            let Ok(primitives) = self.scene.software_preview(
                bounds.size(),
                self.orbit,
                self.dark,
                self.line_width_scale,
            ) else {
                return;
            };
            for primitive in primitives {
                match primitive {
                    ProjectedPrimitive3d::Line {
                        start,
                        end,
                        width,
                        color,
                        ..
                    } => {
                        frame.stroke(
                            &Path::line(start, end),
                            Stroke::default()
                                .with_color(color)
                                .with_width(width)
                                .with_line_cap(LineCap::Round)
                                .with_line_join(LineJoin::Round),
                        );
                    }
                    ProjectedPrimitive3d::Polygon {
                        vertices, color, ..
                    } => {
                        let Some(first) = vertices.first().copied() else {
                            continue;
                        };
                        let path = Path::new(|builder| {
                            builder.move_to(first);
                            for point in vertices.into_iter().skip(1) {
                                builder.line_to(point);
                            }
                            builder.close();
                        });
                        frame.fill(&path, color);
                    }
                }
            }
        });

        #[cfg(feature = "desktop")]
        if matches!(renderer, IcedRenderer::Secondary(_)) {
            self.scene.mark_software_preview_complete();
        }

        vec![canvas_clip::layer_clipped(geometry)]
    }
}

struct LsystemCanvas<'a> {
    scene: Option<&'a Scene2d>,
    transition: Option<(&'a TransitionScene, f32)>,
    line_bounds: RenderBounds,
    exact_total_line_length: Option<f64>,
    line_scene: Option<&'a LineScene>,
    camera: Camera2d,
    dark: bool,
    line_width_scale: f32,
    cache: &'a Cache,
}

impl<Message> canvas::Program<Message> for LsystemCanvas<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        visualizer: &IcedRenderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geometry = self.cache.draw(visualizer, bounds.size(), |frame| {
            let background = Path::rectangle(Point::ORIGIN, bounds.size());
            frame.fill(
                &background,
                self.transition
                    .and_then(|(transition, _)| transition.background())
                    .or_else(|| self.scene.and_then(|scene| scene.background))
                    .map(|[red, green, blue]| Color::from_rgb8(red, green, blue))
                    .unwrap_or_else(|| {
                        if self.dark {
                            DARK_CANVAS_BACKGROUND
                        } else {
                            CANVAS_BACKGROUND
                        }
                    }),
            );

            if let Some((transition, progress)) = self.transition {
                for line in transition.software_preview(
                    SOFTWARE_FALLBACK_PREVIEW_LINES,
                    bounds.size(),
                    self.camera,
                    self.dark,
                    progress,
                    self.line_width_scale,
                ) {
                    if line.color[3] <= 0.0 || line.width <= 0.0 {
                        continue;
                    }
                    let drawing = Path::line(
                        Point::new(line.start[0], line.start[1]),
                        Point::new(line.end[0], line.end[1]),
                    );
                    frame.stroke(
                        &drawing,
                        Stroke::default()
                            .with_color(Color::from_rgba(
                                line.color[0],
                                line.color[1],
                                line.color[2],
                                line.color[3],
                            ))
                            .with_width(line.width)
                            .with_line_cap(LineCap::Round)
                            .with_line_join(LineJoin::Round),
                    );
                }
                return;
            }

            let lines = self
                .scene
                .into_iter()
                .flat_map(|scene| &scene.primitives)
                .filter_map(|primitive| match primitive {
                    Primitive2d::Line(line) => Some((
                        RenderLine {
                            start: (line.line.0.0 as f32, line.line.0.1 as f32),
                            end: (line.line.1.0 as f32, line.line.1.1 as f32),
                            width: line.width as f32,
                        },
                        line.color,
                    )),
                    Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
                })
                .collect::<Vec<_>>();
            let polygon_items = self
                .scene
                .into_iter()
                .flat_map(|scene| &scene.primitives)
                .filter_map(|primitive| match primitive {
                    Primitive2d::Polygon(polygon) => Some(polygon),
                    Primitive2d::Line(_) | Primitive2d::Text(_) => None,
                })
                .collect::<Vec<_>>();
            let text_items = self
                .scene
                .into_iter()
                .flat_map(|scene| &scene.primitives)
                .filter_map(|primitive| match primitive {
                    Primitive2d::Text(text) => Some(text),
                    Primitive2d::Line(_) | Primitive2d::Polygon(_) => None,
                })
                .collect::<Vec<_>>();

            if lines.is_empty() && polygon_items.is_empty() && text_items.is_empty() {
                let empty = Path::rectangle(
                    Point::new(bounds.width * 0.15, bounds.height * 0.15),
                    Size::new(bounds.width * 0.7, bounds.height * 0.7),
                );
                frame.stroke(
                    &empty,
                    Stroke::default()
                        .with_color(Color::from_rgba8(110, 120, 132, 0.35))
                        .with_width(1.0),
                );
                return;
            }

            let transform = ViewTransform::new(
                self.line_bounds.into(),
                ViewportSize::new(f64::from(bounds.width), f64::from(bounds.height)),
                self.camera,
            );

            for polygon in polygon_items {
                let Some(first) = polygon.vertices.first() else {
                    continue;
                };
                let first = transform.project(WorldPoint::new(first.0, first.1));
                let drawing = Path::new(|builder| {
                    builder.move_to(Point::new(first.x as f32, first.y as f32));
                    for point in polygon.vertices.iter().skip(1) {
                        let point = transform.project(WorldPoint::new(point.0, point.1));
                        builder.line_to(Point::new(point.x as f32, point.y as f32));
                    }
                    builder.close();
                });
                let color = turtle_stroke_rgb(
                    polygon.color,
                    if self.dark {
                        Palette::Dark
                    } else {
                        Palette::Light
                    },
                )
                .map(|rgb| Color::from_rgb8(rgb[0], rgb[1], rgb[2]))
                .unwrap_or_else(|| palette_color(COLOR_BUCKETS / 2, self.dark));
                frame.fill(&drawing, color);
            }

            let mut width_estimator = StrokeWidthEstimator::default();
            for (line, _) in &lines {
                width_estimator.observe(braken_viz::Line2d(
                    (f64::from(line.start.0), f64::from(line.start.1)),
                    (f64::from(line.end.0), f64::from(line.end.1)),
                ));
            }
            let width = canvas_stroke_width(
                self.exact_total_line_length
                    .or_else(|| width_estimator.total_line_length()),
                self.line_scene
                    .map_or_else(|| width_estimator.line_count(), LineScene::line_count),
                render_bounds_extent(self.line_bounds),
                transform.fit_scale(),
                self.camera.zoom,
                self.line_width_scale,
            );
            let mut buckets = vec![Vec::new(); COLOR_BUCKETS];
            for (line, color) in &lines {
                let world_start = WorldPoint::new(f64::from(line.start.0), f64::from(line.start.1));
                let world_end = WorldPoint::new(f64::from(line.end.0), f64::from(line.end.1));
                let start = transform.project(world_start);
                let end = transform.project(world_end);
                if *color != StrokeColor::ThemeDefault {
                    let Some(effective_width) = canvas_effective_line_width(width, line.width)
                    else {
                        continue;
                    };
                    let rgb = turtle_stroke_rgb(
                        *color,
                        if self.dark {
                            Palette::Dark
                        } else {
                            Palette::Light
                        },
                    )
                    .expect("non-default turtle colors resolve to RGB");
                    let drawing = Path::line(
                        Point::new(start.x as f32, start.y as f32),
                        Point::new(end.x as f32, end.y as f32),
                    );
                    frame.stroke(
                        &drawing,
                        Stroke::default()
                            .with_color(Color::from_rgb8(rgb[0], rgb[1], rgb[2]))
                            .with_width(effective_width)
                            .with_line_cap(LineCap::Round)
                            .with_line_join(LineJoin::Round),
                    );
                    continue;
                }
                let position = transform.world_palette_position(world_start, world_end);
                let bucket = theme_default_bucket(position);
                buckets[bucket].push((
                    Point::new(start.x as f32, start.y as f32),
                    Point::new(end.x as f32, end.y as f32),
                    line.width,
                ));
            }

            for (index, mut segments) in buckets.into_iter().enumerate() {
                if segments.is_empty() {
                    continue;
                }
                segments.sort_by(|left, right| left.2.total_cmp(&right.2));
                for group in segments.chunk_by(|left, right| left.2 == right.2) {
                    let Some(effective_width) = canvas_effective_line_width(width, group[0].2)
                    else {
                        continue;
                    };
                    let drawing = Path::new(|builder| {
                        for (start, end, _) in group {
                            builder.move_to(*start);
                            builder.line_to(*end);
                        }
                    });
                    frame.stroke(
                        &drawing,
                        Stroke::default()
                            .with_color(palette_color(index, self.dark))
                            .with_width(effective_width)
                            .with_line_cap(LineCap::Round)
                            .with_line_join(LineJoin::Round),
                    );
                }
            }

            for item in text_items {
                let color = match item.role {
                    TextRole::Muted => Color::from_rgb8(125, 137, 154),
                    TextRole::Error => Color::from_rgb8(220, 80, 80),
                    _ if self.dark => Color::from_rgb8(238, 242, 248),
                    _ => Color::from_rgb8(38, 50, 71),
                };
                frame.fill_text(canvas::Text {
                    content: item.content.clone(),
                    position: Point::new(item.position.0 as f32, item.position.1 as f32),
                    color,
                    size: iced::Pixels(item.size as f32),
                    ..canvas::Text::default()
                });
            }
        });

        #[cfg(feature = "desktop")]
        if matches!(visualizer, IcedRenderer::Secondary(_))
            && let Some(line_scene) = self.line_scene
        {
            line_scene.mark_software_preview_complete();
        }

        vec![canvas_clip::layer_clipped(geometry)]
    }
}

fn canvas_stroke_width(
    total_line_length: Option<f64>,
    line_count: usize,
    drawing_extent: (f64, f64),
    fit_scale: f64,
    zoom: f64,
    line_width_scale: f32,
) -> f32 {
    (adaptive_scene_stroke_width(total_line_length, line_count, drawing_extent, fit_scale) * zoom)
        as f32
        * line_width_scale
}

fn canvas_effective_line_width(base_width: f32, source_width: f32) -> Option<f32> {
    (source_width > 0.0).then_some(base_width * source_width)
}

fn palette_color(index: usize, dark: bool) -> Color {
    let [red, green, blue] =
        theme_default_color(index, if dark { Palette::Dark } else { Palette::Light });
    Color::from_rgb(red, green, blue)
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use iced::widget::canvas::Program as _;

    #[test]
    fn preset_preview_depth_cue_is_specific_to_three_dimensions() {
        for theme in [Theme::TokyoNightLight, Theme::TokyoNightStorm] {
            let planar = preview_style(&theme, false, false);
            assert!(matches!(planar.background, Some(Background::Color(_))));
            assert_eq!(planar.shadow, Shadow::default());

            let spatial = preview_style(&theme, false, true);
            assert!(matches!(spatial.background, Some(Background::Gradient(_))));
            assert!(spatial.shadow.blur_radius > 0.0);
            assert_ne!(
                dimension_badge_style(&theme, false),
                dimension_badge_style(&theme, true)
            );

            let selected_planar = preview_style(&theme, true, false);
            let selected_spatial = preview_style(&theme, true, true);
            assert_eq!(selected_planar.border, selected_spatial.border);
        }
    }

    #[test]
    fn filled_planar_scenes_support_navigation_without_moving_inspector_text() {
        let polygon = Primitive2d::Polygon(Polygon2d {
            vertices: vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            color: StrokeColor::ThemeDefault,
        });
        assert!(scene_supports_planar_navigation(&Scene2d {
            primitives: vec![polygon.clone()],
            background: None,
        }));
        assert!(!scene_supports_planar_navigation(&Scene2d::default()));
        assert!(!scene_supports_planar_navigation(&Scene2d {
            primitives: vec![
                polygon,
                Primitive2d::Text(braken_viz::Text2d {
                    position: (0.0, 0.0),
                    content: String::from("fixed"),
                    size: 12.0,
                    role: TextRole::Body,
                }),
            ],
            background: None,
        }));
    }

    #[test]
    fn every_preset_has_generated_svg_previews_for_both_themes() {
        let presets = presets::get_presets();
        let names = presets
            .iter()
            .map(|preset| preset.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, generated_preset_previews::PRESET_PREVIEW_NAMES);
        for preset in presets {
            for palette in [Palette::Light, Palette::Dark] {
                let svg = generated_preset_previews::preset_preview_svg(&preset.name, palette)
                    .unwrap_or_else(|| {
                        panic!("{} is missing its {palette:?} preview", preset.name)
                    });
                assert!(svg.starts_with(b"<svg "), "{} has invalid SVG", preset.name);
            }
        }
    }

    #[test]
    fn preset_catalog_starts_with_every_model_visible() {
        let (app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();

        assert!(app.preset_search.is_empty());
        assert_eq!(app.matching_preset_indices().len(), app.presets.len());
    }

    #[test]
    fn preset_search_is_case_insensitive_multi_term_and_sorted() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let _ = app.update(Message::PresetSearchChanged(String::from(
            "  DrAgOn   CuRvE ",
        )));

        let matches = app.matching_preset_indices();
        assert!(!matches.is_empty());
        assert!(matches.iter().all(|index| {
            app.presets[*index]
                .matches_search_terms(&[String::from("dragon"), String::from("curve")])
        }));
        let names = matches
            .iter()
            .map(|index| app.presets[*index].name.to_lowercase())
            .collect::<Vec<_>>();
        assert!(names.windows(2).all(|pair| pair[0] <= pair[1]));

        let _ = app.update(Message::ClearPresetSearch);
        assert!(app.preset_search.is_empty());
        assert_eq!(app.matching_preset_indices().len(), app.presets.len());

        let _ = app.update(Message::PresetSearchChanged(String::from(
            "bracketed deterministic",
        )));
        assert!(
            !app.matching_preset_indices().is_empty(),
            "imported feature metadata should be searchable"
        );

        let _ = app.update(Message::PresetSearchChanged(String::from("   ")));
        assert_eq!(app.matching_preset_indices().len(), app.presets.len());
    }

    #[test]
    fn filtering_out_the_selected_model_does_not_change_the_active_preset() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let index = app
            .presets
            .iter()
            .position(|preset| preset.name == "Sierpiński Triangle")
            .expect("curated Sierpiński alternative should be in the catalog");

        let _ = app.update(Message::PresetSelected(PresetChoice(index)));
        app.render_coordinator.cancel_current();
        let selected_source = app.source_editor.text();
        let _ = app.update(Message::PresetSearchChanged(String::from(
            "definitely-no-matching-model",
        )));
        assert!(app.matching_preset_indices().is_empty());
        assert_eq!(app.selected_preset, Some(PresetChoice(index)));
        assert_eq!(app.source_editor.text(), selected_source);
    }

    fn cache_key(source: &str, iterations: usize) -> GenerationCacheKey {
        GenerationCacheKey {
            source: Arc::from(source),
            ir_json: None,
            iterations,
            seed: 0,
            semantics: DerivationSemantics::default(),
        }
    }

    fn generation(name: &str) -> Arc<Generation> {
        use braken::{GenerationItem, Module};

        Arc::new(Generation(vec![GenerationItem::Module(Module::new(
            name,
            Vec::new(),
        ))]))
    }

    fn derivation(backend: &'static str) -> DerivationInfo {
        DerivationInfo {
            backend,
            elapsed: Some(Duration::from_millis(7)),
            cached: false,
        }
    }

    fn test_identity() -> RenderIdentity {
        RenderIdentity {
            source: Arc::from("axiom F;"),
            ir_json: None,
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 0,
            semantics: DerivationSemantics::default(),
        }
    }

    fn test_ir_tooling() -> IrToolingSnapshot {
        IrToolingSnapshot {
            disassembly: Arc::from("test IR"),
            json: None,
        }
    }

    fn ready_result(request_id: u64, view_epoch: u64) -> RenderResult {
        RenderResult {
            request_id,
            view_epoch,
            scene: Some(Arc::new(Scene2d::default())),
            line_scene: None,
            spatial_scene: None,
            element_count: 0,
            refinement_started_at: RenderInstant::now(),
            elapsed_ms: 1,
            backend: String::from("CPU"),
            bounds: RenderBounds::default(),
            cache_insert: None,
            iteration: 1,
            identity: test_identity(),
            ir_tooling: test_ir_tooling(),
        }
    }

    fn viewport_controller(mobile: bool) -> ViewportController {
        ViewportController {
            camera: Camera2d::fit(),
            line_bounds: RenderBounds {
                min_x: 0.0,
                max_x: 100.0,
                min_y: 0.0,
                max_y: 100.0,
            },
            mobile,
            enabled: true,
            epoch: 0,
        }
    }

    fn viewport_bounds() -> Rectangle {
        Rectangle::new(Point::new(10.0, 20.0), Size::new(200.0, 100.0))
    }

    fn test_line_scene(line_count: usize) -> Arc<LineScene> {
        let mut builder = LineSceneBuilder::default();
        builder
            .extend((0..line_count).map(|index| braken_viz::StyledLine2d {
                line: braken_viz::Line2d((index as f64, 0.0), (index as f64, 1.0)),
                width: 1.0,
                color: braken_viz::StrokeColor::ThemeDefault,
            }))
            .unwrap();
        builder.finish(RenderBounds {
            min_x: 0.0,
            max_x: line_count.saturating_sub(1) as f32,
            min_y: 0.0,
            max_y: 1.0,
        })
    }

    fn test_spatial_scene() -> Arc<SpatialScene> {
        SpatialScene::from_scene(Scene3d {
            primitives: vec![Primitive3d::Line(braken_viz::StyledLine3d {
                line: braken_viz::Line3d((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        })
        .unwrap()
    }

    fn spatial_ready_result(request_id: u64, view_epoch: u64) -> RenderResult {
        let mut result = ready_result(request_id, view_epoch);
        result.scene = None;
        result.spatial_scene = Some(test_spatial_scene());
        result.element_count = 1;
        result.identity.visualizer = VisualizerKind::Turtle3d;
        result
    }

    #[test]
    fn classifies_mobile_and_desktop_layouts() {
        assert_eq!(layout_mode(390.0), LayoutMode::Mobile);
        assert_eq!(layout_mode(699.0), LayoutMode::Mobile);
        assert_eq!(layout_mode(700.0), LayoutMode::Desktop);
        assert_eq!(layout_mode(1440.0), LayoutMode::Desktop);
    }

    #[test]
    fn accepted_spatial_preset_autorotates_until_the_first_owned_gesture() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        app.request_id = 41;
        app.pending_autorotate_request = Some(41);
        let _ = app.update(Message::Rendered(RenderOutcome::Ready(
            spatial_ready_result(41, 2),
        )));

        assert!(app.autorotate_3d);
        let initial = app.orbit_3d;
        let now = RenderInstant::now();
        app.autorotate_frame_at = Some(now - Duration::from_millis(50));
        let request_id = app.request_id;
        let _ = app.update(Message::Frame);
        assert_ne!(app.orbit_3d, initial);
        assert_eq!(app.request_id, request_id);

        let stopped = app.orbit_3d;
        app.pending_autorotate_request = Some(42);
        let _ = app.update(Message::Viewport(ViewportMessage::OrbitGestureStarted));
        assert!(!app.autorotate_3d);
        assert_eq!(app.pending_autorotate_request, None);
        let _ = app.update(Message::Frame);
        assert_eq!(app.orbit_3d, stopped);

        app.request_id = 42;
        let _ = app.update(Message::Rendered(RenderOutcome::Ready(
            spatial_ready_result(42, 2),
        )));
        assert_eq!(app.orbit_3d, stopped);
        assert!(!app.autorotate_3d);

        app.pending_autorotate_request = Some(43);
        let _ = app.update(Message::Viewport(ViewportMessage::GestureStarted));
        assert_eq!(app.pending_autorotate_request, None);
        app.pending_autorotate_request = Some(44);
        let _ = app.update(Message::Viewport(ViewportMessage::WheelZoom {
            factor: 1.1,
            anchor: ScreenPoint::new(50.0, 50.0),
            viewport: ViewportSize::new(100.0, 100.0),
        }));
        assert_eq!(app.pending_autorotate_request, None);
    }

    #[test]
    fn spatial_autorotate_respects_reduced_motion() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        app.reduced_motion = true;
        app.request_id = 51;
        app.pending_autorotate_request = Some(51);

        let _ = app.update(Message::Rendered(RenderOutcome::Ready(
            spatial_ready_result(51, 2),
        )));

        assert!(!app.autorotate_3d);
        assert_eq!(app.pending_autorotate_request, None);
    }

    #[test]
    fn spatial_gpu_failure_status_names_the_software_fallback() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        let mut displayed = spatial_ready_result(app.request_id, app.system_view_epoch);
        let spatial = displayed.spatial_scene.as_ref().unwrap();
        spatial
            .record_gpu_refinement_failure(SpatialSceneError::ResourceExhausted { requested: 17 });
        displayed.refinement_started_at = RenderInstant::now() - Duration::from_secs(1);
        app.render.displayed = Some(displayed);

        let (_, status) = app.render_scene_and_status();

        assert!(status.contains("Showing software 3D preview"));
        assert!(status.contains("17 spatial primitives"));
    }

    #[test]
    fn spatial_controller_ignores_wheel_and_rotates_with_mouse_drag() {
        let controller = SpatialViewportController {
            orbit: Orbit3d::canonical(),
            epoch: 1,
        };
        let bounds = viewport_bounds();
        let mut state = SpatialViewportControllerState::default();
        let wheel = controller.update(
            &mut state,
            &canvas::Event::Mouse(mouse::Event::WheelScrolled {
                delta: mouse::ScrollDelta::Lines { x: 0.0, y: 1.0 },
            }),
            bounds,
            mouse::Cursor::Available(Point::new(60.0, 70.0)),
        );
        assert!(wheel.is_none());

        let press = controller
            .update(
                &mut state,
                &canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
                bounds,
                mouse::Cursor::Available(Point::new(60.0, 70.0)),
            )
            .unwrap();
        assert!(matches!(
            press.into_inner().0,
            Some(Message::Viewport(ViewportMessage::OrbitGestureStarted))
        ));
        let moved = controller
            .update(
                &mut state,
                &canvas::Event::Mouse(mouse::Event::CursorMoved {
                    position: Point::new(100.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Available(Point::new(100.0, 70.0)),
            )
            .unwrap();
        let orbit = match moved.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::OrbitGestureChanged(orbit))) => orbit,
            other => panic!("unexpected spatial drag action: {other:?}"),
        };
        assert_ne!(orbit, Orbit3d::canonical());
    }

    #[test]
    fn native_spatial_visualization_streams_progress_and_honors_cancellation() {
        use braken::{GenerationItem, Module};

        let generation = Generation(vec![
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let job = RenderJob::default();
        let request = AppVisualizeRequest {
            request_id: 1,
            view_epoch: 1,
            key: cache_key("axiom F F;", 0),
            angle: 90.0,
            visualizer: VisualizerKind::Turtle3d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 0,
            cached_generation: None,
            transition: None,
            transition_source: None,
            job: job.clone(),
        };
        let display = visualize_generation(
            &generation,
            &request,
            derivation("CPU"),
            &mut Turtle2dStreamer::new(),
        )
        .unwrap();
        assert!(display.spatial_scene.is_some());
        assert_eq!(display.element_count, 2);
        let progress = job.progress.lock().unwrap().visualization.unwrap();
        assert_eq!(progress.lines_emitted, 2);

        let cancelled_job = RenderJob::default();
        cancelled_job.cancel();
        let mut cancelled_request = request;
        cancelled_request.job = cancelled_job;
        let Err(error) = visualize_generation(
            &generation,
            &cancelled_request,
            derivation("CPU"),
            &mut Turtle2dStreamer::new(),
        ) else {
            panic!("cancelled spatial visualization unexpectedly succeeded")
        };
        assert!(error.contains("cancelled"));
    }

    fn captured_spatial_touch(
        controller: &SpatialViewportController,
        state: &mut SpatialViewportControllerState,
        bounds: Rectangle,
        event: iced::touch::Event,
    ) -> Option<ViewportMessage> {
        let (message, _, status) = controller
            .update(
                state,
                &canvas::Event::Touch(event),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("owned spatial touch should be captured")
            .into_inner();
        assert_eq!(status, iced::event::Status::Captured);
        match message {
            Some(Message::Viewport(message)) => Some(message),
            None => None,
            other => panic!("unexpected spatial touch action: {other:?}"),
        }
    }

    fn changed_spatial_orbit(message: Option<ViewportMessage>) -> Orbit3d {
        match message {
            Some(ViewportMessage::OrbitGestureChanged(orbit)) => orbit,
            other => panic!("expected a spatial rotation update: {other:?}"),
        }
    }

    fn assert_same_spatial_orbit(actual: Orbit3d, expected: Orbit3d) {
        for (actual, expected) in actual.components().into_iter().zip(expected.components()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn spatial_first_touch_tap_stops_autorotation_without_derivation() {
        let controller = SpatialViewportController {
            orbit: Orbit3d::canonical(),
            epoch: 1,
        };
        let bounds = viewport_bounds();
        let mut state = SpatialViewportControllerState::default();
        let finger = iced::touch::Finger(1);
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        app.autorotate_3d = true;
        app.pending_autorotate_request = Some(app.request_id);
        let request_id = app.request_id;
        let cache_clock = app.generation_cache.clock;
        let camera = app.live_camera;

        let started = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerPressed {
                id: finger,
                position: Point::new(50.0, 70.0),
            },
        )
        .expect("first touch should start rotation ownership");
        assert!(matches!(started, ViewportMessage::OrbitGestureStarted));
        assert!(state.touch_captured);
        let _ = app.update(Message::Viewport(started));
        assert!(!app.autorotate_3d);
        assert_eq!(app.pending_autorotate_request, None);

        let ended = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerLifted {
                id: finger,
                position: Point::new(50.0, 70.0),
            },
        )
        .expect("lifting the tapping finger should finish ownership");
        match &ended {
            ViewportMessage::OrbitGestureEnded(orbit) => {
                assert_same_spatial_orbit(*orbit, controller.orbit);
            }
            other => panic!("unexpected tap release: {other:?}"),
        }
        let _ = app.update(Message::Viewport(ended));
        assert!(!app.view_gesture_active);
        assert!(!app.autorotate_3d);
        assert_eq!(app.request_id, request_id);
        assert_eq!(app.generation_cache.clock, cache_clock);
        assert_eq!(app.live_camera, camera);
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
    }

    #[test]
    fn spatial_single_touch_drag_applies_the_final_outside_lift_position() {
        let controller = SpatialViewportController {
            orbit: Orbit3d::canonical(),
            epoch: 1,
        };
        let bounds = viewport_bounds();
        let mut state = SpatialViewportControllerState::default();
        let finger = iced::touch::Finger(1);
        let _ = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerPressed {
                id: finger,
                position: Point::new(50.0, 70.0),
            },
        );
        let moved = changed_spatial_orbit(captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerMoved {
                id: finger,
                position: Point::new(100.0, 70.0),
            },
        ));
        assert_ne!(moved, controller.orbit);

        let position = Point::new(250.0, 90.0);
        let ended = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerLifted {
                id: finger,
                position,
            },
        );
        let Some(ViewportMessage::OrbitGestureEnded(orbit)) = ended else {
            panic!("final lift should finish spatial rotation");
        };
        let expected = controller.orbit.arcball_drag(
            SpatialViewportController::local_position(Point::new(50.0, 70.0), bounds),
            SpatialViewportController::local_position(position, bounds),
            SpatialViewportController::viewport(bounds),
        );
        assert_same_spatial_orbit(orbit, expected);
        assert_ne!(orbit, moved);
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
        assert!(state.touch_gesture.is_none());
        assert!(state.working_orbit.is_none());
    }

    #[test]
    fn spatial_lost_touch_keeps_the_last_rotation_and_releases_capture() {
        let controller = SpatialViewportController {
            orbit: Orbit3d::canonical(),
            epoch: 1,
        };
        let bounds = viewport_bounds();
        let mut state = SpatialViewportControllerState::default();
        let finger = iced::touch::Finger(1);
        let _ = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerPressed {
                id: finger,
                position: Point::new(50.0, 70.0),
            },
        );
        let moved = changed_spatial_orbit(captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerMoved {
                id: finger,
                position: Point::new(90.0, 80.0),
            },
        ));
        let ended = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerLost {
                id: finger,
                position: Point::new(100_000.0, -100_000.0),
            },
        );
        assert!(matches!(
            ended,
            Some(ViewportMessage::OrbitGestureEnded(orbit)) if orbit == moved
        ));
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
        assert!(state.touch_gesture.is_none());
    }

    #[test]
    fn spatial_second_finger_rebases_without_releasing_the_owned_gesture() {
        let controller = SpatialViewportController {
            orbit: Orbit3d::canonical(),
            epoch: 1,
        };
        let bounds = viewport_bounds();
        let mut state = SpatialViewportControllerState::default();
        let first = iced::touch::Finger(1);
        let second = iced::touch::Finger(2);
        let _ = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerPressed {
                id: first,
                position: Point::new(50.0, 70.0),
            },
        );
        let before_second = changed_spatial_orbit(captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerMoved {
                id: first,
                position: Point::new(80.0, 70.0),
            },
        ));

        assert!(
            captured_spatial_touch(
                &controller,
                &mut state,
                bounds,
                iced::touch::Event::FingerPressed {
                    id: second,
                    position: Point::new(250.0, 70.0),
                },
            )
            .is_none()
        );
        assert_eq!(state.touches.len(), 2);
        let after_second = changed_spatial_orbit(captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerMoved {
                id: first,
                position: Point::new(80.0, 70.0),
            },
        ));
        assert_same_spatial_orbit(after_second, before_second);

        let after_lift = changed_spatial_orbit(captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerLifted {
                id: second,
                position: Point::new(250.0, 70.0),
            },
        ));
        assert_same_spatial_orbit(after_lift, before_second);
        assert!(state.touch_captured);
        assert_eq!(state.touches.len(), 1);
        let resumed = changed_spatial_orbit(captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerMoved {
                id: first,
                position: Point::new(80.0, 70.0),
            },
        ));
        assert_same_spatial_orbit(resumed, before_second);

        assert!(
            captured_spatial_touch(
                &controller,
                &mut state,
                bounds,
                iced::touch::Event::FingerMoved {
                    id: iced::touch::Finger(99),
                    position: Point::new(-1.0, -1.0),
                },
            )
            .is_none()
        );
        let ended = captured_spatial_touch(
            &controller,
            &mut state,
            bounds,
            iced::touch::Event::FingerLifted {
                id: first,
                position: Point::new(80.0, 70.0),
            },
        );
        assert!(matches!(ended, Some(ViewportMessage::OrbitGestureEnded(_))));
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
    }

    #[test]
    fn spatial_touch_rebases_across_resize_or_epoch_change_without_a_jump() {
        for (epoch, changed_bounds) in [
            (
                1,
                Rectangle::new(Point::new(30.0, 10.0), Size::new(250.0, 150.0)),
            ),
            (2, viewport_bounds()),
        ] {
            let controller = SpatialViewportController {
                orbit: Orbit3d::canonical(),
                epoch: 1,
            };
            let bounds = viewport_bounds();
            let mut state = SpatialViewportControllerState::default();
            let finger = iced::touch::Finger(1);
            let _ = captured_spatial_touch(
                &controller,
                &mut state,
                bounds,
                iced::touch::Event::FingerPressed {
                    id: finger,
                    position: Point::new(50.0, 70.0),
                },
            );
            let moved = changed_spatial_orbit(captured_spatial_touch(
                &controller,
                &mut state,
                bounds,
                iced::touch::Event::FingerMoved {
                    id: finger,
                    position: Point::new(90.0, 70.0),
                },
            ));
            let changed = SpatialViewportController {
                orbit: moved,
                epoch,
            };
            let after_change = changed_spatial_orbit(captured_spatial_touch(
                &changed,
                &mut state,
                changed_bounds,
                iced::touch::Event::FingerMoved {
                    id: finger,
                    position: Point::new(90.0, 70.0),
                },
            ));
            assert_same_spatial_orbit(after_change, moved);
            assert_eq!(state.epoch, epoch);
            assert_eq!(state.last_bounds, Some(changed_bounds));
            assert!(state.touch_captured);

            let (message, _, status) = changed
                .update(
                    &mut state,
                    &canvas::Event::Window(iced::window::Event::Unfocused),
                    changed_bounds,
                    mouse::Cursor::Unavailable,
                )
                .expect("focus loss should finish the captured gesture")
                .into_inner();
            assert_eq!(status, iced::event::Status::Captured);
            assert!(matches!(
                message,
                Some(Message::Viewport(ViewportMessage::OrbitGestureEnded(_)))
            ));
            assert!(!state.touch_captured);
            assert!(state.touches.is_empty());
        }
    }

    #[test]
    fn cancel_affordance_is_delayed_for_render_work() {
        assert!(!should_offer_render_cancel(Duration::from_millis(1_999)));
        assert!(should_offer_render_cancel(Duration::from_secs(2)));
    }

    #[test]
    fn calculates_bounds_for_wide_and_tall_geometry() {
        for (end, expected) in [
            ((10.0, 2.0), (0.0, 10.0, 0.0, 2.0)),
            ((2.0, 10.0), (0.0, 2.0, 0.0, 10.0)),
        ] {
            let bounds = render_bounds(&[RenderLine {
                start: (0.0, 0.0),
                end,
                width: 1.0,
            }]);
            assert_eq!(
                (bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y),
                expected
            );
        }
    }

    #[test]
    fn projection_preserves_aspect_ratio_and_centers_geometry() {
        let line_bounds = RenderBounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 2.0,
        };
        let transform = ViewTransform::new(
            line_bounds.into(),
            ViewportSize::new(1_000.0, 600.0),
            Camera2d::fit(),
        );
        let start = transform.project(WorldPoint::new(0.0, 0.0));
        let end = transform.project(WorldPoint::new(10.0, 2.0));

        assert!(((start.x + end.x) * 0.5 - 500.0).abs() < 1.0e-9);
        assert!(((start.y + end.y) * 0.5 - 300.0).abs() < 1.0e-9);
        assert!(((end.x - start.x) / (start.y - end.y) - 5.0).abs() < 1.0e-9);
    }

    #[test]
    fn spatial_palette_position_clamps_instead_of_wrapping() {
        let transform = ViewTransform::new(
            ViewBounds::new(0.0, 100.0, 0.0, 100.0),
            ViewportSize::new(100.0, 100.0),
            Camera2d::fit(),
        );

        assert_eq!(
            transform.world_palette_position(
                WorldPoint::new(-100.0, 200.0),
                WorldPoint::new(-100.0, 200.0),
            ),
            0.0
        );
        assert_eq!(
            transform.world_palette_position(
                WorldPoint::new(200.0, -100.0),
                WorldPoint::new(200.0, -100.0),
            ),
            1.0
        );
    }

    #[test]
    fn software_preview_stroke_width_uses_the_exact_scene_metric() {
        let fit_scale = 8.0;
        let zoom = 3.0;
        let drawing_extent = (10.0, 10.0);
        let exact_total_length = Some(100.0);
        let preview_total_length = Some(25.0);
        let width = canvas_stroke_width(
            exact_total_length,
            100,
            drawing_extent,
            fit_scale,
            zoom,
            1.0,
        );

        assert!((width - 15.6).abs() < 1.0e-5);
        assert_ne!(
            width,
            canvas_stroke_width(
                preview_total_length,
                50,
                drawing_extent,
                fit_scale,
                zoom,
                1.0,
            )
        );
        assert!(
            (canvas_stroke_width(
                exact_total_length,
                100,
                drawing_extent,
                fit_scale,
                zoom,
                0.05
            ) / width
                - 0.05)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn software_preview_preserves_fractional_positive_widths() {
        assert_eq!(canvas_effective_line_width(2.0, 0.1), Some(0.2));
        assert_eq!(canvas_effective_line_width(2.0, 0.75), Some(1.5));
        assert_eq!(canvas_effective_line_width(2.0, 0.0), None);
        assert_eq!(canvas_effective_line_width(2.0, -1.0), None);
    }

    #[test]
    fn raster_invalidation_restarts_a_completed_line_scene() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let line_scene = test_line_scene(8);
        line_scene.set_refinement_progress_for_test(line_scene.line_count());
        assert!(!line_scene.is_refining());
        let mut result = ready_result(app.request_id, app.system_view_epoch);
        result.line_scene = Some(Arc::clone(&line_scene));
        app.render.displayed = Some(result);
        let raster_epoch = app.raster_epoch;

        app.invalidate_view_raster();

        assert_eq!(app.raster_epoch, raster_epoch.wrapping_add(1));
        assert!(line_scene.is_refining());
        assert_eq!(line_scene.refinement_progress(), (0, 8));
    }

    #[test]
    fn line_width_is_display_only_and_supports_five_percent() {
        let (mut app, _) = BrakenGui::new();
        let request_id = app.request_id;
        let cache_clock = app.generation_cache.clock;
        let raster_epoch = app.raster_epoch;
        let active_cancel = Arc::clone(
            &app.render
                .active
                .as_ref()
                .expect("initial render should be active")
                .cancelled,
        );
        let line_scene = test_line_scene(8);
        line_scene.set_refinement_progress_for_test(line_scene.line_count());
        let mut displayed = ready_result(app.request_id, app.system_view_epoch);
        displayed.line_scene = Some(Arc::clone(&line_scene));
        app.render.displayed = Some(displayed);
        let export_cancelled = Arc::new(AtomicBool::new(false));
        app.export_job = Some(ExportJob {
            export_id: 1,
            cancelled: Arc::clone(&export_cancelled),
        });
        app.export_notice = Some(String::from("Preparing SVG…"));

        let _ = app.update(Message::LineWidthChanged(5.0));

        assert_eq!(app.line_width_percent, 5.0);
        assert_eq!(app.line_width_scale(), 0.05);
        assert_eq!(app.request_id, request_id);
        assert_eq!(app.generation_cache.clock, cache_clock);
        assert_eq!(app.raster_epoch, raster_epoch.wrapping_add(1));
        assert!(app.render.active.as_ref().is_some_and(|job| {
            Arc::ptr_eq(&job.cancelled, &active_cancel) && !job.is_cancelled()
        }));
        assert!(line_scene.is_refining());
        assert!(export_cancelled.load(Ordering::Acquire));
        assert!(app.export_job.is_none());
        assert!(app.export_notice.is_none());
        app.render_coordinator.cancel_current();
    }

    #[test]
    fn software_preview_completion_stops_exact_refinement_status() {
        let line_scene = test_line_scene(8);
        assert!(line_scene.is_refining());

        line_scene.mark_software_preview_complete();

        assert!(!line_scene.is_refining());
        assert_eq!(line_scene.refinement_progress(), (8, 8));
    }

    #[test]
    fn spatial_preview_refinement_can_be_cancelled_from_the_status_action() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        let spatial = test_spatial_scene();
        let mut displayed = ready_result(app.request_id, app.system_view_epoch);
        displayed.scene = None;
        displayed.spatial_scene = Some(Arc::clone(&spatial));
        app.render.displayed = Some(displayed);

        assert!(spatial.is_refining());
        let _ = app.update(Message::CancelRender);

        assert!(!spatial.is_refining());
        assert!(spatial.refinement_was_cancelled());
        assert_eq!(spatial.refinement_progress(), (0, 1));
    }

    #[test]
    fn transient_render_and_refinement_statuses_wait_for_the_delay() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let started_at = RenderInstant::now();
        let mut displayed = ready_result(app.request_id, app.system_view_epoch);
        displayed.element_count = 8;
        displayed.refinement_started_at = started_at;
        displayed.line_scene = Some(test_line_scene(8));
        app.render.displayed = Some(displayed);
        app.render.active = None;

        let (_, immediate) = app.render_scene_and_status_at(started_at);
        assert_eq!(immediate, "8 elements");
        let (_, delayed) = app.render_scene_and_status_at(started_at + TRANSIENT_STATUS_DELAY);
        assert!(delayed.contains("refining display"));

        app.render.displayed.as_mut().unwrap().line_scene = None;
        app.render.active = Some(RenderJob {
            started_at,
            ..RenderJob::default()
        });
        let (_, immediate) = app.render_scene_and_status_at(started_at);
        assert_eq!(immediate, "8 elements");
        let (_, delayed) = app.render_scene_and_status_at(started_at + TRANSIENT_STATUS_DELAY);
        assert_eq!(delayed, "Queued…");
    }

    #[test]
    fn restarted_refinement_gets_a_fresh_status_delay() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let line_scene = test_line_scene(8);
        line_scene.set_refinement_progress_for_test(line_scene.line_count());
        let mut displayed = ready_result(app.request_id, app.system_view_epoch);
        displayed.element_count = 8;
        displayed.line_scene = Some(Arc::clone(&line_scene));
        displayed.refinement_started_at = RenderInstant::now() - TRANSIENT_STATUS_DELAY;
        app.render.displayed = Some(displayed);
        app.render.active = None;

        app.invalidate_view_raster();

        assert!(line_scene.is_refining());
        let restarted_at = app.render.displayed.as_ref().unwrap().refinement_started_at;
        let (_, status) = app.render_scene_and_status_at(restarted_at);
        assert_eq!(status, "8 elements");
    }

    #[test]
    fn desktop_mouse_drag_captures_until_release_outside() {
        let mut controller = viewport_controller(false);
        let bounds = viewport_bounds();
        controller.camera = controller.camera.zoom_by_in(
            4.0,
            controller.line_bounds.into(),
            ViewportSize::new(f64::from(bounds.width), f64::from(bounds.height)),
        );
        let mut state = ViewportControllerState::default();
        let press_cursor = mouse::Cursor::Available(Point::new(60.0, 70.0));
        let press = controller
            .update(
                &mut state,
                &canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
                bounds,
                press_cursor,
            )
            .expect("press should start a gesture");
        let (message, _, status) = press.into_inner();
        assert!(matches!(
            message,
            Some(Message::Viewport(ViewportMessage::GestureStarted))
        ));
        assert_eq!(status, iced::event::Status::Captured);

        let moved = controller
            .update(
                &mut state,
                &canvas::Event::Mouse(mouse::Event::CursorMoved {
                    position: Point::new(90.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Available(Point::new(90.0, 70.0)),
            )
            .expect("drag should update the camera");
        let moved_camera = match moved.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::GestureChanged(camera))) => camera,
            other => panic!("unexpected drag action: {other:?}"),
        };
        assert!(moved_camera.focus[0] < 0.5);

        let released = controller
            .update(
                &mut state,
                &canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
                bounds,
                mouse::Cursor::Available(Point::new(260.0, 70.0)),
            )
            .expect("release outside should still finish the gesture");
        assert!(matches!(
            released.into_inner().0,
            Some(Message::Viewport(ViewportMessage::GestureEnded(_)))
        ));
        assert!(state.mouse_drag.is_none());
    }

    #[test]
    fn mobile_touch_passes_one_finger_then_captures_two_until_all_lift() {
        let controller = viewport_controller(true);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let first = iced::touch::Finger(1);
        let second = iced::touch::Finger(2);

        let first_press = controller.update(
            &mut state,
            &canvas::Event::Touch(iced::touch::Event::FingerPressed {
                id: first,
                position: Point::new(50.0, 70.0),
            }),
            bounds,
            mouse::Cursor::Unavailable,
        );
        assert!(first_press.is_none());
        assert!(!state.touch_captured);

        let second_press = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed {
                    id: second,
                    position: Point::new(70.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("second finger should capture the canvas");
        assert!(state.touch_captured);
        assert!(matches!(
            second_press.into_inner().0,
            Some(Message::Viewport(ViewportMessage::GestureStarted))
        ));

        let pinch = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerMoved {
                    id: second,
                    position: Point::new(90.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("pinch should update the camera");
        let pinch_camera = match pinch.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::GestureChanged(camera))) => camera,
            other => panic!("unexpected pinch action: {other:?}"),
        };
        assert!(pinch_camera.zoom > 1.0);

        let first_lift = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerLifted {
                    id: second,
                    position: Point::new(90.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("captured gesture should retain the remaining finger");
        assert!(matches!(
            first_lift.into_inner().0,
            Some(Message::Viewport(ViewportMessage::GestureChanged(_)))
        ));
        assert!(state.touch_captured);
        assert_eq!(state.touches.len(), 1);

        let final_lift = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerLifted {
                    id: first,
                    position: Point::new(50.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("final lift should finish the gesture");
        assert!(matches!(
            final_lift.into_inner().0,
            Some(Message::Viewport(ViewportMessage::GestureEnded(_)))
        ));
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
    }

    #[test]
    fn focus_loss_finishes_an_active_view_gesture() {
        let controller = viewport_controller(false);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let _ = controller.update(
            &mut state,
            &canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            bounds,
            mouse::Cursor::Available(Point::new(50.0, 50.0)),
        );

        let action = controller
            .update(
                &mut state,
                &canvas::Event::Window(iced::window::Event::Unfocused),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("focus loss should close a captured gesture");
        assert!(matches!(
            action.into_inner().0,
            Some(Message::Viewport(ViewportMessage::GestureEnded(_)))
        ));
        assert!(state.mouse_drag.is_none());
    }

    #[test]
    fn lost_touch_ignores_its_reported_position_and_rebases_remaining_fingers() {
        let controller = viewport_controller(false);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let first = iced::touch::Finger(1);
        let second = iced::touch::Finger(2);
        for (id, position) in [
            (first, Point::new(50.0, 70.0)),
            (second, Point::new(70.0, 70.0)),
        ] {
            let _ = controller.update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }),
                bounds,
                mouse::Cursor::Unavailable,
            );
        }
        let moved = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerMoved {
                    id: second,
                    position: Point::new(90.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("pinch should update");
        let before_loss = match moved.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::GestureChanged(camera))) => camera,
            other => panic!("unexpected pinch action: {other:?}"),
        };

        let lost = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerLost {
                    id: second,
                    position: Point::new(100_000.0, -100_000.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("lost captured finger should be handled");
        let after_loss = match lost.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::GestureChanged(camera))) => camera,
            other => panic!("unexpected lost-finger action: {other:?}"),
        };
        assert_eq!(after_loss, before_loss);
        assert_eq!(state.touches.len(), 1);
        assert_eq!(
            state.touch_gesture.map(|gesture| gesture.baseline),
            Some(before_loss)
        );
    }

    #[test]
    fn view_epoch_change_keeps_two_finger_capture_until_all_fingers_lift() {
        let controller = viewport_controller(true);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let first = iced::touch::Finger(1);
        let second = iced::touch::Finger(2);
        for (id, position) in [
            (first, Point::new(50.0, 70.0)),
            (second, Point::new(70.0, 70.0)),
        ] {
            let _ = controller.update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }),
                bounds,
                mouse::Cursor::Unavailable,
            );
        }
        assert!(state.touch_captured);

        let replacement = ViewportController {
            camera: Camera2d {
                zoom: 2.0,
                focus: [0.25, 0.75],
            },
            epoch: 1,
            ..viewport_controller(true)
        };
        let moved = replacement
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerMoved {
                    id: second,
                    position: Point::new(80.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("move after replacement should remain captured");
        assert_eq!(moved.into_inner().2, iced::event::Status::Captured);
        assert_eq!(state.epoch, 1);
        assert!(state.touch_captured);

        for (id, position) in [
            (second, Point::new(80.0, 70.0)),
            (first, Point::new(50.0, 70.0)),
        ] {
            let lifted = replacement
                .update(
                    &mut state,
                    &canvas::Event::Touch(iced::touch::Event::FingerLifted { id, position }),
                    bounds,
                    mouse::Cursor::Unavailable,
                )
                .expect("every captured lift should remain captured");
            assert_eq!(lifted.into_inner().2, iced::event::Status::Captured);
        }
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
    }

    #[test]
    fn captured_touch_owns_outside_and_unknown_finger_events() {
        let controller = viewport_controller(true);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        for (id, position) in [
            (iced::touch::Finger(1), Point::new(50.0, 70.0)),
            (iced::touch::Finger(2), Point::new(70.0, 70.0)),
        ] {
            let _ = controller.update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }),
                bounds,
                mouse::Cursor::Unavailable,
            );
        }

        let outside = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed {
                    id: iced::touch::Finger(3),
                    position: Point::new(500.0, 500.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("outside third finger should remain owned");
        assert_eq!(outside.into_inner().2, iced::event::Status::Captured);
        assert_eq!(state.touches.len(), 3);

        for event in [
            iced::touch::Event::FingerMoved {
                id: iced::touch::Finger(99),
                position: Point::new(-1.0, -1.0),
            },
            iced::touch::Event::FingerLifted {
                id: iced::touch::Finger(99),
                position: Point::new(-1.0, -1.0),
            },
        ] {
            let action = controller
                .update(
                    &mut state,
                    &canvas::Event::Touch(event),
                    bounds,
                    mouse::Cursor::Unavailable,
                )
                .expect("unknown touch event should not bubble mid-gesture");
            assert_eq!(action.into_inner().2, iced::event::Status::Captured);
        }
        assert_eq!(state.touches.len(), 3);
    }

    #[test]
    fn disabling_navigation_drains_existing_capture_without_emitting_camera_changes() {
        let controller = viewport_controller(true);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let first = iced::touch::Finger(1);
        let second = iced::touch::Finger(2);
        for (id, position) in [
            (first, Point::new(50.0, 70.0)),
            (second, Point::new(70.0, 70.0)),
        ] {
            let _ = controller.update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }),
                bounds,
                mouse::Cursor::Unavailable,
            );
        }

        let disabled = ViewportController {
            enabled: false,
            epoch: 1,
            ..viewport_controller(true)
        };
        let moved = disabled
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerMoved {
                    id: second,
                    position: Point::new(90.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("disabled controller should drain the old gesture");
        let (message, _, status) = moved.into_inner();
        assert!(message.is_none());
        assert_eq!(status, iced::event::Status::Captured);

        for (id, position) in [
            (second, Point::new(90.0, 70.0)),
            (first, Point::new(50.0, 70.0)),
        ] {
            let action = disabled
                .update(
                    &mut state,
                    &canvas::Event::Touch(iced::touch::Event::FingerLifted { id, position }),
                    bounds,
                    mouse::Cursor::Unavailable,
                )
                .expect("disabled controller should capture old lifts");
            assert!(action.into_inner().0.is_none());
        }
        assert!(!state.touch_captured);
        assert!(state.touches.is_empty());
    }

    #[test]
    fn view_epoch_change_keeps_mouse_capture_through_release() {
        let controller = viewport_controller(false);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let _ = controller.update(
            &mut state,
            &canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            bounds,
            mouse::Cursor::Available(Point::new(50.0, 50.0)),
        );

        let replacement = ViewportController {
            camera: Camera2d {
                zoom: 3.0,
                focus: [0.1, 0.9],
            },
            epoch: 1,
            ..viewport_controller(false)
        };
        let released = replacement
            .update(
                &mut state,
                &canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
                bounds,
                mouse::Cursor::Available(Point::new(260.0, 70.0)),
            )
            .expect("release after replacement should remain captured");
        assert_eq!(released.into_inner().2, iced::event::Status::Captured);
        assert!(state.mouse_drag.is_none());
    }

    #[test]
    fn bounds_change_rebases_an_active_pinch_without_a_camera_jump() {
        let controller = viewport_controller(false);
        let bounds = viewport_bounds();
        let mut state = ViewportControllerState::default();
        let first = iced::touch::Finger(1);
        let second = iced::touch::Finger(2);
        for (id, position) in [
            (first, Point::new(50.0, 70.0)),
            (second, Point::new(70.0, 70.0)),
        ] {
            let _ = controller.update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerPressed { id, position }),
                bounds,
                mouse::Cursor::Unavailable,
            );
        }
        let moved = controller
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerMoved {
                    id: second,
                    position: Point::new(90.0, 70.0),
                }),
                bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("pinch should update");
        let before_resize = match moved.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::GestureChanged(camera))) => camera,
            other => panic!("unexpected pinch action: {other:?}"),
        };

        let resized_bounds = Rectangle::new(Point::new(30.0, 10.0), Size::new(250.0, 150.0));
        let rebased = ViewportController {
            camera: before_resize,
            ..viewport_controller(false)
        };
        let unchanged_move = rebased
            .update(
                &mut state,
                &canvas::Event::Touch(iced::touch::Event::FingerMoved {
                    id: second,
                    position: Point::new(90.0, 70.0),
                }),
                resized_bounds,
                mouse::Cursor::Unavailable,
            )
            .expect("pinch should stay captured across a resize");
        let after_resize = match unchanged_move.into_inner().0 {
            Some(Message::Viewport(ViewportMessage::GestureChanged(camera))) => camera,
            other => panic!("unexpected resized pinch action: {other:?}"),
        };

        assert!((after_resize.zoom - before_resize.zoom).abs() < 1.0e-12);
        assert!((after_resize.focus[0] - before_resize.focus[0]).abs() < 1.0e-12);
        assert!((after_resize.focus[1] - before_resize.focus[1]).abs() < 1.0e-12);
        assert_eq!(state.last_bounds, Some(resized_bounds));
        assert!(state.touch_captured);
    }

    #[test]
    fn theme_preference_cycles_through_all_modes() {
        assert_eq!(ThemePreference::Auto.next(), ThemePreference::Light);
        assert_eq!(ThemePreference::Light.next(), ThemePreference::Dark);
        assert_eq!(ThemePreference::Dark.next(), ThemePreference::Auto);
    }

    #[test]
    fn auto_theme_resolves_system_mode_and_defaults_none_to_light() {
        assert_eq!(
            resolve_theme_mode(ThemePreference::Auto, theme::Mode::Dark),
            theme::Mode::Dark
        );
        assert_eq!(
            resolve_theme_mode(ThemePreference::Auto, theme::Mode::Light),
            theme::Mode::Light
        );
        assert_eq!(
            resolve_theme_mode(ThemePreference::Auto, theme::Mode::None),
            theme::Mode::Light
        );
    }

    #[test]
    fn manual_theme_preferences_override_system_mode() {
        for system in [theme::Mode::None, theme::Mode::Light, theme::Mode::Dark] {
            assert_eq!(
                resolve_theme_mode(ThemePreference::Light, system),
                theme::Mode::Light
            );
            assert_eq!(
                resolve_theme_mode(ThemePreference::Dark, system),
                theme::Mode::Dark
            );
        }
    }

    #[test]
    fn generation_cache_uses_exact_source_iterations_and_seed() {
        let mut cache = GenerationCache::new(1024 * 1024);
        let key = cache_key("axiom F;", 3);
        let size = estimate_cache_entry_bytes(&key, 1, 1);
        cache.insert(key.clone(), generation("F"), derivation("CPU"), size);

        assert!(cache.get(&key).is_some());
        assert!(cache.get(&cache_key("axiom  F;", 3)).is_none());
        assert!(cache.get(&cache_key("axiom F;", 4)).is_none());
        let mut different_seed = key.clone();
        different_seed.seed = 1;
        assert!(cache.get(&different_seed).is_none());
    }

    #[test]
    fn generation_cache_evicts_the_least_recently_used_entry() {
        let first = cache_key("first", 1);
        let second = cache_key("second", 1);
        let third = cache_key("third", 1);
        let entry_size = estimate_cache_entry_bytes(&first, 1, 1);
        let mut cache = GenerationCache::new(entry_size * 2 + 8);

        cache.insert(
            first.clone(),
            generation("F"),
            derivation("CPU"),
            entry_size,
        );
        cache.insert(
            second.clone(),
            generation("F"),
            derivation("CPU"),
            entry_size,
        );
        assert!(cache.get(&first).is_some());
        cache.insert(
            third.clone(),
            generation("F"),
            derivation("CPU"),
            entry_size,
        );

        assert!(cache.get(&first).is_some());
        assert!(cache.get(&second).is_none());
        assert!(cache.get(&third).is_some());
    }

    #[test]
    fn generation_cache_rejects_an_oversized_entry() {
        let key = cache_key("axiom F;", 1);
        let mut cache = GenerationCache::new(1);
        let size = estimate_cache_entry_bytes(&key, 1, 1);
        cache.insert(key.clone(), generation("F"), derivation("CPU"), size);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn generation_cache_preserves_the_actual_derivation_backend() {
        let key = cache_key("axiom F;", 1);
        let mut cache = GenerationCache::new(1024 * 1024);
        let size = estimate_cache_entry_bytes(&key, 1, 1);
        cache.insert(key.clone(), generation("F"), derivation("CUDA"), size);

        let cached = cache.get(&key).expect("cached generation");
        assert_eq!(cached.derivation.backend, "CUDA");
        assert_eq!(cached.derivation.label(), "CUDA");
        assert_eq!(cached.derivation.as_cache_hit().label(), "CUDA (cached)");
        assert_eq!(cached.derivation.elapsed, Some(Duration::from_millis(7)));
    }

    #[test]
    fn cached_requests_report_the_backend_that_derived_the_generation() {
        let request = AppVisualizeRequest {
            request_id: 1,
            view_epoch: 1,
            key: cache_key("axiom F;", 1),
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 0,
            cached_generation: Some(CachedGeneration {
                generation: generation("F"),
                derivation: derivation("WGPU"),
            }),
            transition: None,
            transition_source: None,
            job: RenderJob::default(),
        };

        let (cached, insertion, info, ir_tooling) = generation_for_request(&request).unwrap();
        assert_eq!(cached, generation("F"));
        assert!(insertion.is_none());
        assert_eq!(info.label(), "WGPU (cached)");
        assert_eq!(info.elapsed, Some(Duration::from_millis(7)));
        assert!(ir_tooling.disassembly.contains("Braken IR"));
    }

    #[test]
    fn imported_ir_is_validated_and_can_drive_derivation_without_source_parsing() {
        let grammar = CompiledGrammar::parse("axiom A; match A then B;").unwrap();
        let json = grammar.to_ir_document().to_json_pretty().unwrap();
        let key = GenerationCacheKey {
            source: Arc::from("this is deliberately not grammar source"),
            ir_json: Some(Arc::from(json)),
            iterations: 1,
            seed: 0,
            semantics: DerivationSemantics::default(),
        };
        let imported = compiled_grammar_for_key(&key).unwrap();
        assert_eq!(
            imported
                .grammar()
                .compile_cpu()
                .unwrap()
                .run(1)
                .unwrap()
                .to_string(),
            "B"
        );
    }

    #[test]
    fn gui_import_keeps_ir_inactive_until_the_user_applies_it() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let grammar = CompiledGrammar::parse("axiom Imported;").unwrap();
        let json = grammar.to_ir_document().to_json_pretty().unwrap();
        let _ = app.update(Message::IrClipboardRead(Some(json)));
        assert!(app.imported_ir.is_some());
        assert!(app.active_ir_json.is_none());

        let _ = app.update(Message::UseImportedIr);
        app.render_coordinator.cancel_current();
        assert!(app.active_ir_json.is_some());
    }

    #[test]
    fn adjacent_transition_handles_branches_and_promotes_exact_target() {
        let source = Arc::<str>::from("axiom F; match F then F [ TurnLeft F ] F;");
        let grammar = braken::Grammar::parse(&source).unwrap();
        let program = grammar.compile_cpu().unwrap();
        let first = Arc::new(program.run_with_seed(1, 9).unwrap());
        let second = Arc::new(program.run_with_seed(2, 9).unwrap());
        let request = AppVisualizeRequest {
            request_id: 7,
            view_epoch: 3,
            key: GenerationCacheKey {
                source,
                ir_json: None,
                iterations: 2,
                seed: 9,
                semantics: DerivationSemantics::default(),
            },
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 9,
            cached_generation: Some(CachedGeneration {
                generation: second,
                derivation: derivation("CPU"),
            }),
            transition: Some(IterationTransitionRequest {
                from_iteration: 1,
                to_iteration: 2,
            }),
            transition_source: Some(CachedGeneration {
                generation: first,
                derivation: derivation("CPU"),
            }),
            job: RenderJob::default(),
        };

        let outcome = render_system(request, &mut Turtle2dStreamer::new());
        let RenderOutcome::TransitionReady(transition) = outcome else {
            panic!("expected a prepared transition")
        };
        assert_eq!(transition.from_iteration, 1);
        assert_eq!(transition.to_iteration, 2);
        assert_eq!(transition.scene.line_count(), 9);
        assert_eq!(transition.target.iteration, 2);
        assert_eq!(transition.target.element_count, 9);
    }

    #[test]
    fn initial_hilbert_transition_runs_from_iteration_zero_to_one() {
        let source = Arc::<str>::from(
            "axiom L; match L then Left R Draw Right L Draw L Right Draw R Left; match R then Right L Draw Left R Draw R Left Draw L Right;",
        );
        let grammar = braken::Grammar::parse(&source).unwrap();
        let program = grammar.compile_cpu().unwrap();
        let zeroth = Arc::new(program.run_with_seed(0, 9).unwrap());
        let first = Arc::new(program.run_with_seed(1, 9).unwrap());
        let request = AppVisualizeRequest {
            request_id: 8,
            view_epoch: 4,
            key: GenerationCacheKey {
                source,
                ir_json: None,
                iterations: 1,
                seed: 9,
                semantics: DerivationSemantics::default(),
            },
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 9,
            cached_generation: Some(CachedGeneration {
                generation: first,
                derivation: derivation("CPU"),
            }),
            transition: Some(IterationTransitionRequest {
                from_iteration: 0,
                to_iteration: 1,
            }),
            transition_source: Some(CachedGeneration {
                generation: zeroth,
                derivation: derivation("CPU"),
            }),
            job: RenderJob::default(),
        };

        let outcome = render_system(request, &mut Turtle2dStreamer::new());
        let RenderOutcome::TransitionReady(transition) = outcome else {
            panic!("expected the initial prepared transition")
        };
        assert_eq!(transition.from_iteration, 0);
        assert_eq!(transition.to_iteration, 1);
        assert_eq!(transition.scene.line_count(), 3);
        assert_eq!(transition.target.iteration, 1);
        assert_eq!(transition.target.element_count, 3);
    }

    #[test]
    fn endpoint_orientation_keeps_the_dragon_target_upright() {
        let source = Arc::<str>::from(
            "axiom Draw L; match L then L Left R Draw Left; match R then Right Draw L Right R;",
        );
        let grammar = braken::Grammar::parse(&source).unwrap();
        let generation = Arc::new(grammar.compile_cpu().unwrap().run_with_seed(2, 0).unwrap());
        let request = AppVisualizeRequest {
            request_id: 9,
            view_epoch: 5,
            key: GenerationCacheKey {
                source,
                ir_json: None,
                iterations: 2,
                seed: 0,
                semantics: DerivationSemantics::default(),
            },
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: Some(OrientationAnchor::Endpoint),
            seed: 0,
            cached_generation: Some(CachedGeneration {
                generation,
                derivation: derivation("CPU"),
            }),
            transition: None,
            transition_source: None,
            job: RenderJob::default(),
        };

        let RenderOutcome::Ready(result) = render_system(request, &mut Turtle2dStreamer::new())
        else {
            panic!("expected an upright exact Dragon scene")
        };
        let preview = result
            .line_scene
            .expect("Dragon should produce a line scene")
            .fallback_preview(usize::MAX)
            .unwrap();
        let lines = preview
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive2d::Line(line) => Some(line.line),
                Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
            });
        let lines = lines.collect::<Vec<_>>();
        let chord = (
            lines.last().unwrap().1.0 - lines.first().unwrap().0.0,
            lines.last().unwrap().1.1 - lines.first().unwrap().0.1,
        );
        assert!(chord.0.abs() < 1.0e-5, "Dragon chord was {chord:?}");
        assert!(chord.1 > 0.0, "Dragon chord was {chord:?}");
    }

    #[test]
    fn slider_routes_from_the_last_compatible_completed_iteration() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.visualizer = VisualizerKind::Turtle2d;
        app.request_id = 20;
        let identity = app.current_render_identity();
        let mut result = ready_result(20, app.system_view_epoch);
        result.iteration = 2;
        result.identity = identity;
        app.render.displayed = Some(result);

        let _ = app.update(Message::IterationsChanged(5.0));

        let route = app.iteration_route.as_ref().expect("iteration route");
        assert_eq!(route.target_iteration, 5);
        assert!(app.render.active.is_some());
        app.render_coordinator.cancel_current();
    }

    #[test]
    fn selecting_iteration_zero_routes_the_reverse_one_to_zero_transition() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.visualizer = VisualizerKind::Turtle2d;
        app.request_id = 21;
        let identity = app.current_render_identity();
        let mut result = ready_result(21, app.system_view_epoch);
        result.iteration = 1;
        result.identity = identity;
        app.render.displayed = Some(result);

        let _ = app.update(Message::IterationsChanged(0.0));

        assert_eq!(app.iterations, 0);
        assert_eq!(
            app.iteration_route
                .as_ref()
                .map(|route| route.target_iteration),
            Some(0)
        );
        assert!(app.render.active.is_some());
        app.render_coordinator.cancel_current();
    }

    #[test]
    fn replacing_a_mid_animation_route_restarts_from_the_last_exact_scene() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.visualizer = VisualizerKind::Turtle2d;
        app.request_id = 30;
        let identity = app.current_render_identity();
        let mut displayed = ready_result(30, app.system_view_epoch);
        displayed.iteration = 2;
        displayed.identity = identity.clone();
        app.render.displayed = Some(displayed);
        let mut target = ready_result(30, app.system_view_epoch);
        target.iteration = 3;
        target.identity = identity.clone();
        app.iteration_route = Some(IterationRoute {
            request_id: 30,
            target_iteration: 4,
            identity,
        });
        app.iteration_animation = Some(IterationAnimation {
            started_at: RenderInstant::now(),
            paused_at: None,
            paused_duration: Duration::ZERO,
            progress: 0.5,
            prepared: PreparedIterationTransition {
                request_id: 30,
                from_iteration: 2,
                to_iteration: 3,
                scene: TransitionScene::new(
                    Vec::new(),
                    RenderBounds::default(),
                    RenderBounds::default(),
                    (0, None),
                    (0, None),
                    None,
                ),
                target,
            },
        });

        let _ = app.update(Message::IterationsChanged(5.0));

        assert!(app.iteration_animation.is_none());
        assert_eq!(
            app.render.displayed.as_ref().map(|result| result.iteration),
            Some(2)
        );
        assert_eq!(
            app.iteration_route
                .as_ref()
                .map(|route| route.target_iteration),
            Some(5)
        );
        assert_eq!(app.request_id, 31);
        assert!(app.render.active.is_some());
        app.render_coordinator.cancel_current();
    }

    struct TestWork {
        request_id: u64,
        cancelled: Arc<AtomicBool>,
    }

    impl TestWork {
        fn new(request_id: u64) -> Self {
            Self {
                request_id,
                cancelled: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    impl WorkerWork for TestWork {
        fn request_id(&self) -> u64 {
            self.request_id
        }

        fn cancel(&self) {
            self.cancelled.store(true, Ordering::Release);
        }
    }

    #[test]
    fn worker_queue_keeps_one_replaceable_pending_request() {
        let mut queue = WorkerQueue::default();
        assert!(matches!(
            queue.submit(TestWork::new(1)),
            Some(WorkerCommand::Run(TestWork { request_id: 1, .. }))
        ));

        let second = TestWork::new(2);
        let second_cancelled = Arc::clone(&second.cancelled);
        assert!(matches!(
            queue.submit(second),
            Some(WorkerCommand::Cancel { request_id: 1 })
        ));
        let third = TestWork::new(3);
        let third_cancelled = Arc::clone(&third.cancelled);
        assert!(queue.submit(third).is_none());
        assert!(second_cancelled.load(Ordering::Acquire));
        assert_eq!(queue.pending.as_ref().map(WorkerWork::request_id), Some(3));

        let command = queue.finish(1).expect("latest work should start");
        assert!(matches!(
            command,
            WorkerCommand::Run(TestWork { request_id: 3, .. })
        ));
        assert!(!third_cancelled.load(Ordering::Acquire));
        assert!(queue.pending.is_none());
    }

    #[test]
    fn cancelling_worker_queue_drops_pending_and_emits_cancel_once() {
        let mut queue = WorkerQueue::default();
        let _run = queue.submit(TestWork::new(1));
        let pending = TestWork::new(2);
        let pending_cancelled = Arc::clone(&pending.cancelled);
        let _cancel = queue.submit(pending);

        assert!(queue.cancel().is_none());
        assert!(pending_cancelled.load(Ordering::Acquire));
        assert!(queue.pending.is_none());
        assert!(queue.cancelling);
    }

    #[test]
    fn stale_ready_results_never_enter_the_generation_cache() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.request_id = 20;
        let key = cache_key("stale", 1);
        let _ = app.update(Message::Rendered(RenderOutcome::Ready(RenderResult {
            request_id: 19,
            view_epoch: 1,
            scene: Some(Arc::new(Scene2d::default())),
            line_scene: None,
            spatial_scene: None,
            element_count: 0,
            refinement_started_at: RenderInstant::now(),
            elapsed_ms: 1,
            backend: String::from("CPU"),
            bounds: RenderBounds::default(),
            cache_insert: Some((key.clone(), generation("F"), derivation("CPU"), 128)),
            iteration: 1,
            identity: test_identity(),
            ir_tooling: test_ir_tooling(),
        })));
        assert!(app.generation_cache.get(&key).is_none());
    }

    #[test]
    fn viewport_navigation_preserves_render_work_cache_and_raster_epoch() {
        let (mut app, _) = BrakenGui::new();
        let request_id = app.request_id;
        let cache_clock = app.generation_cache.clock;
        let raster_epoch = app.raster_epoch;
        let active_cancel = Arc::clone(
            &app.render
                .active
                .as_ref()
                .expect("initial render should be active")
                .cancelled,
        );

        let _ = app.update(Message::Viewport(ViewportMessage::ZoomBy {
            factor: 2.0,
            viewport: ViewportSize::new(800.0, 600.0),
        }));

        assert_eq!(app.request_id, request_id);
        assert_eq!(app.generation_cache.clock, cache_clock);
        assert_eq!(app.raster_epoch, raster_epoch);
        assert!(app.render.active.as_ref().is_some_and(|job| {
            Arc::ptr_eq(&job.cancelled, &active_cancel) && !job.is_cancelled()
        }));
        assert_eq!(app.live_camera.zoom, 2.0);
        assert_eq!(app.settled_camera, app.live_camera);

        app.live_camera.focus = [-100.0, 100.0];
        let _ = app.update(Message::Viewport(ViewportMessage::ZoomBy {
            factor: 0.001,
            viewport: ViewportSize::new(800.0, 600.0),
        }));
        assert_eq!(app.live_camera, Camera2d::fit());
        assert_eq!(app.request_id, request_id);
        app.render_coordinator.cancel_current();
    }

    #[test]
    fn view_resets_only_after_a_successful_new_system_result() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.live_camera = Camera2d {
            zoom: 4.0,
            focus: [0.2, 0.8],
        };
        app.settled_camera = app.live_camera;
        app.system_view_epoch = 7;
        app.displayed_view_epoch = 3;
        app.request_id = 40;

        let _ = app.update(Message::Rendered(RenderOutcome::Failed {
            request_id: 40,
            message: String::from("invalid source"),
        }));
        assert_eq!(app.live_camera.zoom, 4.0);
        assert_eq!(app.displayed_view_epoch, 3);

        app.render.active = Some(RenderJob::default());
        let _ = app.update(Message::Rendered(RenderOutcome::Ready(ready_result(40, 7))));
        assert_eq!(app.live_camera, Camera2d::fit());
        assert_eq!(app.settled_camera, Camera2d::fit());
        assert_eq!(app.displayed_view_epoch, 7);
    }

    #[test]
    fn successful_parameter_result_preserves_the_current_view() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let camera = Camera2d {
            zoom: 3.0,
            focus: [0.25, 0.75],
        };
        app.live_camera = camera;
        app.settled_camera = camera;
        app.system_view_epoch = 5;
        app.displayed_view_epoch = 5;
        app.request_id = 12;

        let _ = app.update(Message::Rendered(RenderOutcome::Ready(ready_result(12, 5))));

        assert_eq!(app.live_camera, camera);
        assert_eq!(app.settled_camera, camera);
        assert_eq!(app.displayed_view_epoch, 5);
    }

    #[test]
    fn wheel_zoom_uses_live_camera_until_settle_then_restarts_display_refinement() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let line_scene = test_line_scene(8);
        line_scene.set_refinement_progress_for_test(line_scene.line_count());
        let mut displayed = ready_result(app.request_id, app.system_view_epoch);
        displayed.line_scene = Some(Arc::clone(&line_scene));
        app.render.displayed = Some(displayed);
        let _ = app.update(Message::Viewport(ViewportMessage::WheelZoom {
            factor: 2.0,
            anchor: ScreenPoint::new(50.0, 50.0),
            viewport: ViewportSize::new(100.0, 100.0),
        }));
        assert_eq!(app.live_camera.zoom, 2.0);
        assert_eq!(app.settled_camera.zoom, 1.0);
        assert!(app.view_is_interacting());

        app.wheel_changed_at = Some(RenderInstant::now() - VIEW_WHEEL_SETTLE_DELAY);
        let _ = app.update(Message::Frame);

        assert_eq!(app.settled_camera, app.live_camera);
        assert!(!app.view_is_interacting());
        assert_eq!(line_scene.refinement_progress(), (0, 8));
    }

    #[test]
    fn source_epoch_is_not_changed_by_parameter_or_theme_updates() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let epoch = app.system_view_epoch;

        let _ = app.update(Message::AngleChanged(45.0));
        app.render_coordinator.cancel_current();
        let _ = app.update(Message::IterationsChanged(2.0));
        app.render_coordinator.cancel_current();
        let _ = app.update(Message::SeedChanged(String::from("42")));
        app.render_coordinator.cancel_current();
        let _ = app.update(Message::CycleTheme);

        assert_eq!(app.system_view_epoch, epoch);
        let _ = app.update(Message::PresetSelected(PresetChoice(0)));
        app.render_coordinator.cancel_current();
        assert_eq!(app.system_view_epoch, epoch.wrapping_add(1));
    }

    #[test]
    fn an_actual_source_edit_advances_the_view_epoch_without_resetting_yet() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let epoch = app.system_view_epoch;
        let camera = Camera2d {
            zoom: 5.0,
            focus: [0.1, 0.9],
        };
        app.live_camera = camera;
        app.settled_camera = camera;

        let _ = app.update(Message::SourceEdited(text_editor::Action::Edit(
            text_editor::Edit::Insert(' '),
        )));

        assert_eq!(app.system_view_epoch, epoch.wrapping_add(1));
        assert_eq!(app.live_camera, camera);
        assert_eq!(app.settled_camera, camera);
        assert!(app.source_changed_at.is_some());
    }

    #[test]
    fn iteration_value_is_not_clamped_to_the_preset_suggestion() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.suggested_max_iterations = 5;
        app.set_iterations(1_000_000);
        assert_eq!(app.iterations, 1_000_000);
        assert_eq!(app.iterations_input, "1000000");
    }

    #[test]
    fn weighted_grammars_are_cached_under_their_exact_seed() {
        let request = AppVisualizeRequest {
            request_id: 1,
            view_epoch: 1,
            key: cache_key(
                "axiom F; match F weight 1 then F; match F weight 2 then F F;",
                1,
            ),
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 0,
            cached_generation: None,
            transition: None,
            transition_source: None,
            job: RenderJob::default(),
        };
        let (_, insertion, _, _) = generation_for_request(&request).unwrap();
        let (key, _, _, _) = insertion.expect("completed derivation should be cacheable");
        assert_eq!(key.seed, request.seed);
    }

    #[test]
    fn ambiguous_grammars_are_cached_under_their_exact_seed() {
        let request = AppVisualizeRequest {
            request_id: 1,
            view_epoch: 1,
            key: cache_key("axiom F; match F then F F; match F then F F F;", 1),
            angle: 90.0,
            visualizer: VisualizerKind::Turtle2d,
            turtle_config: Turtle2dConfig::default(),
            orientation_anchor: None,
            seed: 0,
            cached_generation: None,
            transition: None,
            transition_source: None,
            job: RenderJob::default(),
        };
        let (_, insertion, _, _) = generation_for_request(&request).unwrap();
        assert!(insertion.is_some());
    }

    #[test]
    fn seed_input_switches_between_automatic_and_custom_modes() {
        let (mut app, _) = BrakenGui::new();
        assert!(app.seed_input.is_empty());
        assert!(app.seed_placeholder.parse::<u64>().is_ok());
        let automatic_seed = app.effective_seed;

        let _ = app.update(Message::SeedChanged(String::from("42")));
        assert_eq!(app.seed_input, "42");
        assert_eq!(app.effective_seed, 42);
        assert!(app.seed_notice.is_none());

        let _ = app.update(Message::SeedChanged(String::from("invalid")));
        assert_eq!(app.effective_seed, 42);
        assert!(app.seed_notice.is_some());

        let _ = app.update(Message::SeedChanged(String::new()));
        assert!(app.seed_input.is_empty());
        assert!(app.seed_placeholder.parse::<u64>().is_ok());
        assert_eq!(app.effective_seed, automatic_seed);
        assert!(app.seed_notice.is_none());
    }

    #[test]
    fn ordinary_control_changes_reuse_the_current_seed() {
        let (mut app, _) = BrakenGui::new();
        let seed = app.effective_seed;

        let _ = app.update(Message::AngleChanged(30.0));
        assert_eq!(app.effective_seed, seed);
        let _ = app.update(Message::IterationsChanged(2.0));
        assert_eq!(app.effective_seed, seed);
    }

    #[test]
    fn angle_slider_preserves_fractional_values() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        let request_id = app.request_id;

        let _ = app.update(Message::AngleChanged(42.5));

        assert_eq!(app.angle, 42.5);
        assert_eq!(app.request_id, request_id.wrapping_add(1));
        assert!(app.selected_preset.is_none());
        app.render_coordinator.cancel_current();
    }

    #[test]
    fn angle_buttons_snap_to_the_adjacent_integer() {
        assert_eq!(adjacent_integer_angle(42.5, -1.0), 42.0);
        assert_eq!(adjacent_integer_angle(42.5, 1.0), 43.0);
        assert_eq!(adjacent_integer_angle(42.0, -1.0), 41.0);
        assert_eq!(adjacent_integer_angle(42.0, 1.0), 43.0);
        assert_eq!(adjacent_integer_angle(MIN_ANGLE, -1.0), MIN_ANGLE);
        assert_eq!(adjacent_integer_angle(MAX_ANGLE, 1.0), MAX_ANGLE);

        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.angle = 42.5;
        let _ = app.update(Message::AngleDecrement);
        assert_eq!(app.angle, 42.0);
        app.render_coordinator.cancel_current();

        app.angle = 42.5;
        let _ = app.update(Message::AngleIncrement);
        assert_eq!(app.angle, 43.0);
        app.render_coordinator.cancel_current();
    }

    #[test]
    fn angle_scroll_preserves_the_fractional_part() {
        assert_eq!(
            stepped_fractional_value(
                42.5,
                mouse::ScrollDelta::Lines { x: 0.0, y: 1.0 },
                MIN_ANGLE,
                MAX_ANGLE,
            ),
            43.5
        );
        assert_eq!(
            stepped_fractional_value(
                42.5,
                mouse::ScrollDelta::Lines { x: 0.0, y: -1.0 },
                MIN_ANGLE,
                MAX_ANGLE,
            ),
            41.5
        );
    }

    #[test]
    fn iteration_easing_is_bounded_and_settles() {
        assert_eq!(iteration_ease(0.0), 0.0);
        assert_eq!(iteration_ease(1.0), 1.0);
        let samples = (0..=100)
            .map(|step| iteration_ease(step as f32 / 100.0))
            .collect::<Vec<_>>();
        assert!(samples.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(samples.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn selecting_a_turtle_preset_keeps_its_configured_angle() {
        let (mut app, _) = BrakenGui::new();
        let (index, target_angle) = app
            .presets
            .iter()
            .enumerate()
            .find(|(_, preset)| {
                preset.visualizer == VisualizerKind::Turtle2d && preset.angle != 0.0
            })
            .map(|(index, preset)| (index, preset.angle as f32))
            .unwrap();

        let _ = app.update(Message::PresetSelected(PresetChoice(index)));

        assert_eq!(app.angle, target_angle);
        assert_eq!(app.selected_preset, Some(PresetChoice(index)));
    }

    #[test]
    fn selecting_an_angle_independent_preset_keeps_its_configured_angle() {
        let (mut app, _) = BrakenGui::new();
        let index = 0;
        let target_angle = 37.0;
        app.presets[index].visualizer = VisualizerKind::Inspector;
        app.presets[index].angle = f64::from(target_angle);

        let _ = app.update(Message::PresetSelected(PresetChoice(index)));

        assert_eq!(app.angle, target_angle);
    }

    #[test]
    fn randomize_seed_returns_to_automatic_mode() {
        let (mut app, _) = BrakenGui::new();
        let _ = app.update(Message::SeedChanged(String::from("42")));
        let _ = app.update(Message::RandomizeSeed);

        assert!(app.seed_input.is_empty());
        assert_eq!(
            app.effective_seed,
            app.seed_placeholder.parse::<u64>().unwrap()
        );
        assert_ne!(app.effective_seed, 42);
    }

    #[test]
    fn derivation_semantics_are_selectable_and_part_of_cache_identity() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        assert_eq!(app.derivation_semantics.float_width, FloatWidth::F32);
        assert_eq!(
            app.derivation_semantics.ambiguous_rules,
            AmbiguousRulePolicy::Uniform
        );

        let _ = app.update(Message::FloatWidthChanged(FloatWidth::F64));
        app.render_coordinator.cancel_current();
        let _ = app.update(Message::AmbiguousRulesChanged(AmbiguousRulePolicy::Error));
        app.render_coordinator.cancel_current();
        assert_eq!(
            app.current_render_identity().semantics,
            DerivationSemantics {
                float_width: FloatWidth::F64,
                ambiguous_rules: AmbiguousRulePolicy::Error,
            }
        );
        assert_ne!(
            GenerationCacheKey {
                semantics: DerivationSemantics::default(),
                ..cache_key("axiom A;", 1)
            },
            GenerationCacheKey {
                semantics: app.derivation_semantics,
                ..cache_key("axiom A;", 1)
            }
        );
    }

    #[test]
    fn mobile_catalog_pauses_spatial_autorotation_without_a_return_jump() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        app.layout_mode = LayoutMode::Mobile;
        app.render.displayed = Some(spatial_ready_result(app.request_id, app.system_view_epoch));
        app.autorotate_3d = true;
        let orbit = app.orbit_3d;
        let request = app.request_id;

        let _ = app.update(Message::OpenCatalog);
        app.autorotate_frame_at = Some(RenderInstant::now() - Duration::from_secs(10));
        let _ = app.update(Message::Frame);
        assert_eq!(app.orbit_3d, orbit);
        assert!(app.autorotate_3d);

        let _ = app.update(Message::CloseCatalog);
        assert_eq!(app.autorotate_frame_at, None);
        let _ = app.update(Message::Frame);
        assert_same_spatial_orbit(app.orbit_3d, orbit);
        app.autorotate_frame_at = Some(RenderInstant::now() - Duration::from_millis(50));
        let _ = app.update(Message::Frame);
        assert_ne!(app.orbit_3d, orbit);
        assert_eq!(app.request_id, request);
    }

    #[test]
    fn hidden_transition_and_visible_resize_respect_animation_pause_ownership() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        app.layout_mode = LayoutMode::Mobile;
        let request = app.request_id;
        app.render.displayed = Some(ready_result(request, app.system_view_epoch));
        let mut target = ready_result(request, app.system_view_epoch);
        target.iteration = 2;
        app.iteration_route = Some(IterationRoute {
            request_id: request,
            target_iteration: 2,
            identity: target.identity.clone(),
        });
        let _ = app.update(Message::OpenCatalog);
        let _ = app.update(Message::Rendered(RenderOutcome::TransitionReady(
            PreparedIterationTransition {
                request_id: request,
                from_iteration: 1,
                to_iteration: 2,
                scene: TransitionScene::new(
                    Vec::new(),
                    RenderBounds::default(),
                    RenderBounds::default(),
                    (0, None),
                    (0, None),
                    None,
                ),
                target,
            },
        )));
        let animation = app
            .iteration_animation
            .as_mut()
            .expect("accepted transition");
        assert!(animation.paused_at.is_some());
        let hidden_since = RenderInstant::now() - Duration::from_secs(10);
        animation.started_at = hidden_since;
        animation.paused_at = Some(hidden_since);

        let _ = app.update(Message::Frame);
        assert_eq!(app.iteration_animation.as_ref().unwrap().progress, 0.0);
        assert_eq!(app.render.displayed.as_ref().unwrap().iteration, 1);
        let _ = app.update(Message::CloseCatalog);
        let animation = app.iteration_animation.as_ref().unwrap();
        assert!(animation.paused_at.is_none());
        assert!(animation.paused_duration >= Duration::from_secs(10));
        let _ = app.update(Message::Frame);
        assert!(app.iteration_animation.is_some());
        assert_eq!(app.render.displayed.as_ref().unwrap().iteration, 1);

        let _ = app.update(Message::Viewport(ViewportMessage::GestureStarted));
        let _ = app.update(Message::Frame);
        let animation = app.iteration_animation.as_ref().unwrap();
        let progress = animation.progress;
        let paused_at = animation.paused_at;
        assert!(paused_at.is_some());
        let _ = app.update(Message::WindowResized(Size::new(1280.0, 820.0)));
        assert!(app.view_gesture_active);
        assert_eq!(
            app.iteration_animation.as_ref().unwrap().paused_at,
            paused_at
        );
        let _ = app.update(Message::Frame);
        assert_eq!(app.iteration_animation.as_ref().unwrap().progress, progress);
        assert_eq!(app.request_id, request);
    }

    #[test]
    fn panel_changes_preserve_the_scene_camera_and_manual_refinement_cancellation() {
        let (mut app, _) = BrakenGui::new();
        app.render_coordinator.cancel_current();
        app.render.active = None;
        let lines = test_line_scene(8);
        lines.set_refinement_progress_for_test(3);
        lines.cancel_refinement();
        let mut displayed = ready_result(app.request_id, app.system_view_epoch);
        displayed.line_scene = Some(Arc::clone(&lines));
        app.render.displayed = Some(displayed);
        let camera = Camera2d {
            zoom: 2.0,
            focus: [0.3, 0.7],
        };
        app.live_camera = camera;
        app.settled_camera = camera;
        let request = app.request_id;
        let cache_clock = app.generation_cache.clock;

        for message in [
            Message::ToggleSettings,
            Message::ToggleEditor(LayoutMode::Mobile),
            Message::CloseEditor,
            Message::OpenCatalog,
            Message::CloseCatalog,
            Message::WindowResized(Size::new(1280.0, 820.0)),
        ] {
            let _ = app.update(message);
            let retained = app
                .render
                .displayed
                .as_ref()
                .unwrap()
                .line_scene
                .as_ref()
                .unwrap();
            assert!(Arc::ptr_eq(retained, &lines));
            assert_eq!(lines.refinement_progress(), (3, 8));
            assert!(lines.refinement_was_cancelled());
            assert!(!lines.is_refining());
            assert_eq!(app.live_camera, camera);
            assert_eq!(app.settled_camera, camera);
            assert_eq!(app.request_id, request);
            assert_eq!(app.generation_cache.clock, cache_clock);
        }
    }
}
