//! Yew application state and browser coordination.
//!
//! Expensive grammar and visualization work is submitted to the persistent
//! Worker in `worker.rs`. This module keeps the last accepted scene visible,
//! rejects stale request IDs, and owns only DOM controls and display state.
//! Preset browsing and panel changes are display-only. Mobile grammar edits
//! remain a draft until applied; the mounted canvas retains its scene while
//! the catalog or editor occupies the screen.

use std::cell::Cell;
use std::rc::Rc;

use braken_viz::{Turtle2dConfig, VisualizerKind, targets::Palette};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{
    AddEventListenerOptions, Blob, BlobPropertyBag, Event, HtmlAnchorElement, HtmlCanvasElement,
    HtmlElement, HtmlInputElement, HtmlSelectElement, HtmlTextAreaElement, MediaQueryList,
    PointerEvent, ResizeObserver, TouchEvent, WheelEvent,
};
use yew::html::Scope;
use yew::prelude::*;

use crate::camera::{Camera2d, Orbit3d, ScreenPoint, ViewBounds, ViewportSize};
use crate::canvas::{CanvasDisplayScene, CanvasTransition, clear_canvas};
use crate::generated_preset_previews::preset_preview_svg;
use crate::model::{
    GrammarDraft, ThemePreference, adjacent_integer_angle, autorotation_after_motion_preference,
    display_refinement_is_running, formatted_angle, is_primary_mouse_button, next_iteration,
    normalized_search_terms, preset_autorotation_enabled, required_canvas_touches,
    source_uses_filled_turtle_polygons, touch_ownership,
};
use crate::orientation::OrientationAnchor;
use crate::presets::{self, Preset};
use crate::worker::{ClientRequest, WorkerClient, WorkerUpdate};
use crate::worker_protocol::{
    WorkerAmbiguousRules, WorkerFloatWidth, WorkerIterationTransition, WorkerRenderRequest,
    WorkerStrokeColor, WorkerTurtleConfig,
};

const SOURCE_EDIT_DEBOUNCE_MILLIS: i32 = 180;
const TRANSIENT_STATUS_DELAY_MILLIS: f64 = 250.0;
const SLOW_NOTICE_DELAY_MILLIS: f64 = 2_000.0;
const WHEEL_SETTLE_MILLIS: i32 = 150;
const TRANSITION_MILLIS: f64 = 600.0;
const VIEW_BUTTON_ZOOM_FACTOR: f64 = 1.25;
const VIEW_WHEEL_FACTOR: f64 = 1.15;
const VIEW_WHEEL_PIXELS_PER_LINE: f64 = 100.0;
const AUTOROTATE_RADIANS_PER_SECOND: f64 = std::f64::consts::PI / 15.0;
const MAX_AUTOROTATE_FRAME_MILLIS: f64 = 50.0;
const MIN_LINE_WIDTH_PERCENT: u16 = 5;
const MAX_LINE_WIDTH_PERCENT: u16 = 200;
const DEFAULT_LINE_WIDTH_PERCENT: u16 = 100;
const FEATURED_PRESETS: &[&str] = &[
    "Braken",
    "Koch Snowflake",
    "3D Hilbert Curve",
    "Orthogonal Virus",
    "Stochastic Plant",
    "Penrose Tiling",
];

struct PresetPreview {
    light: AttrValue,
    dark: AttrValue,
}

impl PresetPreview {
    fn for_theme(&self, dark: bool) -> AttrValue {
        if dark {
            self.dark.clone()
        } else {
            self.light.clone()
        }
    }
}

pub(crate) struct App {
    presets: Vec<Preset>,
    preset_previews: Vec<PresetPreview>,
    selected_preset: Option<usize>,
    preset_search: String,

    source: String,
    source_edit_token: u64,
    angle: f32,
    line_width_percent: u16,
    iterations: usize,
    iterations_input: String,
    iterations_notice: Option<String>,
    suggested_max_iterations: usize,
    seed_input: String,
    seed_placeholder: String,
    effective_seed: u64,
    seed_notice: Option<String>,
    float_width: WorkerFloatWidth,
    ambiguous_rules: WorkerAmbiguousRules,
    visualizer: VisualizerKind,
    turtle_config: Turtle2dConfig,
    orientation_anchor: Option<OrientationAnchor>,

    catalog_open: bool,
    editor_open: bool,
    editor_draft_mode: bool,
    editor_draft: GrammarDraft,
    adjustments_open: bool,
    focus_after_render: Option<&'static str>,
    scroll_after_render: Option<f64>,
    explore_scroll_y: f64,
    theme_preference: ThemePreference,
    system_dark: bool,
    reduced_motion: bool,
    viewport_width: f64,
    camera: Camera2d,
    orbit: Orbit3d,
    autorotate_active: bool,
    autorotate_last_frame: Option<f64>,
    pending_autorotate_epoch: Option<u64>,
    mouse_gesture: Option<MouseGesture>,
    touch_gesture: Option<TouchGesture>,
    touch_owned: Rc<Cell<bool>>,
    touch_navigation_enabled: Rc<Cell<bool>>,
    touch_required_count: Rc<Cell<usize>>,
    wheel_token: u64,
    wheel_active: bool,

    request_id: u64,
    system_view_epoch: u64,
    displayed_view_epoch: u64,
    active: Option<ActiveJob>,
    displayed: Option<Displayed>,
    iteration_route: Option<IterationRoute>,
    animation: Option<IterationAnimation>,
    failure: Option<String>,
    worker_fatal: Option<String>,
    worker: WorkerClient,

    base_canvas: NodeRef,
    exact_canvas: NodeRef,
    input_layer: NodeRef,
    last_canvas_size: Option<(i32, i32)>,
    canvas_dirty: bool,
    frame_scheduled: bool,
    frames_since_sample: u32,
    fps: u32,
    fps_sample_started: f64,

    export_sequence: u64,
    export_job: Option<ExportJob>,
    export_notice: Option<String>,

    media_query: Option<MediaQueryList>,
    media_listener: Option<Closure<dyn FnMut(Event)>>,
    reduced_motion_query: Option<MediaQueryList>,
    reduced_motion_listener: Option<Closure<dyn FnMut(Event)>>,
    resize_listener: Option<Closure<dyn FnMut(Event)>>,
    layout_observer: Option<ResizeObserver>,
    layout_listener: Option<Closure<dyn FnMut()>>,
    blur_listener: Option<Closure<dyn FnMut(Event)>>,
    touch_listeners: Option<TouchListeners>,
    touch_listener_attempted: bool,
}

#[derive(Clone)]
struct RenderIdentity {
    source: Rc<str>,
    angle: f32,
    visualizer: VisualizerKind,
    turtle_config: Turtle2dConfig,
    orientation_anchor: Option<OrientationAnchor>,
    seed: u64,
    float_width: WorkerFloatWidth,
    ambiguous_rules: WorkerAmbiguousRules,
}

impl RenderIdentity {
    fn same_as(&self, other: &Self) -> bool {
        self.source == other.source
            && self.angle.to_bits() == other.angle.to_bits()
            && self.visualizer == other.visualizer
            && turtle_configs_equal(&self.turtle_config, &other.turtle_config)
            && self.orientation_anchor == other.orientation_anchor
            && self.seed == other.seed
            && self.float_width == other.float_width
            && self.ambiguous_rules == other.ambiguous_rules
    }
}

struct Displayed {
    scene: Rc<CanvasDisplayScene>,
    iteration: usize,
    identity: RenderIdentity,
    view_epoch: u64,
    elapsed_millis: u64,
    backend: String,
    refinement_started: f64,
}

struct ActiveJob {
    request_id: u64,
    started: f64,
    progress: Option<ProgressSnapshot>,
}

struct ProgressSnapshot {
    phase: String,
    phase_completed: usize,
    phase_total: Option<usize>,
    completed_iterations: usize,
    total_iterations: usize,
    modules: usize,
    items: usize,
    elapsed_millis: u64,
}

#[derive(Clone)]
struct IterationRoute {
    request_id: u64,
    target_iteration: usize,
    identity: RenderIdentity,
}

struct IterationAnimation {
    transition: CanvasTransition,
    target: Displayed,
    from_iteration: usize,
    to_iteration: usize,
    started: f64,
    paused_at: Option<f64>,
    paused_millis: f64,
    progress: f32,
}

struct MouseGesture {
    pointer_id: i32,
    last: ScreenPoint,
    spatial: Option<SpatialGesture>,
}

#[derive(Clone, Copy)]
struct SpatialGesture {
    baseline_orbit: Orbit3d,
    initial_position: ScreenPoint,
}

#[derive(Clone, Copy)]
enum TouchGesture {
    TwoD {
        baseline_camera: Camera2d,
        initial_centroid: ScreenPoint,
        initial_distance: f64,
    },
    ThreeD(SpatialGesture),
}

struct TouchListeners {
    element: HtmlElement,
    start: Closure<dyn FnMut(TouchEvent)>,
    move_: Closure<dyn FnMut(TouchEvent)>,
    end: Closure<dyn FnMut(TouchEvent)>,
    cancel: Closure<dyn FnMut(TouchEvent)>,
}

#[derive(Clone, Copy)]
pub(crate) struct TouchPoint {
    identifier: i32,
    position: ScreenPoint,
}

#[derive(Clone, Copy)]
pub(crate) enum TouchPhase {
    Start,
    Move,
    End,
}

struct ExportJob {
    id: u64,
    cancelled: Rc<Cell<bool>>,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum Msg {
    SourceChanged(String),
    SourceDebounced(u64),
    AngleChanged(f32),
    AngleStep(f32),
    AngleWheel(f64),
    LineWidthChanged(u16),
    IterationsSlider(usize),
    IterationsInput(String),
    IterationsStep(bool),
    IterationsWheel(f64),
    SeedChanged(String),
    RandomizeSeed,
    FloatWidthChanged(WorkerFloatWidth),
    AmbiguousRulesChanged(WorkerAmbiguousRules),
    PresetSelected(usize),
    SearchChanged(String),
    ClearSearch,
    OpenCatalog,
    CloseCatalog,
    OpenEditor,
    CloseEditor,
    DraftChanged(String),
    ApplyDraft,
    ToggleAdjustments,
    CycleTheme,
    SystemTheme(bool),
    ReducedMotion(bool),
    WindowResized,
    CanvasResized,
    InputCancelled,
    Worker(WorkerUpdate),
    CancelWork,
    PointerDown(i32, ScreenPoint),
    PointerMove(i32, ScreenPoint),
    PointerUp(i32),
    Touch {
        phase: TouchPhase,
        points: Vec<TouchPoint>,
        owned: bool,
    },
    WheelZoom {
        factor: f64,
        anchor: ScreenPoint,
    },
    WheelSettled(u64),
    ZoomBy(f64),
    FitView,
    Frame(f64),
    CanvasFailed(String),
    StatusPulse(u64),
    ExportSvg,
    CancelExport,
    Exported {
        id: u64,
        result: Result<String, String>,
    },
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let presets = presets::get_presets();
        let preset_previews = presets
            .iter()
            .map(|preset| {
                let preview = |palette| {
                    let bytes = preset_preview_svg(&preset.name, palette).unwrap_or_else(|| {
                        panic!(
                            "{} is missing its generated {palette:?} preview",
                            preset.name
                        )
                    });
                    AttrValue::from(
                        std::str::from_utf8(bytes)
                            .expect("generated preset previews must be UTF-8 SVG")
                            .to_owned(),
                    )
                };
                PresetPreview {
                    light: preview(Palette::Light),
                    dark: preview(Palette::Dark),
                }
            })
            .collect();
        let default = presets
            .first()
            .cloned()
            .expect("the preset catalog must not be empty");
        let window = web_sys::window().expect("Yew requires a browser window");
        let viewport_width = window
            .inner_width()
            .ok()
            .and_then(|value| value.as_f64())
            .unwrap_or(1_280.0);
        let media_query = window
            .match_media("(prefers-color-scheme: dark)")
            .ok()
            .flatten();
        let system_dark = media_query.as_ref().is_some_and(MediaQueryList::matches);
        let reduced_motion_query = window
            .match_media("(prefers-reduced-motion: reduce)")
            .ok()
            .flatten();
        let reduced_motion = reduced_motion_query
            .as_ref()
            .is_some_and(MediaQueryList::matches);
        let effective_seed = getrandom::u64().unwrap_or(0);
        let worker = WorkerClient::new(ctx.link().callback(Msg::Worker));
        let pending_autorotate_epoch = preset_autorotation_enabled(
            default.visualizer == VisualizerKind::Turtle3d,
            reduced_motion,
        )
        .then_some(1);

        let mut app = Self {
            presets,
            preset_previews,
            selected_preset: Some(0),
            preset_search: String::new(),
            source: default.source,
            source_edit_token: 0,
            angle: default.angle as f32,
            line_width_percent: DEFAULT_LINE_WIDTH_PERCENT,
            iterations: default.iters as usize,
            iterations_input: default.iters.to_string(),
            iterations_notice: None,
            suggested_max_iterations: default.max_iters as usize,
            seed_input: String::new(),
            seed_placeholder: effective_seed.to_string(),
            effective_seed,
            seed_notice: None,
            float_width: WorkerFloatWidth::F32,
            ambiguous_rules: WorkerAmbiguousRules::Uniform,
            visualizer: default.visualizer,
            turtle_config: default.turtle_config,
            orientation_anchor: default.orientation_anchor,
            catalog_open: false,
            editor_open: false,
            editor_draft_mode: false,
            editor_draft: GrammarDraft::default(),
            adjustments_open: false,
            focus_after_render: None,
            scroll_after_render: None,
            explore_scroll_y: 0.0,
            theme_preference: ThemePreference::Auto,
            system_dark,
            reduced_motion,
            viewport_width,
            camera: Camera2d::fit(),
            orbit: Orbit3d::canonical(),
            autorotate_active: false,
            autorotate_last_frame: None,
            pending_autorotate_epoch,
            mouse_gesture: None,
            touch_gesture: None,
            touch_owned: Rc::new(Cell::new(false)),
            touch_navigation_enabled: Rc::new(Cell::new(false)),
            touch_required_count: Rc::new(Cell::new(2)),
            wheel_token: 0,
            wheel_active: false,
            request_id: 0,
            system_view_epoch: 1,
            displayed_view_epoch: 0,
            active: None,
            displayed: None,
            iteration_route: None,
            animation: None,
            failure: None,
            worker_fatal: None,
            worker,
            base_canvas: NodeRef::default(),
            exact_canvas: NodeRef::default(),
            input_layer: NodeRef::default(),
            last_canvas_size: None,
            canvas_dirty: true,
            frame_scheduled: false,
            frames_since_sample: 0,
            fps: 0,
            fps_sample_started: now_millis(),
            export_sequence: 0,
            export_job: None,
            export_notice: None,
            media_query,
            media_listener: None,
            reduced_motion_query,
            reduced_motion_listener: None,
            resize_listener: None,
            layout_observer: None,
            layout_listener: None,
            blur_listener: None,
            touch_listeners: None,
            touch_listener_attempted: false,
        };
        app.install_browser_listeners(ctx);
        app.begin_iteration_route(ctx);
        app
    }

    fn rendered(&mut self, ctx: &Context<Self>, _first_render: bool) {
        if self.canvas_visible() {
            self.resume_animation(ctx);
        } else {
            self.pause_animation();
            self.autorotate_last_frame = None;
            self.last_canvas_size = None;
        }
        if self.layout_observer.is_none()
            && let Some(element) = self.input_layer.cast::<HtmlElement>()
        {
            let link = ctx.link().clone();
            let listener = Closure::<dyn FnMut()>::new(move || {
                link.send_message(Msg::CanvasResized);
            });
            if let Ok(observer) = ResizeObserver::new(listener.as_ref().unchecked_ref()) {
                observer.observe(&element);
                self.layout_observer = Some(observer);
                self.layout_listener = Some(listener);
            }
        }
        self.refresh_canvas_size();
        if !self.touch_listener_attempted
            && let Some(element) = self.input_layer.cast::<HtmlElement>()
        {
            self.touch_listener_attempted = true;
            match TouchListeners::install(
                element,
                ctx.link().clone(),
                Rc::clone(&self.touch_owned),
                Rc::clone(&self.touch_navigation_enabled),
                Rc::clone(&self.touch_required_count),
            ) {
                Ok(listeners) => self.touch_listeners = Some(listeners),
                Err(error) => {
                    web_sys::console::error_1(&JsValue::from_str(&error));
                }
            }
        }
        if self.canvas_dirty && self.canvas_visible() {
            self.canvas_dirty = false;
            if let Err(error) = self.paint_current() {
                ctx.link()
                    .send_message(Msg::CanvasFailed(error.to_string()));
            }
        }
        if self.needs_animation_frame() {
            self.schedule_frame(ctx.link());
        }
        if let Some(id) = self.focus_after_render.take()
            && let Some(element) = web_sys::window()
                .and_then(|window| window.document())
                .and_then(|document| document.get_element_by_id(id))
                .and_then(|element| element.dyn_into::<HtmlElement>().ok())
        {
            let _ = element.focus();
        }
        if let Some(y) = self.scroll_after_render.take()
            && let Some(window) = web_sys::window()
        {
            window.scroll_to_with_x_and_y(0.0, y);
        }
    }

    fn update(&mut self, ctx: &Context<Self>, message: Self::Message) -> bool {
        match message {
            Msg::SourceChanged(source) => {
                if source == self.source {
                    return false;
                }
                self.source = source;
                self.selected_preset = None;
                self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                self.pending_autorotate_epoch = None;
                self.cancel_obsolete_work();
                self.source_edit_token = self.source_edit_token.wrapping_add(1);
                schedule_timeout(
                    ctx.link(),
                    Msg::SourceDebounced(self.source_edit_token),
                    SOURCE_EDIT_DEBOUNCE_MILLIS,
                );
                true
            }
            Msg::SourceDebounced(token) => {
                if token != self.source_edit_token {
                    return false;
                }
                self.begin_exact_render(ctx);
                true
            }
            Msg::AngleChanged(angle) => {
                self.angle = angle.clamp(0.0, 180.0);
                self.selected_preset = None;
                self.pending_autorotate_epoch = None;
                self.begin_exact_render(ctx);
                true
            }
            Msg::AngleStep(direction) => {
                self.angle = adjacent_integer_angle(self.angle, direction);
                self.selected_preset = None;
                self.pending_autorotate_epoch = None;
                self.begin_exact_render(ctx);
                true
            }
            Msg::AngleWheel(delta) => {
                if delta != 0.0 {
                    let direction = if delta.is_sign_negative() { 1.0 } else { -1.0 };
                    self.angle = (self.angle + direction).clamp(0.0, 180.0);
                    self.selected_preset = None;
                    self.pending_autorotate_epoch = None;
                    self.begin_exact_render(ctx);
                }
                true
            }
            Msg::LineWidthChanged(percent) => {
                let percent = percent.clamp(MIN_LINE_WIDTH_PERCENT, MAX_LINE_WIDTH_PERCENT);
                if percent == self.line_width_percent {
                    return false;
                }
                self.line_width_percent = percent;
                self.cancel_export_for_replacement();
                self.invalidate_display();
                true
            }
            Msg::IterationsSlider(iterations) => {
                self.set_iterations(iterations);
                self.selected_preset = None;
                self.pending_autorotate_epoch = None;
                self.begin_iteration_route(ctx);
                true
            }
            Msg::IterationsInput(input) => {
                self.iterations_input = input;
                if self.iterations_input.is_empty() {
                    self.iterations_notice = None;
                    return true;
                }
                match self.iterations_input.parse::<usize>() {
                    Ok(iterations) => {
                        self.iterations = iterations;
                        self.iterations_notice = None;
                        self.selected_preset = None;
                        self.pending_autorotate_epoch = None;
                        self.begin_exact_render(ctx);
                    }
                    Err(_) => {
                        self.iterations_notice = Some(String::from(
                            "Iterations must be a non-negative whole number",
                        ));
                    }
                }
                true
            }
            Msg::IterationsStep(increment) => {
                let iterations = if increment {
                    self.iterations.saturating_add(1)
                } else {
                    self.iterations.saturating_sub(1)
                };
                self.set_iterations(iterations);
                self.selected_preset = None;
                self.pending_autorotate_epoch = None;
                self.begin_iteration_route(ctx);
                true
            }
            Msg::IterationsWheel(delta) => {
                if delta != 0.0 {
                    let iterations = if delta.is_sign_negative() {
                        self.iterations.saturating_add(1)
                    } else {
                        self.iterations.saturating_sub(1)
                    };
                    self.set_iterations(iterations);
                    self.selected_preset = None;
                    self.pending_autorotate_epoch = None;
                    self.begin_iteration_route(ctx);
                }
                true
            }
            Msg::SeedChanged(seed) => {
                self.seed_input = seed;
                if self.seed_input.is_empty() {
                    if let Ok(seed) = self.seed_placeholder.parse::<u64>() {
                        self.effective_seed = seed;
                    }
                    self.seed_notice = None;
                    self.begin_exact_render(ctx);
                } else {
                    match self.seed_input.parse::<u64>() {
                        Ok(seed) => {
                            self.effective_seed = seed;
                            self.seed_notice = None;
                            self.begin_exact_render(ctx);
                        }
                        Err(_) => {
                            self.seed_notice = Some(String::from(
                                "Seed must be a decimal integer from 0 to 18446744073709551615",
                            ));
                        }
                    }
                }
                true
            }
            Msg::RandomizeSeed => {
                self.regenerate_seed();
                self.begin_exact_render(ctx);
                true
            }
            Msg::FloatWidthChanged(float_width) => {
                if self.float_width == float_width {
                    return false;
                }
                self.float_width = float_width;
                self.begin_exact_render(ctx);
                true
            }
            Msg::AmbiguousRulesChanged(ambiguous_rules) => {
                if self.ambiguous_rules == ambiguous_rules {
                    return false;
                }
                self.ambiguous_rules = ambiguous_rules;
                self.begin_exact_render(ctx);
                true
            }
            Msg::PresetSelected(index) => {
                self.select_preset(index, ctx);
                self.editor_draft.reset(&self.source);
                if self.catalog_open {
                    self.catalog_open = false;
                    self.canvas_dirty = true;
                }
                if self.is_mobile() {
                    self.focus_after_render = Some("drawing-input");
                    self.scroll_after_render = Some(0.0);
                }
                true
            }
            Msg::SearchChanged(search) => {
                self.preset_search = search;
                true
            }
            Msg::ClearSearch => {
                self.preset_search.clear();
                true
            }
            Msg::OpenCatalog => {
                self.remember_explore_scroll();
                self.editor_open = false;
                self.catalog_open = true;
                self.focus_after_render = Some("presets-heading");
                self.scroll_after_render = Some(0.0);
                self.pause_animation();
                self.autorotate_last_frame = None;
                true
            }
            Msg::CloseCatalog => {
                self.catalog_open = false;
                self.focus_after_render = Some("browse-presets");
                self.scroll_after_render = Some(self.explore_scroll_y);
                self.resume_animation(ctx);
                self.canvas_dirty = true;
                true
            }
            Msg::OpenEditor => {
                self.remember_explore_scroll();
                self.catalog_open = false;
                self.editor_draft_mode = self.is_mobile() || self.editor_draft.is_modified();
                if self.editor_draft_mode {
                    self.editor_draft.open(&self.source);
                }
                self.editor_open = true;
                self.focus_after_render = Some("source-editor");
                if self.is_mobile() {
                    self.scroll_after_render = Some(0.0);
                    self.pause_animation();
                    self.autorotate_last_frame = None;
                }
                true
            }
            Msg::CloseEditor => {
                self.close_editor(ctx);
                true
            }
            Msg::DraftChanged(source) => {
                self.editor_draft.edit(source);
                true
            }
            Msg::ApplyDraft => {
                if let Some(source) = self.editor_draft.apply(&self.source) {
                    self.source = source;
                    self.selected_preset = None;
                    self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
                    self.pending_autorotate_epoch = None;
                    self.begin_exact_render(ctx);
                }
                self.close_editor(ctx);
                if self.is_mobile() {
                    self.focus_after_render = Some("drawing-input");
                    self.scroll_after_render = Some(0.0);
                }
                true
            }
            Msg::ToggleAdjustments => {
                self.adjustments_open = !self.adjustments_open;
                true
            }
            Msg::CycleTheme => {
                let was_dark = self.effective_dark();
                self.theme_preference = self.theme_preference.next();
                if was_dark != self.effective_dark() {
                    self.invalidate_display();
                }
                true
            }
            Msg::SystemTheme(dark) => {
                let was_dark = self.effective_dark();
                self.system_dark = dark;
                if was_dark != self.effective_dark() {
                    self.invalidate_display();
                    return true;
                }
                false
            }
            Msg::ReducedMotion(reduced_motion) => {
                if reduced_motion == self.reduced_motion {
                    return false;
                }
                self.reduced_motion = reduced_motion;
                if !reduced_motion {
                    // Turning the preference back off must not resume an old
                    // preset's rotation. A subsequent 3D preset selection is
                    // the only event that arms autorotation again.
                    return false;
                }
                let was_autorotating = self.autorotate_active;
                let had_pending_autorotation = self.pending_autorotate_epoch.is_some();
                (self.autorotate_active, self.pending_autorotate_epoch) =
                    autorotation_after_motion_preference(
                        self.autorotate_active,
                        self.pending_autorotate_epoch,
                        reduced_motion,
                    );
                self.autorotate_last_frame = None;
                if was_autorotating {
                    // Exact display work is paused while the scene turns. Give
                    // the now-stationary view a fresh refinement pass and slow-
                    // work timer when motion reduction stops that rotation.
                    self.restart_refinement();
                    self.canvas_dirty = true;
                }
                was_autorotating || had_pending_autorotation
            }
            Msg::WindowResized => {
                self.touch_gesture = None;
                self.viewport_width = web_sys::window()
                    .and_then(|window| window.inner_width().ok())
                    .and_then(|value| value.as_f64())
                    .unwrap_or(self.viewport_width);
                if self.is_mobile() && self.editor_open && !self.editor_draft_mode {
                    self.editor_draft_mode = true;
                    self.editor_draft.open(&self.source);
                }
                self.invalidate_display();
                true
            }
            Msg::CanvasResized => self.refresh_canvas_size(),
            Msg::InputCancelled => {
                let was_interacting = self.interaction_active();
                self.mouse_gesture = None;
                self.touch_gesture = None;
                self.touch_owned.set(false);
                self.wheel_active = false;
                self.wheel_token = self.wheel_token.wrapping_add(1);
                if was_interacting {
                    self.restart_refinement();
                    self.resume_animation(ctx);
                    self.canvas_dirty = true;
                }
                was_interacting
            }
            Msg::Worker(update) => self.handle_worker_update(ctx, update),
            Msg::CancelWork => {
                self.cancel_visible_work();
                true
            }
            Msg::PointerDown(pointer_id, position) => {
                if !self.is_mobile() && self.navigation_enabled() {
                    let spatial = self.spatial_navigation().then_some(SpatialGesture {
                        baseline_orbit: self.orbit,
                        initial_position: position,
                    });
                    self.mouse_gesture = Some(MouseGesture {
                        pointer_id,
                        last: position,
                        spatial,
                    });
                    // Any owned canvas gesture is authoritative, including a
                    // planar scene that remains visible while a newly selected
                    // 3D preset is still rendering in the Worker.
                    self.stop_autorotate();
                    self.pause_animation();
                }
                false
            }
            Msg::PointerMove(pointer_id, position) => {
                let (last, spatial) = {
                    let Some(gesture) = self
                        .mouse_gesture
                        .as_mut()
                        .filter(|gesture| gesture.pointer_id == pointer_id)
                    else {
                        return false;
                    };
                    let last = gesture.last;
                    gesture.last = position;
                    (last, gesture.spatial)
                };
                if let Some(gesture) = spatial {
                    if let Some(viewport) = self.viewport_size() {
                        self.orbit = gesture.baseline_orbit.arcball_drag(
                            gesture.initial_position,
                            position,
                            viewport,
                        );
                        self.invalidate_display();
                        return true;
                    }
                    return false;
                }
                let delta = ScreenPoint::new(position.x - last.x, position.y - last.y);
                if let Some((bounds, viewport)) = self.view_geometry() {
                    self.camera = self.camera.pan_by_pixels(delta, bounds, viewport);
                    self.invalidate_display();
                    return true;
                }
                false
            }
            Msg::PointerUp(pointer_id) => {
                if self
                    .mouse_gesture
                    .as_ref()
                    .is_some_and(|gesture| gesture.pointer_id == pointer_id)
                {
                    self.mouse_gesture = None;
                    self.resume_animation(ctx);
                    self.restart_refinement();
                    self.canvas_dirty = true;
                    return true;
                }
                false
            }
            Msg::Touch {
                phase,
                points,
                owned,
            } => self.update_touch(ctx, phase, &points, owned),
            Msg::WheelZoom { factor, anchor } => {
                if self.spatial_navigation() {
                    return false;
                }
                if let Some((bounds, viewport)) = self.view_geometry() {
                    self.stop_autorotate();
                    self.camera = self.camera.zoom_about(factor, anchor, bounds, viewport);
                    self.wheel_active = true;
                    self.wheel_token = self.wheel_token.wrapping_add(1);
                    self.pause_animation();
                    self.canvas_dirty = true;
                    schedule_timeout(
                        ctx.link(),
                        Msg::WheelSettled(self.wheel_token),
                        WHEEL_SETTLE_MILLIS,
                    );
                    return true;
                }
                false
            }
            Msg::WheelSettled(token) => {
                if token != self.wheel_token {
                    return false;
                }
                self.wheel_active = false;
                self.restart_refinement();
                self.resume_animation(ctx);
                self.canvas_dirty = true;
                true
            }
            Msg::ZoomBy(factor) => {
                if self.spatial_navigation() {
                    return false;
                }
                if let Some((bounds, viewport)) = self.view_geometry() {
                    self.camera = self.camera.zoom_by_in(factor, bounds, viewport);
                    self.restart_refinement();
                    self.canvas_dirty = true;
                    return true;
                }
                false
            }
            Msg::FitView => {
                if self.spatial_navigation() {
                    return false;
                }
                self.camera = Camera2d::fit();
                self.restart_refinement();
                self.canvas_dirty = true;
                true
            }
            Msg::Frame(timestamp) => self.update_frame(ctx, timestamp),
            Msg::CanvasFailed(error) => {
                self.record_canvas_failure(error);
                true
            }
            Msg::StatusPulse(request_id) => {
                self.active
                    .as_ref()
                    .is_some_and(|job| job.request_id == request_id)
                    || self.displayed.is_some()
            }
            Msg::ExportSvg => {
                self.start_export(ctx);
                true
            }
            Msg::CancelExport => {
                if let Some(job) = self.export_job.take() {
                    job.cancelled.set(true);
                    self.export_notice = Some(String::from("SVG export cancelled"));
                }
                true
            }
            Msg::Exported { id, result } => {
                if self.export_job.as_ref().is_none_or(|job| job.id != id) {
                    return false;
                }
                self.export_job = None;
                match result {
                    Ok(svg) => match download_svg(&svg) {
                        Ok(()) => self.export_notice = Some(String::from("Downloaded lsystem.svg")),
                        Err(error) => {
                            self.export_notice = Some(format!("SVG export failed: {error}"));
                        }
                    },
                    Err(error) if error == "cancelled" => {
                        self.export_notice = Some(String::from("SVG export cancelled"));
                    }
                    Err(error) => {
                        self.export_notice = Some(format!("SVG export failed: {error}"));
                    }
                }
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let root_class = classes!(
            "braken-app",
            if self.effective_dark() {
                "theme-dark"
            } else {
                "theme-light"
            }
        );
        let workspace_class = classes!(
            "workspace",
            self.catalog_open.then_some("catalog-open"),
            self.editor_open.then_some("editor-open"),
            self.adjustments_open.then_some("adjustments-open"),
        );
        let editor_open = self.editor_open;
        let catalog_open = self.catalog_open;
        let onkeydown = ctx.link().batch_callback(move |event: KeyboardEvent| {
            if event.key() != "Escape" {
                return None;
            }
            let message = if editor_open {
                Msg::CloseEditor
            } else if catalog_open {
                Msg::CloseCatalog
            } else {
                return None;
            };
            event.prevent_default();
            Some(message)
        });
        html! {
            <div class={root_class}>
                if let Some(message) = &self.worker_fatal {
                    <div class="fatal-banner" role="alert">
                        <strong>{"Browser Worker error"}</strong>
                        <span>{message}</span>
                    </div>
                }
                <div class={workspace_class} {onkeydown}>
                    {self.view_header(ctx)}
                    {self.view_canvas(ctx)}
                    {self.view_featured_presets(ctx)}
                    {self.view_editor(ctx)}
                    {self.view_control_bar(ctx)}
                    {self.view_adjustments(ctx)}
                    {self.view_presets(ctx)}
                </div>
            </div>
        }
    }

    fn destroy(&mut self, _ctx: &Context<Self>) {
        if let Some(observer) = &self.layout_observer {
            observer.disconnect();
        }
        if let Some(query) = &self.media_query
            && let Some(listener) = &self.media_listener
        {
            let _ = query
                .remove_event_listener_with_callback("change", listener.as_ref().unchecked_ref());
        }
        if let Some(query) = &self.reduced_motion_query
            && let Some(listener) = &self.reduced_motion_listener
        {
            let _ = query
                .remove_event_listener_with_callback("change", listener.as_ref().unchecked_ref());
        }
        if let Some(window) = web_sys::window()
            && let Some(listener) = &self.resize_listener
        {
            let _ = window
                .remove_event_listener_with_callback("resize", listener.as_ref().unchecked_ref());
        }
        if let Some(window) = web_sys::window()
            && let Some(listener) = &self.blur_listener
        {
            let _ = window
                .remove_event_listener_with_callback("blur", listener.as_ref().unchecked_ref());
        }
        if let Some(job) = self.export_job.take() {
            job.cancelled.set(true);
        }
        if let Some(listeners) = self.touch_listeners.take() {
            listeners.remove();
        }
        self.touch_owned.set(false);
    }
}

impl App {
    fn remember_explore_scroll(&mut self) {
        self.explore_scroll_y = web_sys::window()
            .and_then(|window| window.scroll_y().ok())
            .unwrap_or(0.0);
    }

    fn close_editor(&mut self, ctx: &Context<Self>) {
        self.editor_open = false;
        self.focus_after_render = Some(if self.is_mobile() {
            "mobile-edit-grammar"
        } else {
            "desktop-edit-grammar"
        });
        if self.is_mobile() {
            self.scroll_after_render = Some(self.explore_scroll_y);
        }
        self.resume_animation(ctx);
        self.canvas_dirty = true;
    }

    fn canvas_visible(&self) -> bool {
        !(self.is_mobile() && (self.catalog_open || self.editor_open))
    }

    fn refresh_canvas_size(&mut self) -> bool {
        if !self.canvas_visible() {
            return false;
        }
        let Some(element) = self.input_layer.cast::<HtmlElement>() else {
            return false;
        };
        let size = (element.client_width(), element.client_height());
        // Hidden panels must not fit the camera to a zero-sized viewport.
        if size.0 <= 0 || size.1 <= 0 || self.last_canvas_size == Some(size) {
            return false;
        }
        self.last_canvas_size = Some(size);
        self.constrain_camera();
        self.invalidate_display();
        true
    }

    fn install_browser_listeners(&mut self, ctx: &Context<Self>) {
        if let Some(query) = &self.media_query {
            let link = ctx.link().clone();
            let listener = Closure::wrap(Box::new(move |event: Event| {
                let dark = event
                    .dyn_ref::<web_sys::MediaQueryListEvent>()
                    .is_some_and(web_sys::MediaQueryListEvent::matches);
                link.send_message(Msg::SystemTheme(dark));
            }) as Box<dyn FnMut(Event)>);
            let _ =
                query.add_event_listener_with_callback("change", listener.as_ref().unchecked_ref());
            self.media_listener = Some(listener);
        }

        if let Some(query) = &self.reduced_motion_query {
            let link = ctx.link().clone();
            let listener = Closure::wrap(Box::new(move |event: Event| {
                let reduced_motion = event
                    .dyn_ref::<web_sys::MediaQueryListEvent>()
                    .is_some_and(web_sys::MediaQueryListEvent::matches);
                link.send_message(Msg::ReducedMotion(reduced_motion));
            }) as Box<dyn FnMut(Event)>);
            let _ =
                query.add_event_listener_with_callback("change", listener.as_ref().unchecked_ref());
            self.reduced_motion_listener = Some(listener);
        }

        if let Some(window) = web_sys::window() {
            let link = ctx.link().clone();
            let listener = Closure::wrap(Box::new(move |_event: Event| {
                link.send_message(Msg::WindowResized);
            }) as Box<dyn FnMut(Event)>);
            let _ = window
                .add_event_listener_with_callback("resize", listener.as_ref().unchecked_ref());
            self.resize_listener = Some(listener);

            let link = ctx.link().clone();
            let listener = Closure::wrap(Box::new(move |_event: Event| {
                link.send_message(Msg::InputCancelled);
            }) as Box<dyn FnMut(Event)>);
            let _ =
                window.add_event_listener_with_callback("blur", listener.as_ref().unchecked_ref());
            self.blur_listener = Some(listener);
        }
    }

    fn effective_dark(&self) -> bool {
        self.theme_preference.resolve_dark(self.system_dark)
    }

    fn line_width_scale(&self) -> f64 {
        f64::from(self.line_width_percent) / f64::from(DEFAULT_LINE_WIDTH_PERCENT)
    }

    fn is_mobile(&self) -> bool {
        self.viewport_width < 700.0
    }

    fn set_iterations(&mut self, iterations: usize) {
        self.iterations = iterations;
        self.iterations_input = iterations.to_string();
        self.iterations_notice = None;
    }

    fn regenerate_seed(&mut self) {
        match getrandom::u64() {
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

    fn select_preset(&mut self, index: usize, ctx: &Context<Self>) {
        let Some(preset) = self.presets.get(index).cloned() else {
            return;
        };
        self.system_view_epoch = self.system_view_epoch.wrapping_add(1);
        self.pending_autorotate_epoch = preset_autorotation_enabled(
            preset.visualizer == VisualizerKind::Turtle3d,
            self.reduced_motion,
        )
        .then_some(self.system_view_epoch);
        self.source = preset.source;
        self.set_iterations(preset.iters as usize);
        self.suggested_max_iterations = preset.max_iters as usize;
        self.visualizer = preset.visualizer;
        self.turtle_config = preset.turtle_config;
        self.orientation_anchor = preset.orientation_anchor;
        self.angle = preset.angle as f32;
        self.selected_preset = Some(index);
        self.regenerate_seed();
        self.begin_iteration_route(ctx);
    }

    fn current_identity(&self) -> RenderIdentity {
        RenderIdentity {
            source: Rc::from(self.source.as_str()),
            angle: self.angle,
            visualizer: self.visualizer,
            turtle_config: self.turtle_config.clone(),
            orientation_anchor: self.orientation_anchor,
            seed: self.effective_seed,
            float_width: self.float_width,
            ambiguous_rules: self.ambiguous_rules,
        }
    }

    fn begin_exact_render(&mut self, ctx: &Context<Self>) {
        self.source_edit_token = self.source_edit_token.wrapping_add(1);
        self.iteration_route = None;
        self.discard_animation();
        self.worker.cancel_current();
        self.request_id = self.request_id.wrapping_add(1);
        self.queue_render(ctx, self.request_id, self.iterations, None);
    }

    fn begin_iteration_route(&mut self, ctx: &Context<Self>) {
        self.source_edit_token = self.source_edit_token.wrapping_add(1);
        if self.visualizer != VisualizerKind::Turtle2d {
            self.begin_exact_render(ctx);
            return;
        }
        self.discard_animation();
        self.worker.cancel_current();
        // The old job is obsolete even when the requested iteration already
        // matches the last exact scene and no replacement needs to be queued.
        self.active = None;
        self.request_id = self.request_id.wrapping_add(1);
        let identity = self.current_identity();
        self.iteration_route = Some(IterationRoute {
            request_id: self.request_id,
            target_iteration: self.iterations,
            identity: identity.clone(),
        });
        let compatible = self
            .displayed
            .as_ref()
            .is_some_and(|displayed| displayed.identity.same_as(&identity));
        if compatible {
            self.queue_next_iteration(ctx);
        } else {
            self.queue_render(ctx, self.request_id, 0, None);
        }
    }

    fn queue_next_iteration(&mut self, ctx: &Context<Self>) {
        let Some(route) = self.iteration_route.clone() else {
            return;
        };
        if route.request_id != self.request_id {
            self.iteration_route = None;
            return;
        }
        let Some(displayed) = self.displayed.as_ref() else {
            return;
        };
        if !displayed.identity.same_as(&route.identity) {
            self.iteration_route = None;
            return;
        }
        let from_iteration = displayed.iteration;
        let Some(to_iteration) = next_iteration(from_iteration, route.target_iteration) else {
            self.iteration_route = None;
            return;
        };
        let transition = (!source_uses_filled_turtle_polygons(&route.identity.source)).then_some(
            WorkerIterationTransition {
                from_iteration,
                to_iteration,
            },
        );
        self.queue_render(ctx, route.request_id, to_iteration, transition);
    }

    fn queue_render(
        &mut self,
        ctx: &Context<Self>,
        request_id: u64,
        iteration: usize,
        transition: Option<WorkerIterationTransition>,
    ) {
        self.cancel_export_for_replacement();
        self.suspend_display_refinement();
        self.failure = None;
        let started = now_millis();
        self.active = Some(ActiveJob {
            request_id,
            started,
            progress: None,
        });
        let turtle = &self.turtle_config;
        let wire = WorkerRenderRequest {
            request_id,
            source: self.source.clone(),
            ir_json: None,
            iterations: iteration,
            angle: self.angle,
            visualizer: self.visualizer.to_string(),
            turtle: WorkerTurtleConfig {
                initial_angle: turtle.initial_angle,
                default_step: turtle.default_step,
                scale_multiplier: turtle.scale_multiplier,
                initial_width: turtle.initial_width,
                width_increment: turtle.width_increment,
                turn_angle_increment: turtle.turn_angle_increment,
                initial_color: WorkerStrokeColor::from(turtle.initial_color),
                color_increment: turtle.color_increment,
                palette: turtle.palette.clone(),
                background: turtle.background,
                draw_modules: turtle.draw_modules.clone(),
                move_modules: turtle.move_modules.clone(),
                module_aliases: turtle.module_aliases.clone(),
            },
            orientation_anchor: self
                .orientation_anchor
                .map(OrientationAnchor::as_str)
                .map(str::to_owned),
            seed: self.effective_seed,
            float_width: self.float_width,
            ambiguous_rules: self.ambiguous_rules,
            transition,
        };
        self.worker.submit(ClientRequest {
            wire,
            view_epoch: self.system_view_epoch,
            iteration,
        });
        schedule_timeout(
            ctx.link(),
            Msg::StatusPulse(request_id),
            TRANSIENT_STATUS_DELAY_MILLIS as i32,
        );
        schedule_timeout(
            ctx.link(),
            Msg::StatusPulse(request_id),
            SLOW_NOTICE_DELAY_MILLIS as i32,
        );
    }

    fn cancel_obsolete_work(&mut self) {
        self.cancel_export_for_replacement();
        self.suspend_display_refinement();
        self.request_id = self.request_id.wrapping_add(1);
        self.worker.cancel_current();
        self.active = None;
        self.iteration_route = None;
        self.discard_animation();
        self.failure = None;
    }

    fn cancel_visible_work(&mut self) {
        if self.active.take().is_some() {
            self.worker.cancel_current();
            self.request_id = self.request_id.wrapping_add(1);
            self.iteration_route = None;
            self.animation = None;
            self.failure = None;
        } else if self.animation.take().is_some() {
            self.request_id = self.request_id.wrapping_add(1);
            self.iteration_route = None;
            self.canvas_dirty = true;
        } else if let Some(displayed) = &self.displayed {
            displayed.scene.cancel_refinement();
        }
    }

    fn handle_worker_update(&mut self, ctx: &Context<Self>, update: WorkerUpdate) -> bool {
        match update {
            WorkerUpdate::Ready => {
                self.worker_fatal = None;
                false
            }
            WorkerUpdate::Progress {
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
                let Some(active) = self
                    .active
                    .as_mut()
                    .filter(|active| active.request_id == request_id)
                else {
                    return false;
                };
                active.progress = Some(ProgressSnapshot {
                    phase,
                    phase_completed,
                    phase_total,
                    completed_iterations,
                    total_iterations,
                    modules,
                    items,
                    elapsed_millis,
                });
                true
            }
            WorkerUpdate::Cancelled { request_id } => {
                if request_id != self.request_id {
                    return false;
                }
                self.active = None;
                self.iteration_route = None;
                self.animation = None;
                self.failure = None;
                true
            }
            WorkerUpdate::Fatal(message) => {
                self.worker_fatal = Some(message);
                true
            }
            WorkerUpdate::Finished {
                request,
                result,
                lines,
                morphs,
            } => {
                if request.wire.request_id != self.request_id {
                    return false;
                }
                self.active = None;
                let result = match result {
                    Ok(result) => result,
                    Err(message) => {
                        self.failure = Some(message);
                        self.iteration_route = None;
                        self.animation = None;
                        return true;
                    }
                };
                let Some(lines) = lines else {
                    self.failure = Some(String::from(
                        "browser-worker result omitted its line buffers",
                    ));
                    self.iteration_route = None;
                    return true;
                };
                let transition_metadata = result.transition.clone();
                let elapsed_millis = result.elapsed_millis;
                let backend = format!(
                    "{} derive · {} visualize · Canvas2D",
                    result.derivation_backend, result.visualization_backend
                );
                let scene = match CanvasDisplayScene::from_worker(result, lines) {
                    Ok(scene) => Rc::new(scene),
                    Err(error) => {
                        self.failure = Some(error.to_string());
                        self.iteration_route = None;
                        return true;
                    }
                };
                let target = Displayed {
                    scene: Rc::clone(&scene),
                    iteration: request.iteration,
                    identity: self.current_identity(),
                    view_epoch: request.view_epoch,
                    elapsed_millis,
                    backend,
                    refinement_started: now_millis(),
                };

                if let Some(metadata) = transition_metadata {
                    let Some(planar_scene) = scene.as_2d() else {
                        self.failure = Some(String::from(
                            "browser-worker returned transition data for a 3D scene",
                        ));
                        self.iteration_route = None;
                        return true;
                    };
                    let Some(morphs) = morphs else {
                        self.failure = Some(String::from(
                            "browser-worker transition omitted its morph buffers",
                        ));
                        self.iteration_route = None;
                        return true;
                    };
                    match CanvasTransition::from_worker(metadata.clone(), morphs, planar_scene) {
                        Ok(transition) => {
                            self.animation = Some(IterationAnimation {
                                transition,
                                target,
                                from_iteration: metadata.from_iteration,
                                to_iteration: metadata.to_iteration,
                                started: now_millis(),
                                paused_at: None,
                                paused_millis: 0.0,
                                progress: 0.0,
                            });
                            self.failure = None;
                            self.canvas_dirty = true;
                            self.schedule_frame(ctx.link());
                            true
                        }
                        Err(error) => {
                            self.failure = Some(error.to_string());
                            self.iteration_route = None;
                            true
                        }
                    }
                } else {
                    self.promote_displayed(target);
                    self.queue_next_iteration(ctx);
                    true
                }
            }
        }
    }

    fn promote_displayed(&mut self, mut displayed: Displayed) {
        let spatial = displayed.scene.is_spatial();
        let displayed_spatial = self
            .displayed
            .as_ref()
            .is_some_and(|current| current.scene.is_spatial());
        let view_replaced = displayed.view_epoch != self.displayed_view_epoch
            || self.displayed.is_some() && spatial != displayed_spatial;
        if view_replaced {
            self.camera = Camera2d::fit();
            self.orbit = Orbit3d::canonical();
            self.autorotate_active = preset_autorotation_enabled(spatial, self.reduced_motion)
                && self.pending_autorotate_epoch == Some(displayed.view_epoch);
            self.autorotate_last_frame = None;
            self.pending_autorotate_epoch = None;
            self.displayed_view_epoch = displayed.view_epoch;
            self.touch_gesture = None;
            self.wheel_active = false;
        } else {
            self.touch_gesture = None;
            if let Some(bounds) = displayed.scene.bounds_2d() {
                self.constrain_camera_for(bounds);
            }
        }
        displayed.refinement_started = now_millis();
        self.displayed = Some(displayed);
        self.failure = None;
        self.canvas_dirty = true;
    }

    fn navigation_enabled(&self) -> bool {
        self.animation.is_some()
            || self
                .displayed
                .as_ref()
                .is_some_and(|displayed| displayed.scene.navigation_enabled())
    }

    fn spatial_navigation(&self) -> bool {
        self.animation.is_none()
            && self.displayed.as_ref().is_some_and(|displayed| {
                displayed.scene.is_spatial() && displayed.scene.navigation_enabled()
            })
    }

    fn viewport_size(&self) -> Option<ViewportSize> {
        let element = self.input_layer.cast::<HtmlElement>()?;
        if element.client_width() <= 0 || element.client_height() <= 0 {
            return None;
        }
        Some(ViewportSize::new(
            f64::from(element.client_width()),
            f64::from(element.client_height()),
        ))
    }

    fn view_geometry(&self) -> Option<(ViewBounds, ViewportSize)> {
        let bounds = self.displayed.as_ref()?.scene.bounds_2d()?;
        Some((bounds, self.viewport_size()?))
    }

    fn constrain_camera(&mut self) {
        let Some((bounds, viewport)) = self.view_geometry() else {
            return;
        };
        self.camera = self.camera.constrained(bounds, viewport);
    }

    fn constrain_camera_for(&mut self, bounds: ViewBounds) {
        let Some(viewport) = self.viewport_size() else {
            return;
        };
        self.camera = self.camera.constrained(bounds, viewport);
    }

    fn stop_autorotate(&mut self) {
        self.autorotate_active = false;
        self.autorotate_last_frame = None;
        // A preset replacement may still be rendering in the Worker while the
        // previous spatial scene remains interactive. Treat that gesture as
        // authoritative too: the arriving preset must not unexpectedly begin
        // turning again after the user has already taken control.
        self.pending_autorotate_epoch = None;
    }

    fn update_touch(
        &mut self,
        ctx: &Context<Self>,
        phase: TouchPhase,
        points: &[TouchPoint],
        owned: bool,
    ) -> bool {
        if !owned {
            return false;
        }

        // Once a two-finger gesture is captured, its remaining finger stays
        // owned until the last lift. This prevents a partially completed
        // pinch from turning into a one-finger page scroll.
        self.touch_owned.set(!points.is_empty());
        if points.is_empty() {
            self.touch_gesture = None;
            self.restart_refinement();
            self.resume_animation(ctx);
            self.canvas_dirty = true;
            return true;
        }
        if !self.navigation_enabled() {
            self.touch_gesture = None;
            return false;
        }
        let required_touches = required_canvas_touches(self.is_mobile(), self.spatial_navigation());
        if points.len() < required_touches {
            self.touch_gesture = None;
            return false;
        }
        let (centroid, distance) = touch_measurement(points);
        if self.touch_gesture.is_none() || !matches!(phase, TouchPhase::Move) {
            // This touch sequence has crossed the ownership threshold. It
            // cancels pending autorotation even if the outgoing scene is 2D.
            self.stop_autorotate();
            if self.spatial_navigation() {
                self.touch_gesture = Some(TouchGesture::ThreeD(SpatialGesture {
                    baseline_orbit: self.orbit,
                    initial_position: centroid,
                }));
            } else {
                self.touch_gesture = Some(TouchGesture::TwoD {
                    baseline_camera: self.camera,
                    initial_centroid: centroid,
                    initial_distance: distance,
                });
            }
            self.pause_animation();
            return false;
        }
        let Some(gesture) = self.touch_gesture else {
            return false;
        };
        match gesture {
            TouchGesture::ThreeD(gesture) => {
                let Some(viewport) = self.viewport_size() else {
                    return false;
                };
                self.orbit = gesture.baseline_orbit.arcball_drag(
                    gesture.initial_position,
                    centroid,
                    viewport,
                );
            }
            TouchGesture::TwoD {
                baseline_camera,
                initial_centroid,
                initial_distance,
            } => {
                let Some((bounds, viewport)) = self.view_geometry() else {
                    return false;
                };
                let scale = if initial_distance > f64::EPSILON {
                    distance / initial_distance
                } else {
                    1.0
                };
                self.camera =
                    baseline_camera.pinch(initial_centroid, centroid, scale, bounds, viewport);
            }
        }
        self.canvas_dirty = true;
        true
    }

    fn interaction_active(&self) -> bool {
        self.mouse_gesture.is_some() || self.touch_owned.get() || self.wheel_active
    }

    fn pause_animation(&mut self) {
        if let Some(animation) = &mut self.animation {
            animation.paused_at.get_or_insert_with(now_millis);
        }
    }

    fn resume_animation(&mut self, ctx: &Context<Self>) {
        if !self.canvas_visible() || self.interaction_active() {
            return;
        }
        if let Some(animation) = &mut self.animation {
            if let Some(paused_at) = animation.paused_at.take() {
                animation.paused_millis += (now_millis() - paused_at).max(0.0);
            }
            self.schedule_frame(ctx.link());
        }
    }

    fn invalidate_display(&mut self) {
        self.restart_refinement();
        self.canvas_dirty = true;
    }

    fn restart_refinement(&mut self) {
        if let Some(displayed) = &mut self.displayed {
            displayed.scene.restart_view_refinement();
            if displayed.scene.refinement_progress().is_refining() {
                displayed.refinement_started = now_millis();
            }
        }
    }

    fn suspend_display_refinement(&self) {
        if let Some(displayed) = &self.displayed {
            displayed.scene.cancel_refinement();
        }
    }

    fn discard_animation(&mut self) {
        if self.animation.take().is_some() {
            // The base canvas contains an in-between morph frame. Repaint the
            // last exact displayed scene immediately while replacement work
            // runs, including when the replacement route is a no-op.
            self.canvas_dirty = true;
        }
    }

    fn cancel_export_for_replacement(&mut self) {
        if let Some(job) = self.export_job.take() {
            job.cancelled.set(true);
            self.export_notice = None;
        }
    }

    fn needs_animation_frame(&self) -> bool {
        if !self.canvas_visible() || self.interaction_active() {
            return false;
        }
        self.autorotate_active && self.spatial_navigation()
            || self.animation.is_some()
            || self
                .displayed
                .as_ref()
                .is_some_and(|displayed| displayed.scene.refinement_progress().is_refining())
    }

    fn schedule_frame(&mut self, link: &Scope<Self>) {
        if self.frame_scheduled || !self.needs_animation_frame() {
            return;
        }
        self.frame_scheduled = true;
        let link = link.clone();
        let callback = Closure::once_into_js(move |timestamp: f64| {
            link.send_message(Msg::Frame(timestamp));
        });
        if let Some(window) = web_sys::window()
            && window
                .request_animation_frame(callback.unchecked_ref())
                .is_err()
        {
            self.frame_scheduled = false;
        }
    }

    fn update_frame(&mut self, ctx: &Context<Self>, timestamp: f64) -> bool {
        self.frame_scheduled = false;
        if !self.canvas_visible() {
            return false;
        }
        self.frames_since_sample = self.frames_since_sample.saturating_add(1);
        let sample_elapsed = timestamp - self.fps_sample_started;
        let mut refresh_view = false;
        if sample_elapsed >= 1_000.0 {
            self.fps =
                ((f64::from(self.frames_since_sample) * 1_000.0 / sample_elapsed).round()) as u32;
            self.frames_since_sample = 0;
            self.fps_sample_started = timestamp;
            refresh_view = true;
        }

        let autorotated =
            self.autorotate_active && self.spatial_navigation() && !self.interaction_active();
        if autorotated {
            let delta_millis = self
                .autorotate_last_frame
                .map_or(0.0, |last| timestamp - last)
                .clamp(0.0, MAX_AUTOROTATE_FRAME_MILLIS);
            self.autorotate_last_frame = Some(timestamp);
            self.orbit = self
                .orbit
                .autorotated(delta_millis * AUTOROTATE_RADIANS_PER_SECOND / 1_000.0);
            self.canvas_dirty = true;
            refresh_view = true;
        } else {
            self.autorotate_last_frame = None;
        }

        if self.animation.is_some() && !self.interaction_active() {
            let complete = {
                let animation = self.animation.as_mut().expect("animation was checked");
                let elapsed = (timestamp - animation.started - animation.paused_millis).max(0.0);
                let linear = (elapsed / TRANSITION_MILLIS).clamp(0.0, 1.0) as f32;
                animation.progress = iteration_ease(linear);
                linear >= 1.0
            };
            if let Err(error) = self.paint_current() {
                self.record_canvas_failure(error);
                refresh_view = true;
            } else if complete {
                let animation = self.animation.take().expect("animation was checked");
                self.promote_displayed(animation.target);
                self.canvas_dirty = false;
                if let Err(error) = self.paint_current() {
                    self.record_canvas_failure(error);
                } else {
                    self.queue_next_iteration(ctx);
                }
                refresh_view = true;
            }
        } else if !autorotated
            && !self.interaction_active()
            && let (Some(displayed), Some(exact)) = (
                &self.displayed,
                self.exact_canvas.cast::<HtmlCanvasElement>(),
            )
        {
            match displayed.scene.render_exact_batch(
                &exact,
                self.camera,
                self.orbit,
                self.effective_dark(),
                self.line_width_scale(),
            ) {
                Ok(progress) if progress.is_complete() => {
                    if let Some(base) = self.base_canvas.cast::<HtmlCanvasElement>()
                        && let Err(error) = displayed.scene.render_static(
                            &base,
                            self.camera,
                            self.orbit,
                            self.effective_dark(),
                            self.line_width_scale(),
                        )
                    {
                        self.record_canvas_failure(error.to_string());
                    }
                    refresh_view = true;
                }
                Ok(_) => {}
                Err(error) => {
                    self.record_canvas_failure(error.to_string());
                    refresh_view = true;
                }
            }
        }

        self.schedule_frame(ctx.link());
        refresh_view
    }

    fn paint_current(&self) -> Result<(), String> {
        if !self.canvas_visible() || self.viewport_size().is_none() {
            return Ok(());
        }
        let (Some(base), Some(exact)) = (
            self.base_canvas.cast::<HtmlCanvasElement>(),
            self.exact_canvas.cast::<HtmlCanvasElement>(),
        ) else {
            return Ok(());
        };
        if let Some(animation) = &self.animation {
            clear_canvas(&exact).map_err(|error| error.to_string())?;
            animation
                .transition
                .render(
                    &base,
                    self.camera,
                    self.effective_dark(),
                    self.line_width_scale(),
                    animation.progress,
                )
                .map(|_| ())
                .map_err(|error| error.to_string())?;
        } else if let Some(displayed) = &self.displayed {
            clear_canvas(&exact).map_err(|error| error.to_string())?;
            displayed
                .scene
                .render_preview(
                    &base,
                    self.camera,
                    self.orbit,
                    self.effective_dark(),
                    self.line_width_scale(),
                )
                .map(|_| ())
                .map_err(|error| error.to_string())?;
        } else {
            clear_canvas(&base).map_err(|error| error.to_string())?;
            clear_canvas(&exact).map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn record_canvas_failure(&mut self, error: String) {
        web_sys::console::error_1(&JsValue::from_str(&error));
        self.failure = Some(format!("Canvas display failed: {error}"));
        self.iteration_route = None;
        self.animation = None;
        self.suspend_display_refinement();
        self.canvas_dirty = false;
    }

    fn start_export(&mut self, ctx: &Context<Self>) {
        let Some(displayed) = &self.displayed else {
            return;
        };
        if let Some(job) = self.export_job.take() {
            job.cancelled.set(true);
        }
        self.export_sequence = self.export_sequence.wrapping_add(1);
        let id = self.export_sequence;
        let cancelled = Rc::new(Cell::new(false));
        self.export_job = Some(ExportJob {
            id,
            cancelled: Rc::clone(&cancelled),
        });
        self.export_notice = Some(String::from("Preparing SVG…"));
        let scene = Rc::clone(&displayed.scene);
        let dark = self.effective_dark();
        let line_width_scale = self.line_width_scale();
        let orbit = self.orbit;
        let link = ctx.link().clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = scene
                .encode_svg_yielding(dark, line_width_scale, orbit, &move || cancelled.get())
                .await
                .map_err(|error| match error {
                    crate::canvas::CanvasError::Cancelled => String::from("cancelled"),
                    error => error.to_string(),
                });
            link.send_message(Msg::Exported { id, result });
        });
    }

    fn selected_name(&self) -> &str {
        self.selected_preset
            .and_then(|index| self.presets.get(index))
            .map_or("Custom system", |preset| preset.name.as_str())
    }

    fn view_header(&self, ctx: &Context<Self>) -> Html {
        html! {
            <header class="workspace-header">
                <div class="system-heading">
                    <span class="eyebrow">{"Braken"}</span>
                    <h1 class="system-title">{self.selected_name()}</h1>
                </div>
                <button
                    class="desktop-only icon-button"
                    type="button"
                    title={self.theme_preference.button_label()}
                    aria-label={self.theme_preference.button_label()}
                    onclick={ctx.link().callback(|_| Msg::CycleTheme)}
                >{self.theme_preference.icon()}</button>
            </header>
        }
    }

    fn view_control_bar(&self, ctx: &Context<Self>) -> Html {
        let editor_open = self.editor_open;
        html! {
            <section class="control-bar" aria-label="Drawing controls and status">
                <button
                    id="desktop-edit-grammar"
                    class="desktop-only editor-toggle"
                    type="button"
                    aria-expanded={editor_open.to_string()}
                    aria-controls="grammar-editor"
                    onclick={ctx.link().callback(move |_| if editor_open { Msg::CloseEditor } else { Msg::OpenEditor })}
                >{if editor_open { "Close editor" } else { "Edit grammar" }}</button>
                {self.view_primary_controls(ctx)}
                {self.view_status(ctx)}
                <div class="bar-actions mobile-only">
                    <button
                        class="icon-button"
                        type="button"
                        title={self.theme_preference.button_label()}
                        aria-label={self.theme_preference.button_label()}
                        onclick={ctx.link().callback(|_| Msg::CycleTheme)}
                    >{self.theme_preference.icon()}</button>
                </div>
            </section>
        }
    }

    fn view_primary_controls(&self, ctx: &Context<Self>) -> Html {
        let angle_oninput = ctx.link().callback(|event: InputEvent| {
            Msg::AngleChanged(
                event
                    .target_unchecked_into::<HtmlInputElement>()
                    .value_as_number() as f32,
            )
        });
        let angle_wheel = ctx.link().callback(|event: WheelEvent| {
            event.prevent_default();
            Msg::AngleWheel(event.delta_y())
        });
        let iteration_input = ctx.link().callback(|event: InputEvent| {
            Msg::IterationsInput(event.target_unchecked_into::<HtmlInputElement>().value())
        });
        let iteration_slider = ctx.link().callback(|event: InputEvent| {
            Msg::IterationsSlider(
                event
                    .target_unchecked_into::<HtmlInputElement>()
                    .value_as_number()
                    .round()
                    .max(0.0) as usize,
            )
        });
        let iteration_wheel = ctx.link().callback(|event: WheelEvent| {
            event.prevent_default();
            Msg::IterationsWheel(event.delta_y())
        });
        let soft_max = self.suggested_max_iterations.max(1);
        let soft_value = self.iterations.min(soft_max);
        let modified = self.float_width != WorkerFloatWidth::F32
            || self.ambiguous_rules != WorkerAmbiguousRules::Uniform;

        html! {
            <section class="primary-controls" aria-label="Drawing controls">
                <div class="iteration-controls">
                    <div class="control-label">
                        <label class="field-label" for="iterations-range">{"Iterations"}</label>
                        if self.is_mobile() {
                            <output class="iteration-value" for="iterations-range" aria-label="Current iteration">{self.iterations}</output>
                        } else {
                            <input id="iterations-input" class="text-field" type="text" inputmode="numeric"
                                aria-label="Iterations" value={self.iterations_input.clone()} oninput={iteration_input}
                                aria-describedby="iterations-hint" />
                        }
                    </div>
                    if let Some(notice) = &self.iterations_notice {
                        <p class="notice error" role="alert">{notice}</p>
                    }
                    <div class="control-row">
                        <button type="button" aria-label="Previous iteration" onclick={ctx.link().callback(|_| Msg::IterationsStep(false))}>{"−"}</button>
                        <input id="iterations-range" class="range" aria-label="Suggested iteration range" type="range" min="0"
                            max={soft_max.to_string()} step="1" value={soft_value.to_string()}
                            aria-describedby="iterations-hint" title={format!("Suggested up to {} iterations", self.suggested_max_iterations)}
                            oninput={iteration_slider} onwheel={iteration_wheel} />
                        <button type="button" aria-label="Next iteration" onclick={ctx.link().callback(|_| Msg::IterationsStep(true))}>{"+"}</button>
                    </div>
                    <span class="control-hint" id="iterations-hint">{format!("Suggested up to {}", self.suggested_max_iterations)}</span>
                </div>
                <div class="angle-controls">
                    <div class="control-label">
                        <label class="field-label" for="angle-range">{format!("Angle: {}°", formatted_angle(self.angle))}</label>
                    </div>
                    <div class="control-row" onwheel={angle_wheel}>
                        <button type="button" aria-label="Decrease angle" onclick={ctx.link().callback(|_| Msg::AngleStep(-1.0))}>{"−"}</button>
                        <input id="angle-range" class="range" type="range" min="0" max="180" step="0.1"
                            value={self.angle.to_string()} oninput={angle_oninput} />
                        <button type="button" aria-label="Increase angle" onclick={ctx.link().callback(|_| Msg::AngleStep(1.0))}>{"+"}</button>
                    </div>
                </div>
                <button class="adjust-toggle" type="button"
                    aria-expanded={self.adjustments_open.to_string()} aria-controls="adjustment-controls"
                    onclick={ctx.link().callback(|_| Msg::ToggleAdjustments)}>
                    <span>{"Advanced"}</span>
                    if modified { <span class="modified-badge">{"Modified"}</span> }
                    if self.seed_notice.is_some() { <span class="notice error">{"Check seed"}</span> }
                </button>
            </section>
        }
    }

    fn view_adjustments(&self, ctx: &Context<Self>) -> Html {
        let line_width_oninput = ctx.link().callback(|event: InputEvent| {
            let value = event
                .target_unchecked_into::<HtmlInputElement>()
                .value_as_number()
                .round()
                .clamp(
                    f64::from(MIN_LINE_WIDTH_PERCENT),
                    f64::from(MAX_LINE_WIDTH_PERCENT),
                ) as u16;
            Msg::LineWidthChanged(value)
        });
        let seed_oninput = ctx.link().callback(|event: InputEvent| {
            Msg::SeedChanged(event.target_unchecked_into::<HtmlInputElement>().value())
        });
        let float_width_onchange = ctx.link().callback(|event: Event| {
            match event
                .target_unchecked_into::<HtmlSelectElement>()
                .value()
                .as_str()
            {
                "f32" => Msg::FloatWidthChanged(WorkerFloatWidth::F32),
                _ => Msg::FloatWidthChanged(WorkerFloatWidth::F64),
            }
        });
        let ambiguous_rules_onchange = ctx.link().callback(|event: Event| {
            match event
                .target_unchecked_into::<HtmlSelectElement>()
                .value()
                .as_str()
            {
                "first" => Msg::AmbiguousRulesChanged(WorkerAmbiguousRules::First),
                "error" => Msg::AmbiguousRulesChanged(WorkerAmbiguousRules::Error),
                _ => Msg::AmbiguousRulesChanged(WorkerAmbiguousRules::Uniform),
            }
        });
        html! {
            <section id="adjustment-controls" class="adjustments" hidden={!self.adjustments_open} aria-label="Additional controls">
                <div class="field-group">
                    <label class="field-label" for="line-width-range">{format!("Line width: {}%", self.line_width_percent)}</label>
                    <input id="line-width-range" class="range" type="range"
                        min={MIN_LINE_WIDTH_PERCENT.to_string()} max={MAX_LINE_WIDTH_PERCENT.to_string()}
                        step="5" value={self.line_width_percent.to_string()} oninput={line_width_oninput} />
                </div>
                <div class="field-group">
                    <label class="field-label" for="seed-input">{"Seed"}</label>
                    <div class="control-row">
                        <input id="seed-input" class="text-field" type="text" inputmode="numeric"
                            value={self.seed_input.clone()} placeholder={self.seed_placeholder.clone()} oninput={seed_oninput} />
                        <button class="icon-button" type="button" title="Generate random seed" aria-label="Generate random seed"
                            onclick={ctx.link().callback(|_| Msg::RandomizeSeed)}>{"⚄"}</button>
                    </div>
                    if let Some(notice) = &self.seed_notice {
                        <p class="notice error" role="alert">{notice}</p>
                    }
                </div>
                <div class="advanced-options">
                    <div id="derivation-options" class="semantics-fields">
                        <div class="field-group">
                            <label class="field-label" for="float-width">{"Floating-point precision"}</label>
                            <select id="float-width" class="text-field"
                                onchange={float_width_onchange}>
                                <option value="f32" selected={self.float_width == WorkerFloatWidth::F32}>{"32-bit (f32)"}</option>
                                <option value="f64" selected={self.float_width == WorkerFloatWidth::F64}>{"64-bit (f64)"}</option>
                            </select>
                        </div>
                        <div class="field-group">
                            <label class="field-label" for="ambiguous-rules">{"When multiple rules match"}</label>
                            <select id="ambiguous-rules" class="text-field"
                                onchange={ambiguous_rules_onchange}>
                                <option value="uniform" selected={self.ambiguous_rules == WorkerAmbiguousRules::Uniform}>{"Uniform random choice"}</option>
                                <option value="first" selected={self.ambiguous_rules == WorkerAmbiguousRules::First}>{"First matching rule"}</option>
                                <option value="error" selected={self.ambiguous_rules == WorkerAmbiguousRules::Error}>{"Report ambiguity"}</option>
                            </select>
                        </div>
                    </div>
                </div>
                <button id="mobile-edit-grammar" class="mobile-only" type="button"
                    aria-controls="grammar-editor" onclick={ctx.link().callback(|_| Msg::OpenEditor)}>
                    {if self.editor_draft.is_modified() { "Continue editing grammar" } else { "Edit grammar" }}
                </button>
            </section>
        }
    }

    fn view_editor(&self, ctx: &Context<Self>) -> Html {
        let draft_mode = self.editor_draft_mode;
        let source_oninput = ctx.link().callback(move |event: InputEvent| {
            let source = event.target_unchecked_into::<HtmlTextAreaElement>().value();
            if draft_mode {
                Msg::DraftChanged(source)
            } else {
                Msg::SourceChanged(source)
            }
        });
        html! {
            <section id="grammar-editor" class="editor-panel" hidden={!self.editor_open} aria-labelledby="editor-title">
                <div class="editor-heading">
                    <div>
                        <h2 class="section-title" id="editor-title">{"Edit grammar"}</h2>
                        <p class="control-hint">{
                            if draft_mode && self.is_mobile() { "Apply your changes to see the result. Back keeps your draft." }
                            else if draft_mode { "Apply your changes to see the result. Closing keeps your draft." }
                            else { "Changes update the drawing as you type." }
                        }</p>
                    </div>
                    if draft_mode {
                        <div class="editor-toolbar">
                            if self.is_mobile() {
                                <button type="button" onclick={ctx.link().callback(|_| Msg::CloseEditor)}>{"Back"}</button>
                            }
                            <button class="primary-button" type="button" onclick={ctx.link().callback(|_| Msg::ApplyDraft)}>
                                {"Apply and return"}
                            </button>
                        </div>
                    }
                </div>
                <textarea id="source-editor" class="source-editor" aria-label="Grammar source"
                    value={if draft_mode { self.editor_draft.text().to_owned() } else { self.source.clone() }}
                    placeholder={"axiom Draw;\nmatch Draw then Draw Turn(60) Draw;"}
                    spellcheck="false" autocapitalize="off" autocomplete="off" oninput={source_oninput} />
                if let Some(error) = &self.failure {
                    <p class="notice error" role="alert">{error}</p>
                }
            </section>
        }
    }

    fn view_featured_presets(&self, ctx: &Context<Self>) -> Html {
        html! {
            <section class="featured-presets mobile-only" aria-labelledby="featured-heading">
                <div class="section-heading">
                    <h2 class="section-title" id="featured-heading">{"Presets"}</h2>
                    <button id="browse-presets" class="text-button" type="button"
                        onclick={ctx.link().callback(|_| Msg::OpenCatalog)}>{"Browse all →"}</button>
                </div>
                <div class="preset-strip" aria-label="Featured presets">
                    {for FEATURED_PRESETS.iter().filter_map(|name| {
                        self.presets.iter().enumerate().find(|(_, preset)| preset.name == *name)
                    }).map(|(index, preset)| self.view_preset_card(ctx, index, preset, true))}
                </div>
                <p class="selected-preset-name">{format!("Selected: {}", self.selected_name())}</p>
            </section>
        }
    }

    fn view_presets(&self, ctx: &Context<Self>) -> Html {
        let terms = normalized_search_terms(&self.preset_search);
        let matches = self
            .presets
            .iter()
            .enumerate()
            .filter(|(_, preset)| preset.matches_search_terms(&terms))
            .collect::<Vec<_>>();
        html! {
            <aside class="preset-gallery" aria-labelledby="presets-heading">
                <div class="gallery-heading">
                    <h2 class="section-title" id="presets-heading" tabindex="-1">{"Presets"}</h2>
                    <button class="catalog-back mobile-only" type="button"
                        onclick={ctx.link().callback(|_| Msg::CloseCatalog)}>{"Back to drawing"}</button>
                </div>
                    <div class="preset-tools">
                        <div class="control-row">
                            <input
                                id="preset-search"
                                class="text-field"
                                type="search"
                                aria-label="Search all presets"
                                placeholder="Search presets"
                                value={self.preset_search.clone()}
                                oninput={ctx.link().callback(|event: InputEvent| Msg::SearchChanged(event.target_unchecked_into::<HtmlInputElement>().value()))}
                            />
                            <button type="button" onclick={ctx.link().callback(|_| Msg::ClearSearch)}> {"Clear"} </button>
                        </div>
                        <span class="preset-summary">{format!("{} presets", matches.len())}</span>
                    </div>
                <div class="preset-list">
                    if matches.is_empty() {
                        <p class="notice">{"No presets match this search."}</p>
                    } else {
                        {for matches.into_iter().map(|(index, preset)| self.view_preset_card(ctx, index, preset, false))}
                    }
                </div>
            </aside>
        }
    }

    fn view_preset_card(
        &self,
        ctx: &Context<Self>,
        index: usize,
        preset: &Preset,
        featured: bool,
    ) -> Html {
        let is_3d = preset.visualizer == VisualizerKind::Turtle3d;
        let dimension_class = if is_3d {
            Some("dimension-3d")
        } else if preset.visualizer == VisualizerKind::Turtle2d {
            Some("dimension-2d")
        } else {
            None
        };
        let class = classes!(
            "preset-card",
            featured.then_some("featured-card"),
            (self.selected_preset == Some(index)).then_some("selected"),
            dimension_class,
        );
        let badge_class = classes!("dimension-badge", dimension_class);
        let preview =
            Html::from_html_unchecked(self.preset_previews[index].for_theme(self.effective_dark()));
        html! {
            <button
                key={index}
                {class}
                type="button"
                aria-pressed={(self.selected_preset == Some(index)).to_string()}
                onclick={ctx.link().callback(move |_| Msg::PresetSelected(index))}
            >
                <span class="preset-preview" aria-hidden="true">{preview}</span>
                <span class="preset-copy">
                    <span class="preset-name">{preset.name.as_str()}</span>
                    <span class="preset-summary">{preset.summary.as_str()}</span>
                    <span class="preset-meta">
                        if let Some(label) = preset.dimension_badge_label() {
                            <span
                                class={badge_class}
                                title={format!("{label} turtle visualizer")}
                                aria-label={format!("{label} turtle visualizer")}
                            >{label}</span>
                        }
                    </span>
                </span>
            </button>
        }
    }

    fn view_canvas(&self, ctx: &Context<Self>) -> Html {
        let navigation = self.navigation_enabled();
        let spatial_navigation = self.spatial_navigation();
        let planar_navigation = navigation && !spatial_navigation;
        self.touch_navigation_enabled.set(navigation);
        self.touch_required_count.set(required_canvas_touches(
            self.is_mobile(),
            spatial_navigation,
        ));
        let input_ref = self.input_layer.clone();
        let pointer_down_ref = input_ref.clone();
        let pointer_move_ref = input_ref.clone();
        let pointer_up_ref = input_ref.clone();
        let pointer_lost_ref = input_ref.clone();
        let desktop_navigation = navigation && !self.is_mobile();
        let onpointerdown = ctx.link().batch_callback(move |event: PointerEvent| {
            if !desktop_navigation
                || !is_primary_mouse_button(&event.pointer_type(), event.button())
                || !event.is_primary()
            {
                return None;
            }
            let element = pointer_down_ref.cast::<HtmlElement>()?;
            event.prevent_default();
            let _ = element.set_pointer_capture(event.pointer_id());
            Some(Msg::PointerDown(
                event.pointer_id(),
                local_pointer_position(&event, &element),
            ))
        });
        let onpointermove = ctx.link().batch_callback(move |event: PointerEvent| {
            if event.pointer_type() != "mouse" {
                return None;
            }
            let element = pointer_move_ref.cast::<HtmlElement>()?;
            Some(Msg::PointerMove(
                event.pointer_id(),
                local_pointer_position(&event, &element),
            ))
        });
        let onpointerup = ctx.link().batch_callback(move |event: PointerEvent| {
            if event.pointer_type() != "mouse"
                || (event.type_() != "pointercancel" && event.button() != 0)
            {
                return None;
            }
            let element = pointer_up_ref.cast::<HtmlElement>()?;
            let _ = element.release_pointer_capture(event.pointer_id());
            Some(Msg::PointerUp(event.pointer_id()))
        });
        let onlostpointercapture = ctx.link().batch_callback(move |event: PointerEvent| {
            if event.pointer_type() != "mouse" {
                return None;
            }
            let _element = pointer_lost_ref.cast::<HtmlElement>()?;
            Some(Msg::PointerUp(event.pointer_id()))
        });
        let onwheel = ctx.link().batch_callback(move |event: WheelEvent| {
            if !planar_navigation {
                return None;
            }
            event.prevent_default();
            let amount = match event.delta_mode() {
                WheelEvent::DOM_DELTA_PIXEL => -event.delta_y() / VIEW_WHEEL_PIXELS_PER_LINE,
                _ => -event.delta_y(),
            };
            Some(Msg::WheelZoom {
                factor: VIEW_WHEEL_FACTOR.powf(amount),
                anchor: ScreenPoint::new(f64::from(event.offset_x()), f64::from(event.offset_y())),
            })
        });
        html! {
            <section class="viewport-panel" aria-label="L-system drawing">
                <div class="canvas-stage">
                    <canvas class="canvas-layer" ref={self.base_canvas.clone()} />
                    <canvas class="canvas-layer canvas-exact" ref={self.exact_canvas.clone()} />
                    <div
                        id="drawing-input"
                        class={classes!("canvas-input", spatial_navigation.then_some("spatial-input"))}
                        ref={input_ref}
                        role="img"
                        aria-label={self.canvas_accessible_label()}
                        tabindex="0"
                        onpointerdown={onpointerdown}
                        onpointermove={onpointermove}
                        onpointerup={onpointerup.clone()}
                        onpointercancel={onpointerup}
                        {onlostpointercapture}
                        onwheel={onwheel}
                    />
                    if self.displayed.is_none() {
                        <div class="canvas-empty-label">{"Preparing the first scene…"}</div>
                    }
                    if planar_navigation && !self.is_mobile() {
                        <div class="viewport-toolbar" aria-label="Drawing view controls">
                            <button type="button" aria-label="Zoom out" onclick={ctx.link().callback(|_| Msg::ZoomBy(1.0 / VIEW_BUTTON_ZOOM_FACTOR))}>{"−"}</button>
                            <button type="button" onclick={ctx.link().callback(|_| Msg::FitView)}>{"Fit"}</button>
                            <button type="button" aria-label="Zoom in" onclick={ctx.link().callback(|_| Msg::ZoomBy(VIEW_BUTTON_ZOOM_FACTOR))}>{"+"}</button>
                            <output class="zoom-readout">{format!("{:.0}%", self.camera.zoom * 100.0)}</output>
                        </div>
                    }
                </div>
            </section>
        }
    }

    fn view_status(&self, ctx: &Context<Self>) -> Html {
        let now = now_millis();
        let (primary, mut secondary) = self.status_text(now);
        let slow = self.work_is_slow(now);
        if slow {
            secondary.push_str(" · This is taking a while");
        }
        html! {
            <footer class="status-footer">
                <div class="status-copy">
                    <div class={classes!("status-primary", self.failure.is_some().then_some("error"))} aria-live="polite">{primary}</div>
                    if self.is_mobile() {
                        <details class="status-details">
                            <summary>{"Details"}</summary>
                            <div class="status-secondary">{secondary}</div>
                        </details>
                    } else {
                        <div class="status-secondary">{secondary}</div>
                    }
                    if let Some(notice) = &self.export_notice {
                        <div class="status-notice" aria-live="polite">{notice}</div>
                    }
                </div>
                <div class="status-actions">
                    if self.export_job.is_some() {
                        <button type="button" onclick={ctx.link().callback(|_| Msg::CancelExport)}>{"Cancel SVG export"}</button>
                    } else {
                        <button
                            class="icon-button"
                            type="button"
                            title="Download SVG"
                            aria-label="Download SVG"
                            disabled={self.displayed.is_none()}
                            onclick={ctx.link().callback(|_| Msg::ExportSvg)}
                        >{"⇩"}</button>
                    }
                    if slow {
                        <button class="danger-button" type="button" onclick={ctx.link().callback(|_| Msg::CancelWork)}>{"Cancel"}</button>
                    }
                </div>
            </footer>
        }
    }

    fn status_text(&self, now: f64) -> (String, String) {
        let resting = self.displayed.as_ref().map_or_else(
            || String::from("Ready"),
            |displayed| format!("{} elements", displayed.scene.element_count()),
        );
        let primary = if let Some(animation) = &self.animation {
            format!(
                "Animating iteration {} → {}",
                animation.from_iteration, animation.to_iteration
            )
        } else if let Some(active) = &self.active {
            if now - active.started < TRANSIENT_STATUS_DELAY_MILLIS {
                resting
            } else if let Some(progress) = &active.progress {
                if progress.phase == "Visualizing" {
                    format!(
                        "Visualizing · {} items · {} lines",
                        progress.items, progress.phase_completed
                    )
                } else if progress.total_iterations > 0 {
                    format!(
                        "Deriving {}/{} · {} modules · {} items",
                        progress.completed_iterations,
                        progress.total_iterations,
                        progress.modules,
                        progress.items
                    )
                } else if let Some(total) = progress.phase_total {
                    format!(
                        "{} {}/{}",
                        human_phase(&progress.phase),
                        progress.phase_completed,
                        total
                    )
                } else {
                    format!("{}…", human_phase(&progress.phase))
                }
            } else {
                String::from("Queued…")
            }
        } else if self.failure.is_some() {
            if self.displayed.is_some() {
                String::from("Error — showing the last completed result")
            } else {
                String::from("Error")
            }
        } else if let Some(displayed) = &self.displayed {
            let progress = displayed.scene.refinement_progress();
            let (completed, total) = (progress.completed, progress.total);
            if display_refinement_is_running(
                progress.is_refining(),
                self.autorotate_active,
                self.interaction_active(),
            ) && now - displayed.refinement_started >= TRANSIENT_STATUS_DELAY_MILLIS
            {
                format!("Showing preview · refining display {completed}/{total} lines")
            } else if progress.cancelled && completed < total {
                format!("Showing preview · display refinement stopped at {completed}/{total}")
            } else {
                resting
            }
        } else {
            resting
        };
        let primary = self
            .failure
            .as_ref()
            .map_or(primary.clone(), |failure| format!("{primary} · {failure}"));
        let secondary = if let Some(progress) = self
            .active
            .as_ref()
            .and_then(|active| active.progress.as_ref())
        {
            format!(
                "{} ms in Worker · {} FPS",
                progress.elapsed_millis, self.fps
            )
        } else {
            self.displayed.as_ref().map_or_else(
                || format!("— ms · {} FPS", self.fps),
                |displayed| {
                    format!(
                        "{} ms on {} · {} FPS",
                        displayed.elapsed_millis, displayed.backend, self.fps
                    )
                },
            )
        };
        (primary, secondary)
    }

    fn work_is_slow(&self, now: f64) -> bool {
        self.active
            .as_ref()
            .is_some_and(|job| now - job.started >= SLOW_NOTICE_DELAY_MILLIS)
            || self.displayed.as_ref().is_some_and(|displayed| {
                display_refinement_is_running(
                    displayed.scene.refinement_progress().is_refining(),
                    self.autorotate_active,
                    self.interaction_active(),
                ) && now - displayed.refinement_started >= SLOW_NOTICE_DELAY_MILLIS
            })
    }

    fn canvas_accessible_label(&self) -> String {
        self.displayed.as_ref().map_or_else(
            || String::from("L-system drawing is loading"),
            |displayed| {
                if displayed.scene.is_spatial() {
                    format!(
                        "3D L-system drawing with {} elements at iteration {}; drag to rotate, tap or click to stop automatic rotation",
                        displayed.scene.element_count(),
                        displayed.iteration
                    )
                } else {
                    format!(
                        "L-system drawing with {} elements at iteration {}",
                        displayed.scene.element_count(),
                        displayed.iteration
                    )
                }
            },
        )
    }
}

fn schedule_timeout(link: &Scope<App>, message: Msg, delay_millis: i32) {
    let link = link.clone();
    let callback = Closure::once_into_js(move || link.send_message(message));
    if let Some(window) = web_sys::window() {
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.unchecked_ref(),
            delay_millis,
        );
    }
}

fn now_millis() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or_else(js_sys::Date::now, |performance| performance.now())
}

impl TouchListeners {
    fn install(
        element: HtmlElement,
        link: Scope<App>,
        owned: Rc<Cell<bool>>,
        navigation_enabled: Rc<Cell<bool>>,
        required_count: Rc<Cell<usize>>,
    ) -> Result<Self, String> {
        let start = touch_listener(
            TouchPhase::Start,
            element.clone(),
            link.clone(),
            Rc::clone(&owned),
            Rc::clone(&navigation_enabled),
            Rc::clone(&required_count),
        );
        let move_ = touch_listener(
            TouchPhase::Move,
            element.clone(),
            link.clone(),
            Rc::clone(&owned),
            Rc::clone(&navigation_enabled),
            Rc::clone(&required_count),
        );
        let end = touch_listener(
            TouchPhase::End,
            element.clone(),
            link.clone(),
            Rc::clone(&owned),
            Rc::clone(&navigation_enabled),
            Rc::clone(&required_count),
        );
        let cancel = touch_listener(
            TouchPhase::End,
            element.clone(),
            link,
            owned,
            navigation_enabled,
            required_count,
        );
        let listeners = Self {
            element,
            start,
            move_,
            end,
            cancel,
        };
        if let Err(error) = listeners.add() {
            listeners.remove();
            return Err(error);
        }
        Ok(listeners)
    }

    fn add(&self) -> Result<(), String> {
        let options = AddEventListenerOptions::new();
        options.set_capture(false);
        options.set_passive(false);
        self.add_one("touchstart", &self.start, &options)?;
        self.add_one("touchmove", &self.move_, &options)?;
        self.add_one("touchend", &self.end, &options)?;
        self.add_one("touchcancel", &self.cancel, &options)
    }

    fn add_one(
        &self,
        event: &str,
        listener: &Closure<dyn FnMut(TouchEvent)>,
        options: &AddEventListenerOptions,
    ) -> Result<(), String> {
        self.element
            .add_event_listener_with_callback_and_add_event_listener_options(
                event,
                listener.as_ref().unchecked_ref(),
                options,
            )
            .map_err(|error| format!("could not install {event} input handling: {error:?}"))
    }

    fn remove(&self) {
        for (event, listener) in [
            ("touchstart", &self.start),
            ("touchmove", &self.move_),
            ("touchend", &self.end),
            ("touchcancel", &self.cancel),
        ] {
            let _ = self.element.remove_event_listener_with_callback_and_bool(
                event,
                listener.as_ref().unchecked_ref(),
                false,
            );
        }
    }
}

fn touch_listener(
    phase: TouchPhase,
    element: HtmlElement,
    link: Scope<App>,
    owned: Rc<Cell<bool>>,
    navigation_enabled: Rc<Cell<bool>>,
    required_count: Rc<Cell<usize>>,
) -> Closure<dyn FnMut(TouchEvent)> {
    Closure::wrap(Box::new(move |event: TouchEvent| {
        let points = local_touches(&event, &element);
        let ownership = touch_ownership(
            owned.get(),
            navigation_enabled.get(),
            required_count.get(),
            points.len(),
        );
        if ownership.handle_event {
            event.prevent_default();
        }
        owned.set(ownership.retain_after_event);
        link.send_message(Msg::Touch {
            phase,
            points,
            owned: ownership.handle_event,
        });
    }) as Box<dyn FnMut(TouchEvent)>)
}

fn local_pointer_position(event: &PointerEvent, element: &HtmlElement) -> ScreenPoint {
    let rect = element.get_bounding_client_rect();
    ScreenPoint::new(
        f64::from(event.client_x()) - rect.left(),
        f64::from(event.client_y()) - rect.top(),
    )
}

fn local_touches(event: &TouchEvent, element: &HtmlElement) -> Vec<TouchPoint> {
    let rect = element.get_bounding_client_rect();
    let touches = event.touches();
    let mut points = (0..touches.length())
        .filter_map(|index| touches.item(index))
        .map(|touch| TouchPoint {
            identifier: touch.identifier(),
            position: ScreenPoint::new(
                f64::from(touch.client_x()) - rect.left(),
                f64::from(touch.client_y()) - rect.top(),
            ),
        })
        .collect::<Vec<_>>();
    points.sort_by_key(|point| point.identifier);
    points
}

fn touch_measurement(points: &[TouchPoint]) -> (ScreenPoint, f64) {
    let first = points[0].position;
    points.get(1).map_or((first, 0.0), |second| {
        let second = second.position;
        (
            ScreenPoint::new((first.x + second.x) * 0.5, (first.y + second.y) * 0.5),
            (second.x - first.x).hypot(second.y - first.y),
        )
    })
}

fn download_svg(svg: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or_else(|| String::from("browser window is unavailable"))?;
    let document = window
        .document()
        .ok_or_else(|| String::from("browser document is unavailable"))?;
    let parts = js_sys::Array::new();
    parts.push(&JsValue::from_str(svg));
    let options = BlobPropertyBag::new();
    options.set_type("image/svg+xml;charset=utf-8");
    let blob = Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|error| format!("could not prepare SVG download: {error:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|error| format!("could not create SVG download URL: {error:?}"))?;
    let anchor = document
        .create_element("a")
        .map_err(|error| format!("could not create SVG download link: {error:?}"))?
        .dyn_into::<HtmlAnchorElement>()
        .map_err(|_| String::from("SVG download link has an unexpected browser type"))?;
    anchor.set_href(&url);
    anchor.set_download("lsystem.svg");
    anchor
        .style()
        .set_property("display", "none")
        .map_err(|error| format!("could not hide SVG download link: {error:?}"))?;
    let body = document
        .body()
        .ok_or_else(|| String::from("browser document body is unavailable"))?;
    body.append_child(&anchor)
        .map_err(|error| format!("could not attach SVG download link: {error:?}"))?;
    anchor.click();
    let _ = body.remove_child(&anchor);
    web_sys::Url::revoke_object_url(&url)
        .map_err(|error| format!("could not release SVG download URL: {error:?}"))
}

fn human_phase(phase: &str) -> &str {
    match phase {
        "Preparing" => "Preparing",
        "InspectingInput" => "Inspecting input",
        "Indexing" => "Indexing",
        "SelectingProductions" => "Selecting productions",
        "Rewriting" => "Deriving",
        "ValidatingOutput" => "Validating output",
        "Transferring" => "Transferring",
        "Complete" => "Completing",
        "Visualizing" => "Building preview",
        _ => "Working",
    }
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

fn cubic_bezier_ease(progress: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    fn coordinate(t: f32, first: f32, second: f32) -> f32 {
        let inverse = 1.0 - t;
        3.0 * inverse * inverse * t * first + 3.0 * inverse * t * t * second + t * t * t
    }
    fn derivative(t: f32, first: f32, second: f32) -> f32 {
        let inverse = 1.0 - t;
        3.0 * inverse * inverse * first
            + 6.0 * inverse * t * (second - first)
            + 3.0 * t * t * (1.0 - second)
    }

    let x = progress.clamp(0.0, 1.0);
    let mut parameter = x;
    for _ in 0..8 {
        let slope = derivative(parameter, x1, x2);
        if slope.abs() <= 1.0e-6 {
            break;
        }
        parameter = (parameter - (coordinate(parameter, x1, x2) - x) / slope).clamp(0.0, 1.0);
    }
    coordinate(parameter, y1, y2).clamp(0.0, 1.0)
}

fn iteration_ease(progress: f32) -> f32 {
    cubic_bezier_ease(progress, 0.4, 0.0, 0.2, 1.0)
}
