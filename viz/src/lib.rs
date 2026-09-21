//! Target-neutral visualization of calculated L-system generations.

use std::{fmt, str::FromStr, time::Duration};

use braken::Generation;

pub mod geometry;
pub mod targets;
mod transitions;
mod visualizers;

pub use geometry::{Line2d, Line3d, Lines2d, Point2d, Point3d};
pub use transitions::{
    IndexedLine2d, IndexedModulePosition2d, MorphCoordinateSpace, MorphLine2d,
    build_line_transition,
};

/// Color attached to turtle geometry.
///
/// `ThemeDefault` deliberately preserves the display backend's normal,
/// theme-aware styling. Palette commands switch subsequent geometry away from
/// that default; an RGB value is used when the source supplies an exact
/// palette, while `PaletteIndex` lets the display choose a pleasant light/dark
/// fallback palette when it does not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StrokeColor {
    #[default]
    ThemeDefault,
    PaletteIndex(u16),
    Rgb([u8; 3]),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StyledLine2d {
    pub line: Line2d,
    /// Multiplier applied to the output target's automatic base stroke width.
    /// Zero suppresses the segment.
    pub width: f64,
    pub color: StrokeColor,
}

/// Styled line segment retained in turtle-world three-dimensional coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StyledLine3d {
    pub line: Line3d,
    /// Multiplier applied to the output target's automatic base stroke width.
    /// Zero suppresses the segment.
    pub width: f64,
    pub color: StrokeColor,
}

/// Filled polygon emitted by turtle polygon capture commands.
///
/// Between `PolygonBegin` and `PolygonEnd`, `Vertex` records the current
/// position explicitly and ABOP `F`/`f` advances trace contour edges
/// implicitly. Other configured draw/move modules do not trace the contour.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon2d {
    pub vertices: Vec<Point2d>,
    pub color: StrokeColor,
}

/// Filled polygon retained in turtle-world three-dimensional coordinates.
///
/// It uses the same explicit `Vertex` and implicit ABOP `F`/`f` contour
/// semantics as [`Polygon2d`].
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon3d {
    pub vertices: Vec<Point3d>,
    pub color: StrokeColor,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum VisualizerKind {
    #[default]
    Inspector,
    Sequence,
    Turtle2d,
    Turtle3d,
    AxialTree,
    Asset,
    Polygon,
    Plot,
    PlanarMap,
    SphericalMap,
    Cellwork3d,
}

impl VisualizerKind {
    pub const ALL: [Self; 11] = [
        Self::Inspector,
        Self::Sequence,
        Self::Turtle2d,
        Self::Turtle3d,
        Self::AxialTree,
        Self::Asset,
        Self::Polygon,
        Self::Plot,
        Self::PlanarMap,
        Self::SphericalMap,
        Self::Cellwork3d,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inspector => "inspector",
            Self::Sequence => "sequence",
            Self::Turtle2d => "turtle_2d",
            Self::Turtle3d => "turtle_3d",
            Self::AxialTree => "axial_tree",
            Self::Asset => "asset",
            Self::Polygon => "polygon",
            Self::Plot => "plot",
            Self::PlanarMap => "planar_map",
            Self::SphericalMap => "spherical_map",
            Self::Cellwork3d => "cellwork_3d",
        }
    }
}

impl fmt::Display for VisualizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseVisualizerKindError(String);
impl fmt::Display for ParseVisualizerKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown visualizer {:?}", self.0)
    }
}
impl std::error::Error for ParseVisualizerKindError {}
impl FromStr for VisualizerKind {
    type Err = ParseVisualizerKindError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == value)
            .ok_or_else(|| ParseVisualizerKindError(value.to_owned()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualizerMetadataError {
    Duplicate,
    Invalid(ParseVisualizerKindError),
}
impl fmt::Display for VisualizerMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Duplicate => f.write_str("source contains more than one Visualizer declaration"),
            Self::Invalid(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for VisualizerMetadataError {}

pub fn visualizer_metadata(
    source: &str,
) -> Result<Option<VisualizerKind>, VisualizerMetadataError> {
    let mut values = source.lines().filter_map(|line| {
        line.trim_start()
            .strip_prefix("# Visualizer:")
            .map(str::trim)
    });
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(VisualizerMetadataError::Duplicate);
    }
    value
        .parse()
        .map(Some)
        .map_err(VisualizerMetadataError::Invalid)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VisualizerBackend {
    /// Prefer a supported native accelerator, currently CUDA for the
    /// branch-free scalar 2D turtle subset, with CPU fallback only before
    /// accelerator work starts.
    /// Visualizers without an accelerated implementation use CPU.
    #[default]
    Auto,
    /// Run the visualizer's CPU implementation.
    Cpu,
    /// Require a native CUDA implementation built with the `cuda` feature.
    /// Unsupported targets, visualizers, or input features return a typed
    /// error rather than falling back.
    Cuda,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InspectorConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InspectorPhase {
    #[default]
    Statistics,
    Preview,
    Complete,
}

/// Monotonic progress from an incremental inspector visualization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InspectorProgress {
    pub phase: InspectorPhase,
    pub items_processed: usize,
    pub modules: usize,
    pub branches: usize,
    pub max_branch_depth: usize,
    pub preview_bytes: usize,
}

/// Input and cooperative-cancellation hook for the inspector visualizer.
pub struct InspectorStreamRequest<'a> {
    pub generation: &'a Generation,
    pub config: InspectorConfig,
    pub context: VisualizationContext,
    /// Maximum generation items handled between progress callbacks.
    /// This is a scheduling preference, not an output limit.
    pub work_quantum: usize,
    pub is_cancelled: &'a dyn Fn() -> bool,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct SequenceConfig;
#[derive(Debug, Clone, Default)]
pub struct Turtle3dConfig {
    /// Shared drawing, style, metadata, and planar-turn configuration. The 3D
    /// interpreter adds pitch and roll while retaining these source contracts.
    pub turtle: Turtle2dConfig,
}

impl Turtle3dConfig {
    pub fn with_source_metadata(mut self, source: &str) -> Self {
        self.turtle = self.turtle.with_source_metadata(source);
        self
    }
}

impl From<Turtle2dConfig> for Turtle3dConfig {
    fn from(turtle: Turtle2dConfig) -> Self {
        Self { turtle }
    }
}
#[derive(Debug, Clone, Copy, Default)]
pub struct AxialTreeConfig;
#[derive(Debug, Clone, Copy, Default)]
pub struct AssetConfig;
#[derive(Debug, Clone, Copy, Default)]
pub struct PolygonConfig;
#[derive(Debug, Clone, Copy, Default)]
pub struct PlotConfig;
#[derive(Debug, Clone, Copy, Default)]
pub struct PlanarMapConfig;
#[derive(Debug, Clone, Copy, Default)]
pub struct SphericalMapConfig;
#[derive(Debug, Clone, Copy, Default)]
pub struct Cellwork3dConfig;

#[derive(Debug, Clone)]
pub struct Turtle2dConfig {
    pub turn_angle: f64,
    pub initial_angle: f64,
    pub default_step: f64,
    /// Multiplier used by parameterless `ScaleLength` commands.
    pub scale_multiplier: f64,
    pub initial_width: f64,
    pub width_increment: f64,
    /// Amount applied by parameterless `TurnAngleIncrease` and
    /// `TurnAngleDecrease`, in radians.
    pub turn_angle_increment: f64,
    /// Initial stroke state. The default retains the renderer's themed flare.
    pub initial_color: StrokeColor,
    /// Amount applied by parameterless relative color commands.
    pub color_increment: i32,
    /// Optional exact source palette. Palette indices wrap with Euclidean
    /// modulo. An empty palette selects the renderer's theme-aware fallback.
    pub palette: Vec<[u8; 3]>,
    /// Optional exact source background.
    pub background: Option<[u8; 3]>,
    pub draw_modules: Vec<String>,
    pub move_modules: Vec<String>,
    pub module_aliases: Vec<(String, String)>,
}
impl Default for Turtle2dConfig {
    fn default() -> Self {
        Self {
            turn_angle: std::f64::consts::FRAC_PI_2,
            initial_angle: std::f64::consts::FRAC_PI_2,
            default_step: 1.0,
            scale_multiplier: 1.0,
            initial_width: 1.0,
            width_increment: 1.0,
            turn_angle_increment: 1.0_f64.to_radians(),
            initial_color: StrokeColor::ThemeDefault,
            color_increment: 1,
            palette: Vec::new(),
            background: None,
            draw_modules: vec!["F".into(), "Draw".into()],
            move_modules: vec!["f".into(), "Move".into()],
            module_aliases: Vec::new(),
        }
    }
}

impl Turtle2dConfig {
    pub fn with_source_metadata(mut self, source: &str) -> Self {
        if let Some(heading) = numeric_metadata(source, "Heading") {
            self.initial_angle = heading.to_radians();
        }
        if let Some(width) = numeric_metadata(source, "Initial Width")
            .or_else(|| note_number(source, "Upstream thickness setting"))
        {
            self.initial_width = width;
        }
        if let Some(scale) =
            numeric_metadata(source, "Scale").or_else(|| named_numeric_constant(source, "SCALE"))
        {
            self.scale_multiplier = scale;
        }
        if let Some(increment) = numeric_metadata(source, "Width Increment") {
            self.width_increment = increment;
        }
        if let Some(increment) = numeric_metadata(source, "Angle Increment") {
            self.turn_angle_increment = increment.to_radians();
        }
        if let Some(increment) = integer_metadata(source, "Color Increment") {
            self.color_increment = increment;
        }
        if let Some(palette) = metadata(source, "Palette").and_then(parse_palette)
            && !palette.is_empty()
        {
            self.palette = palette;
        }
        if let Some(color) = metadata(source, "Stroke Color").and_then(parse_rgb) {
            self.initial_color = StrokeColor::Rgb(color);
        } else if let Some(index) = integer_metadata(source, "Initial Color") {
            self.initial_color = palette_color(index, &self.palette);
        }
        if let Some(background) = metadata(source, "Background").and_then(parse_rgb) {
            self.background = Some(background);
        }
        if let Some(modules) = module_list_metadata(source, "Draw") {
            self.draw_modules = modules;
        }
        if let Some(modules) = module_list_metadata(source, "Move") {
            self.move_modules = modules;
        }
        if let Some(aliases) = module_alias_metadata(source) {
            self.module_aliases = aliases;
        }
        self
    }
}

fn numeric_metadata(source: &str, key: &str) -> Option<f64> {
    metadata(source, key)?.parse().ok()
}

fn named_numeric_constant(source: &str, name: &str) -> Option<f64> {
    source.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("let ")?;
        let (found, value) = rest.split_once('=')?;
        if found.trim() != name {
            return None;
        }
        value.trim().strip_suffix(';')?.trim().parse().ok()
    })
}

fn integer_metadata(source: &str, key: &str) -> Option<i32> {
    metadata(source, key)?.parse().ok()
}

fn note_number(source: &str, label: &str) -> Option<f64> {
    source.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix("# Note")?;
        let value = rest.trim_start().strip_prefix(':')?.trim();
        value
            .strip_prefix(label)?
            .trim_start()
            .strip_prefix(':')?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

fn parse_palette(value: &str) -> Option<Vec<[u8; 3]>> {
    value
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|entry| !entry.is_empty())
        .map(parse_rgb)
        .collect()
}

fn parse_rgb(value: &str) -> Option<[u8; 3]> {
    let hex = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if hex.len() == 6 {
        return Some([
            u8::from_str_radix(&hex[0..2], 16).ok()?,
            u8::from_str_radix(&hex[2..4], 16).ok()?,
            u8::from_str_radix(&hex[4..6], 16).ok()?,
        ]);
    }
    let channels = value
        .split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|entry| !entry.is_empty())
        .map(str::parse::<u8>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (channels.len() == 3).then(|| [channels[0], channels[1], channels[2]])
}

pub(crate) fn palette_color(index: i32, palette: &[[u8; 3]]) -> StrokeColor {
    let length = if palette.is_empty() {
        12
    } else {
        palette.len()
    };
    let wrapped = index.rem_euclid(i32::try_from(length).unwrap_or(i32::MAX)) as usize;
    palette
        .get(wrapped)
        .copied()
        .map_or(StrokeColor::PaletteIndex(wrapped as u16), StrokeColor::Rgb)
}

fn module_list_metadata(source: &str, key: &str) -> Option<Vec<String>> {
    Some(
        metadata(source, key)?
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
    )
}

fn module_alias_metadata(source: &str) -> Option<Vec<(String, String)>> {
    Some(
        metadata(source, "Render Map")?
            .split_whitespace()
            .filter_map(|mapping| mapping.split_once('='))
            .map(|(module, action)| (module.to_owned(), action.to_owned()))
            .collect(),
    )
}

fn metadata<'a>(source: &'a str, key: &str) -> Option<&'a str> {
    source.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix("# ")?;
        let (found, value) = rest.split_once(':')?;
        (found.trim() == key).then(|| value.trim())
    })
}

#[derive(Debug, Clone)]
pub enum VisualizerConfig {
    Inspector(InspectorConfig),
    Sequence(SequenceConfig),
    Turtle2d(Turtle2dConfig),
    Turtle3d(Turtle3dConfig),
    AxialTree(AxialTreeConfig),
    Asset(AssetConfig),
    Polygon(PolygonConfig),
    Plot(PlotConfig),
    PlanarMap(PlanarMapConfig),
    SphericalMap(SphericalMapConfig),
    Cellwork3d(Cellwork3dConfig),
}
impl VisualizerConfig {
    pub fn for_kind(kind: VisualizerKind) -> Self {
        match kind {
            VisualizerKind::Inspector => Self::Inspector(InspectorConfig),
            VisualizerKind::Sequence => Self::Sequence(SequenceConfig),
            VisualizerKind::Turtle2d => Self::Turtle2d(Turtle2dConfig::default()),
            VisualizerKind::Turtle3d => Self::Turtle3d(Turtle3dConfig::default()),
            VisualizerKind::AxialTree => Self::AxialTree(AxialTreeConfig),
            VisualizerKind::Asset => Self::Asset(AssetConfig),
            VisualizerKind::Polygon => Self::Polygon(PolygonConfig),
            VisualizerKind::Plot => Self::Plot(PlotConfig),
            VisualizerKind::PlanarMap => Self::PlanarMap(PlanarMapConfig),
            VisualizerKind::SphericalMap => Self::SphericalMap(SphericalMapConfig),
            VisualizerKind::Cellwork3d => Self::Cellwork3d(Cellwork3dConfig),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VisualizationContext {
    pub iterations: usize,
    pub seed: u64,
    pub derivation_backend: Option<&'static str>,
    pub elapsed: Option<Duration>,
}

pub struct VisualizeRequest<'a> {
    pub generation: &'a Generation,
    pub backend: VisualizerBackend,
    pub config: VisualizerConfig,
    pub context: VisualizationContext,
}

/// Build an inspector scene with bounded work quanta and cooperative
/// cancellation. The compatibility [`visualize`] entry point delegates to the
/// same iterative implementation with cancellation disabled.
pub fn stream_inspector(
    request: InspectorStreamRequest<'_>,
    emit: impl FnMut(InspectorProgress) -> Result<(), VisualizeError>,
) -> Result<Scene2d, VisualizeError> {
    visualizers::inspector::stream_cpu(request, emit)
}

/// Axis-aligned bounds accumulated while traversing a two-dimensional turtle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds2d {
    pub min: Point2d,
    pub max: Point2d,
}

impl Bounds2d {
    fn from_line(line: Line2d) -> Self {
        Self {
            min: (line.0.0.min(line.1.0), line.0.1.min(line.1.1)),
            max: (line.0.0.max(line.1.0), line.0.1.max(line.1.1)),
        }
    }

    fn include_line(&mut self, line: Line2d) {
        self.min.0 = self.min.0.min(line.0.0).min(line.1.0);
        self.min.1 = self.min.1.min(line.0.1).min(line.1.1);
        self.max.0 = self.max.0.max(line.0.0).max(line.1.0);
        self.max.1 = self.max.1.max(line.0.1).max(line.1.1);
    }
}

/// Axis-aligned bounds accumulated while traversing a three-dimensional turtle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3d {
    pub min: Point3d,
    pub max: Point3d,
}

impl Bounds3d {
    fn from_line(line: Line3d) -> Self {
        Self {
            min: (
                line.0.0.min(line.1.0),
                line.0.1.min(line.1.1),
                line.0.2.min(line.1.2),
            ),
            max: (
                line.0.0.max(line.1.0),
                line.0.1.max(line.1.1),
                line.0.2.max(line.1.2),
            ),
        }
    }

    fn include_line(&mut self, line: Line3d) {
        self.min.0 = self.min.0.min(line.0.0).min(line.1.0);
        self.min.1 = self.min.1.min(line.0.1).min(line.1.1);
        self.min.2 = self.min.2.min(line.0.2).min(line.1.2);
        self.max.0 = self.max.0.max(line.0.0).max(line.1.0);
        self.max.1 = self.max.1.max(line.0.1).max(line.1.1);
        self.max.2 = self.max.2.max(line.0.2).max(line.1.2);
    }
}

/// Monotonic progress reported by the incremental turtle visualizer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Turtle2dProgress {
    /// Modules and branch nodes consumed from the generation.
    pub items_processed: usize,
    pub modules_processed: usize,
    pub branches_entered: usize,
    pub lines_emitted: usize,
    pub polygons_emitted: usize,
    pub active_branch_depth: usize,
    pub max_branch_depth: usize,
}

/// One bounded piece of an incremental turtle visualization.
#[derive(Debug, Clone, PartialEq)]
pub struct Turtle2dBatch {
    pub lines: Vec<StyledLine2d>,
    pub polygons: Vec<Polygon2d>,
    /// Depth-first generation-module index for each line in [`Self::lines`].
    ///
    /// A module can emit at most one Turtle 2D line, so the vectors always
    /// have equal length. Indices remain stable across CPU and CUDA
    /// visualization and make adjacent-generation lineage usable without
    /// changing the backend-neutral line primitive.
    pub module_indices: Vec<usize>,
    /// Entry position of every processed module in depth-first generation
    /// order, including modules that do not emit geometry.
    ///
    /// Retaining these bounded-batch positions lets adjacent-generation
    /// transitions grow visible successors from an invisible parent's actual
    /// source position. Consumers that do not need correspondence may discard
    /// the vector with the rest of the batch.
    pub module_positions: Vec<IndexedModulePosition2d>,
    /// Bounds of only the lines in this batch.
    pub batch_bounds: Option<Bounds2d>,
    /// Bounds of all lines emitted up to and including this batch.
    pub overall_bounds: Option<Bounds2d>,
    pub progress: Turtle2dProgress,
}

/// Final metadata returned after all turtle batches have been emitted.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Turtle2dStreamSummary {
    pub bounds: Option<Bounds2d>,
    pub progress: Turtle2dProgress,
    /// The backend that produced the emitted batches. This is especially
    /// useful when [`VisualizerBackend::Auto`] selected a fallback.
    pub backend_used: VisualizerBackend,
}

/// Input and cooperative-cancellation hook for incremental turtle traversal.
pub struct Turtle2dStreamRequest<'a> {
    pub generation: &'a Generation,
    pub config: Turtle2dConfig,
    /// Maximum number of generation items consumed between sink calls (and,
    /// consequently, the maximum number of lines in one batch). This is a
    /// batching preference, not a limit on the completed visualization.
    pub batch_size: usize,
    /// Called at every generation-item boundary and before delivering a batch.
    pub is_cancelled: &'a dyn Fn() -> bool,
}

/// Monotonic progress reported by the incremental three-dimensional turtle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Turtle3dProgress {
    /// Modules and branch nodes consumed from the generation.
    pub items_processed: usize,
    pub modules_processed: usize,
    pub branches_entered: usize,
    pub lines_emitted: usize,
    pub polygons_emitted: usize,
    pub active_branch_depth: usize,
    pub max_branch_depth: usize,
}

/// One bounded piece of an incremental three-dimensional turtle visualization.
#[derive(Debug, Clone, PartialEq)]
pub struct Turtle3dBatch {
    pub lines: Vec<StyledLine3d>,
    pub polygons: Vec<Polygon3d>,
    /// Traversal order of the entries in [`Self::lines`] and
    /// [`Self::polygons`]. Entries of each kind consume the next item from the
    /// corresponding vector.
    ///
    /// Keeping the payloads type-separated lets display backends upload them
    /// efficiently, while this tag stream lets consumers that care about
    /// painter/source order reconstruct the original primitive sequence. Its
    /// length always equals `lines.len() + polygons.len()`.
    pub primitive_order: Vec<Turtle3dPrimitiveKind>,
    /// Bounds of only the geometry in this batch.
    pub batch_bounds: Option<Bounds3d>,
    /// Bounds of all geometry emitted up to and including this batch.
    pub overall_bounds: Option<Bounds3d>,
    pub progress: Turtle3dProgress,
}

/// Kind tag used by [`Turtle3dBatch::primitive_order`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turtle3dPrimitiveKind {
    Line,
    Polygon,
}

/// Final metadata returned after all three-dimensional turtle batches are emitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Turtle3dStreamSummary {
    pub bounds: Option<Bounds3d>,
    pub progress: Turtle3dProgress,
    /// Length-weighted reference for interpreting retained source widths in
    /// automatic three-dimensional displays. Raw scene geometry remains
    /// unchanged; targets divide a positive source width by this value before
    /// applying the bounded display multiplier.
    pub width_reference: f64,
    /// Three-dimensional turtle visualization currently uses the complete CPU
    /// reference interpreter.
    pub backend_used: VisualizerBackend,
}

impl Default for Turtle3dStreamSummary {
    fn default() -> Self {
        Self {
            bounds: None,
            progress: Turtle3dProgress::default(),
            width_reference: 1.0,
            backend_used: VisualizerBackend::default(),
        }
    }
}

/// Largest source-style multiplier used by automatic three-dimensional
/// displays after scene-relative normalization.
pub const MAX_NORMALIZED_TURTLE_3D_WIDTH: f64 = 3.0;

/// Incrementally computes the length-weighted source-width reference used by
/// automatic three-dimensional displays.
///
/// Samples with a non-finite or non-positive length do not affect the result.
/// A non-finite or non-positive width contributes zero to the weighted
/// numerator while its valid length still contributes to the denominator.
/// [`Self::width_reference`] therefore always returns a finite value greater
/// than or equal to one, even for independently constructed or malformed
/// scenes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Turtle3dWidthReferenceEstimator {
    length_scale: f64,
    scaled_total_length: f64,
    mean_width: f64,
}

impl Turtle3dWidthReferenceEstimator {
    /// Include one `(segment length, source width)` sample.
    pub fn observe(&mut self, segment_length: f64, source_width: f64) {
        if !segment_length.is_finite() || segment_length <= 0.0 {
            return;
        }
        let width = if source_width.is_finite() && source_width > 0.0 {
            source_width
        } else {
            0.0
        };
        if self.scaled_total_length == 0.0 {
            self.length_scale = segment_length;
            self.scaled_total_length = 1.0;
            self.mean_width = width;
            return;
        }
        let (previous_length, sample_length) = if segment_length > self.length_scale {
            let rescale = self.length_scale / segment_length;
            self.length_scale = segment_length;
            (self.scaled_total_length * rescale, 1.0)
        } else {
            (self.scaled_total_length, segment_length / self.length_scale)
        };
        let next_total = previous_length + sample_length;
        let contribution = sample_length / next_total;
        self.mean_width += contribution * (width - self.mean_width);
        self.scaled_total_length = next_total;
    }

    /// Include one retained three-dimensional line without projecting it.
    pub fn observe_line(&mut self, line: &StyledLine3d) {
        let Line3d(start, end) = line.line;
        let length = (end.0 - start.0)
            .hypot(end.1 - start.1)
            .hypot(end.2 - start.2);
        self.observe(length, line.width);
    }

    /// Return `max(1, sum(length * positive_width) / sum(length))`.
    pub fn width_reference(&self) -> f64 {
        if self.mean_width.is_finite() {
            self.mean_width.max(1.0)
        } else {
            1.0
        }
    }
}

/// Convert a retained raw turtle width into its automatic 3D display
/// multiplier.
///
/// `width_reference` is produced by [`stream_turtle_3d`] and is always finite
/// and at least one. Defensive fallbacks keep independently constructed scenes
/// and decoded metadata bounded as well. Zero and negative widths suppress the
/// segment, matching the retained line contract.
pub fn normalized_turtle_3d_width(source_width: f64, width_reference: f64) -> f64 {
    if !source_width.is_finite() || source_width <= 0.0 {
        return 0.0;
    }
    let reference = if width_reference.is_finite() && width_reference >= 1.0 {
        width_reference
    } else {
        1.0
    };
    (source_width / reference).clamp(0.0, MAX_NORMALIZED_TURTLE_3D_WIDTH)
}

/// Input and cooperative-cancellation hook for incremental 3D turtle traversal.
pub struct Turtle3dStreamRequest<'a> {
    pub generation: &'a Generation,
    pub config: Turtle3dConfig,
    /// Maximum number of generation items consumed between sink calls. This is
    /// a batching preference, not a limit on the completed visualization.
    pub batch_size: usize,
    /// Called at every generation-item boundary and before delivering a batch.
    pub is_cancelled: &'a dyn Fn() -> bool,
}

/// Traverse a turtle generation on CPU without first materializing a complete
/// scene.
///
/// This is the compatibility entry point. Use [`Turtle2dStreamer::stream`] to
/// select `Auto` or `Cuda` and to reuse accelerator state across requests.
///
/// Batches retain source order and contain at most `request.batch_size` lines.
/// A batch can contain no lines when a work quantum only moves or changes the
/// turtle; these progress-only batches keep callers informed during such runs.
/// The sink may move each owned batch directly into a channel for rendering on
/// another thread or in a web worker.
pub fn stream_turtle_2d(
    request: Turtle2dStreamRequest<'_>,
    emit: impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle2dStreamSummary, VisualizeError> {
    visualizers::turtle_2d::stream_cpu(request, emit)
}

/// Traverse a three-dimensional turtle generation on CPU without first
/// materializing a complete scene.
///
/// Batches retain source order and contain at most `request.batch_size` lines.
/// Progress-only batches are delivered when a work quantum moves the turtle or
/// changes state without emitting geometry.
pub fn stream_turtle_3d(
    request: Turtle3dStreamRequest<'_>,
    emit: impl FnMut(Turtle3dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle3dStreamSummary, VisualizeError> {
    visualizers::turtle_3d::stream_cpu(request, emit)
}

/// A reusable turtle executor that retains backend state between requests.
///
/// `Auto` uses CUDA for its supported branch-free scalar input when a native
/// CUDA device is available, and otherwise runs the complete CPU
/// implementation. Structural/style commands rejected during preflight select
/// CPU without beginning accelerator work.
/// It does not restart on CPU after a CUDA kernel has been submitted. Explicit
/// `Cuda` requests never silently fall back. The streamer retains healthy CUDA
/// context and scratch state between calls and discards it after device errors.
#[derive(Default)]
pub struct Turtle2dStreamer {
    state: visualizers::turtle_2d::StreamerState,
}

impl Turtle2dStreamer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stream(
        &mut self,
        backend: VisualizerBackend,
        request: Turtle2dStreamRequest<'_>,
        emit: impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
    ) -> Result<Turtle2dStreamSummary, VisualizeError> {
        visualizers::turtle_2d::stream(&mut self.state, backend, request, emit)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextRole {
    Title,
    Heading,
    Body,
    Muted,
    Error,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Text2d {
    pub position: Point2d,
    pub content: String,
    pub size: f64,
    pub role: TextRole,
}
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive2d {
    Line(StyledLine2d),
    Polygon(Polygon2d),
    Text(Text2d),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Primitive3d {
    Line(StyledLine3d),
    Polygon(Polygon3d),
}
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Scene2d {
    pub primitives: Vec<Primitive2d>,
    /// Exact source background, when one was declared. `None` keeps the
    /// target's normal light/dark themed background.
    pub background: Option<[u8; 3]>,
}
impl Scene2d {
    pub fn lines(lines: Lines2d) -> Self {
        Self {
            primitives: lines
                .0
                .into_iter()
                .map(|line| {
                    Primitive2d::Line(StyledLine2d {
                        line,
                        width: 1.0,
                        color: StrokeColor::ThemeDefault,
                    })
                })
                .collect(),
            background: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Scene3d {
    pub primitives: Vec<Primitive3d>,
    /// Exact source background, when one was declared. `None` keeps the
    /// target's normal light/dark themed background.
    pub background: Option<[u8; 3]>,
}

impl Scene3d {
    /// Apply the stable equal-axis isometric projection used by non-interactive
    /// targets.
    ///
    /// Interactive displays should retain this scene's XYZ coordinates and
    /// choose their own view orientation instead.
    pub fn canonical_projection(&self) -> Scene2d {
        let primitives = self
            .primitives
            .iter()
            .map(|primitive| match primitive {
                Primitive3d::Line(line) => Primitive2d::Line(StyledLine2d {
                    line: Line2d(
                        canonical_project_point_3d(line.line.0),
                        canonical_project_point_3d(line.line.1),
                    ),
                    width: line.width,
                    color: line.color,
                }),
                Primitive3d::Polygon(polygon) => Primitive2d::Polygon(Polygon2d {
                    vertices: polygon
                        .vertices
                        .iter()
                        .copied()
                        .map(canonical_project_point_3d)
                        .collect(),
                    color: polygon.color,
                }),
            })
            .collect();
        Scene2d {
            primitives,
            background: self.background,
        }
    }

    /// Consume this scene and apply the stable equal-axis isometric projection
    /// used by non-interactive targets.
    pub fn into_canonical_projection(self) -> Scene2d {
        let primitives = self
            .primitives
            .into_iter()
            .map(|primitive| match primitive {
                Primitive3d::Line(line) => Primitive2d::Line(StyledLine2d {
                    line: Line2d(
                        canonical_project_point_3d(line.line.0),
                        canonical_project_point_3d(line.line.1),
                    ),
                    width: line.width,
                    color: line.color,
                }),
                Primitive3d::Polygon(polygon) => Primitive2d::Polygon(Polygon2d {
                    vertices: polygon
                        .vertices
                        .into_iter()
                        .map(canonical_project_point_3d)
                        .collect(),
                    color: polygon.color,
                }),
            })
            .collect();
        Scene2d {
            primitives,
            background: self.background,
        }
    }
}

/// Project one turtle-world point using the equal-axis isometric view shared by
/// static targets, generated previews, and interactive displays at startup.
///
/// The view first yaws by +45 degrees around Z and then pitches by
/// `-atan(sqrt(2))` around X before discarding depth.
pub fn canonical_project_point_3d(point: Point3d) -> Point2d {
    let (yaw_sin, yaw_cos) = std::f64::consts::FRAC_PI_4.sin_cos();
    let pitch = -2.0_f64.sqrt().atan();
    let (pitch_sin, pitch_cos) = pitch.sin_cos();
    let yawed_x = point.0 * yaw_cos - point.1 * yaw_sin;
    let yawed_y = point.0 * yaw_sin + point.1 * yaw_cos;
    (yawed_x, yawed_y * pitch_cos - point.2 * pitch_sin)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Visualization {
    Scene2d(Scene2d),
    Scene3d(Scene3d),
}

/// A completed visualization together with the backend that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizationResult {
    pub visualization: Visualization,
    pub backend_used: VisualizerBackend,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualizeError {
    Unimplemented {
        visualizer: VisualizerKind,
        backend: VisualizerBackend,
    },
    BackendUnavailable {
        backend: VisualizerBackend,
    },
    /// The selected backend cannot represent a valid input feature.
    UnsupportedInput {
        visualizer: VisualizerKind,
        backend: VisualizerBackend,
        reason: String,
    },
    /// A backend was selected successfully but failed while preparing or
    /// executing work.
    BackendRuntime {
        backend: VisualizerBackend,
        operation: &'static str,
        reason: String,
    },
    /// The caller no longer needs this visualization.
    Cancelled,
    /// A fallible allocation or size calculation could not be satisfied.
    ResourceExhausted {
        resource: &'static str,
        requested: Option<usize>,
    },
    InvalidConfiguration(String),
}
impl fmt::Display for VisualizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unimplemented {
                visualizer,
                backend,
            } => write!(
                f,
                "visualizer {visualizer} is not implemented for {backend:?}"
            ),
            Self::BackendUnavailable { backend } => {
                write!(f, "visualization backend {backend:?} is unavailable")
            }
            Self::UnsupportedInput {
                visualizer,
                backend,
                reason,
            } => write!(
                f,
                "visualizer {visualizer} does not support this input on {backend:?}: {reason}"
            ),
            Self::BackendRuntime {
                backend,
                operation,
                reason,
            } => write!(
                f,
                "visualization backend {backend:?} failed during {operation}: {reason}"
            ),
            Self::Cancelled => f.write_str("visualization was cancelled"),
            Self::ResourceExhausted {
                resource,
                requested,
            } => match requested {
                Some(requested) => write!(
                    f,
                    "not enough resources for {resource} (requested {requested} entries)"
                ),
                None => write!(f, "not enough resources for {resource}"),
            },
            Self::InvalidConfiguration(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for VisualizeError {}

/// Complete a visualization and report the backend that actually produced it.
pub fn visualize_with_backend(
    request: VisualizeRequest<'_>,
) -> Result<VisualizationResult, VisualizeError> {
    let VisualizeRequest {
        generation,
        backend: requested_backend,
        config,
        context,
    } = request;
    let config = match config {
        VisualizerConfig::Turtle2d(config) => {
            let background = config.background;
            let mut primitives = Vec::new();
            let mut streamer = Turtle2dStreamer::new();
            let summary = streamer.stream(
                requested_backend,
                Turtle2dStreamRequest {
                    generation,
                    config,
                    batch_size: 16 * 1024,
                    is_cancelled: &|| false,
                },
                |batch| {
                    let additional = batch.lines.len().saturating_add(batch.polygons.len());
                    primitives.try_reserve(additional).map_err(|_| {
                        VisualizeError::ResourceExhausted {
                            resource: "complete turtle scene",
                            requested: Some(additional),
                        }
                    })?;
                    primitives.extend(batch.lines.into_iter().map(Primitive2d::Line));
                    primitives.extend(batch.polygons.into_iter().map(Primitive2d::Polygon));
                    Ok(())
                },
            )?;
            return Ok(VisualizationResult {
                visualization: Visualization::Scene2d(Scene2d {
                    primitives,
                    background,
                }),
                backend_used: summary.backend_used,
            });
        }
        config => config,
    };

    let backend = match requested_backend {
        VisualizerBackend::Auto => VisualizerBackend::Cpu,
        value => value,
    };
    if backend == VisualizerBackend::Cuda
        && !cfg!(all(feature = "cuda", not(target_arch = "wasm32")))
    {
        return Err(VisualizeError::BackendUnavailable { backend });
    }
    macro_rules! call {
        ($module:ident, $config:expr, $scene:ident) => {{
            let scene = match backend {
                VisualizerBackend::Cpu => {
                    visualizers::$module::visualize_cpu(generation, $config, context)
                }
                VisualizerBackend::Cuda => {
                    visualizers::$module::visualize_cuda(generation, $config, context)
                }
                VisualizerBackend::Auto => unreachable!(),
            }?;
            Ok(VisualizationResult {
                visualization: Visualization::$scene(scene),
                backend_used: backend,
            })
        }};
    }
    match config {
        VisualizerConfig::Inspector(c) => call!(inspector, c, Scene2d),
        VisualizerConfig::Sequence(c) => call!(sequence, c, Scene2d),
        VisualizerConfig::Turtle2d(_) => unreachable!("turtle handled above"),
        VisualizerConfig::Turtle3d(c) => call!(turtle_3d, c, Scene3d),
        VisualizerConfig::AxialTree(c) => call!(axial_tree, c, Scene2d),
        VisualizerConfig::Asset(c) => call!(asset, c, Scene2d),
        VisualizerConfig::Polygon(c) => call!(polygon, c, Scene2d),
        VisualizerConfig::Plot(c) => call!(plot, c, Scene2d),
        VisualizerConfig::PlanarMap(c) => call!(planar_map, c, Scene2d),
        VisualizerConfig::SphericalMap(c) => call!(spherical_map, c, Scene2d),
        VisualizerConfig::Cellwork3d(c) => call!(cellwork_3d, c, Scene2d),
    }
}

/// Complete a visualization, discarding backend-selection metadata.
pub fn visualize(request: VisualizeRequest<'_>) -> Result<Visualization, VisualizeError> {
    visualize_with_backend(request).map(|result| result.visualization)
}

#[cfg(test)]
mod tests {
    use super::*;
    use braken::{GenerationItem, Module};

    #[test]
    fn visualizer_metadata_and_names_round_trip() {
        for kind in VisualizerKind::ALL {
            assert_eq!(kind.as_str().parse(), Ok(kind));
        }
        assert_eq!(
            visualizer_metadata("# Visualizer: turtle_2d"),
            Ok(Some(VisualizerKind::Turtle2d))
        );
        assert!(visualizer_metadata("# Visualizer: inspector\n# Visualizer: plot").is_err());
    }

    #[test]
    fn inspector_produces_a_text_scene() {
        let generation = Generation(vec![GenerationItem::Module(Module::new("F", Vec::new()))]);
        let Visualization::Scene2d(scene) = visualize(VisualizeRequest {
            generation: &generation,
            backend: VisualizerBackend::Cpu,
            config: VisualizerConfig::Inspector(InspectorConfig),
            context: VisualizationContext {
                iterations: 3,
                seed: 42,
                ..Default::default()
            },
        })
        .unwrap() else {
            panic!("inspector returned a non-2D scene");
        };
        assert!(
            scene
                .primitives
                .iter()
                .any(|item| matches!(item, Primitive2d::Text(text) if text.content == "Inspector"))
        );
        assert!(scene.primitives.iter().any(
            |item| matches!(item, Primitive2d::Text(text) if text.content.contains("Seed: 42"))
        ));
    }

    #[test]
    fn completed_visualization_reports_the_backend_used() {
        let generation = Generation(vec![GenerationItem::Module(Module::new("F", Vec::new()))]);
        let result = visualize_with_backend(VisualizeRequest {
            generation: &generation,
            backend: VisualizerBackend::Cpu,
            config: VisualizerConfig::Turtle2d(Turtle2dConfig::default()),
            context: VisualizationContext::default(),
        })
        .unwrap();
        assert_eq!(result.backend_used, VisualizerBackend::Cpu);
        assert!(matches!(result.visualization, Visualization::Scene2d(_)));
    }

    #[test]
    fn turtle_3d_returns_retained_xyz_geometry() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("PitchUp", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let result = visualize_with_backend(VisualizeRequest {
            generation: &generation,
            backend: VisualizerBackend::Auto,
            config: VisualizerConfig::Turtle3d(config),
            context: VisualizationContext::default(),
        })
        .unwrap();

        let Visualization::Scene3d(scene) = result.visualization else {
            panic!("turtle_3d returned a projected 2D scene");
        };
        assert_eq!(result.backend_used, VisualizerBackend::Cpu);
        assert!(matches!(
            scene.primitives.as_slice(),
            [Primitive3d::Line(StyledLine3d {
                line: Line3d((0.0, 0.0, 0.0), (_, _, z)),
                ..
            })] if (*z - 1.0).abs() < 1.0e-12
        ));
    }

    #[test]
    fn canonical_projection_preserves_scene_style_and_background() {
        let scene = Scene3d {
            primitives: vec![
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                    width: 2.5,
                    color: StrokeColor::Rgb([1, 2, 3]),
                }),
                Primitive3d::Polygon(Polygon3d {
                    vertices: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
                    color: StrokeColor::PaletteIndex(4),
                }),
            ],
            background: Some([9, 8, 7]),
        };

        let projected = scene.canonical_projection();
        assert_eq!(
            projected,
            scene.clone().into_canonical_projection(),
            "borrowed and consuming projection paths must stay equivalent"
        );
        assert_eq!(projected.background, scene.background);
        let Primitive2d::Line(line) = &projected.primitives[0] else {
            panic!("first primitive was not a line");
        };
        assert_eq!(line.line.0, canonical_project_point_3d((1.0, 2.0, 3.0)));
        assert_eq!(line.line.1, canonical_project_point_3d((4.0, 5.0, 6.0)));
        assert_eq!(line.width, 2.5);
        assert_eq!(line.color, StrokeColor::Rgb([1, 2, 3]));
        let Primitive2d::Polygon(polygon) = &projected.primitives[1] else {
            panic!("second primitive was not a polygon");
        };
        assert_eq!(
            polygon.vertices,
            [
                canonical_project_point_3d((0.0, 0.0, 0.0)),
                canonical_project_point_3d((1.0, 0.0, 1.0)),
                canonical_project_point_3d((0.0, 1.0, 0.0)),
            ]
        );
        assert_eq!(polygon.color, StrokeColor::PaletteIndex(4));
    }

    #[test]
    fn canonical_projection_is_equal_axis_isometric() {
        let origin = canonical_project_point_3d((0.0, 0.0, 0.0));
        let projected_length = |point: Point3d| {
            let projected = canonical_project_point_3d(point);
            (projected.0 - origin.0).hypot(projected.1 - origin.1)
        };
        let x = projected_length((1.0, 0.0, 0.0));
        let y = projected_length((0.0, 1.0, 0.0));
        let z = projected_length((0.0, 0.0, 1.0));

        assert!((x - y).abs() < 1.0e-12, "{x} != {y}");
        assert!((x - z).abs() < 1.0e-12, "{x} != {z}");
    }

    #[test]
    fn static_targets_use_the_canonical_3d_projection() {
        let scene = Scene3d {
            primitives: vec![Primitive3d::Line(StyledLine3d {
                line: Line3d((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };
        let projected = scene.canonical_projection();
        let Primitive2d::Line(line) = &projected.primitives[0] else {
            panic!("canonical projection changed primitive kind");
        };
        let expected_color = targets::spatial_theme_default_rgb8_at(0.5, targets::Palette::Light);
        let svg = targets::svg::encode_3d(&scene, targets::Palette::Light);

        assert!(
            svg.contains(&format!(
                "x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"",
                line.line.0.0, -line.line.0.1, line.line.1.0, -line.line.1.1
            )),
            "static SVG did not use the canonical projection: {svg}"
        );
        assert!(svg.contains(&format!(
            "stroke=\"#{:02x}{:02x}{:02x}\"",
            expected_color[0], expected_color[1], expected_color[2]
        )));
        assert_eq!(
            targets::tui::encode_3d(&scene, (20, 8)),
            targets::tui::encode(&projected, (20, 8))
        );
    }

    #[test]
    fn svg_escapes_text_and_tui_draws_braille() {
        let text_scene = Scene2d {
            primitives: vec![Primitive2d::Text(Text2d {
                position: (0.0, 10.0),
                content: "a < b & c".into(),
                size: 12.0,
                role: TextRole::Body,
            })],
            background: None,
        };
        assert!(
            targets::svg::encode(&text_scene, targets::Palette::Light).contains("a &lt; b &amp; c")
        );
        let line_scene = Scene2d::lines(Lines2d(vec![Line2d((0.0, 0.0), (1.0, 1.0))]));
        assert!(
            targets::tui::encode(&line_scene, (8, 4))
                .chars()
                .any(|value| ('\u{2801}'..='\u{28ff}').contains(&value))
        );
    }

    #[test]
    fn svg_multiplies_its_adaptive_width_by_the_line_style() {
        let scene = Scene2d {
            primitives: vec![Primitive2d::Line(StyledLine2d {
                line: Line2d((0.0, 0.0), (10.0, 0.0)),
                width: 3.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };
        let svg = targets::svg::encode(&scene, targets::Palette::Light);
        assert!(svg.contains("stroke-width=\"0.648"), "{svg}");
        assert!(svg.contains("stroke-linecap=\"round\""), "{svg}");
    }

    #[test]
    fn turtle_metadata_preserves_theme_defaults_and_accepts_exact_source_colors() {
        let default = Turtle2dConfig::default().with_source_metadata("axiom F;");
        assert_eq!(default.initial_color, StrokeColor::ThemeDefault);
        assert!(default.background.is_none());

        let configured = Turtle2dConfig::default().with_source_metadata(
            "# Palette: #102030, #abcdef\n\
             # Initial Color: -1\n\
             # Stroke Color: #123456\n\
             # Background: 4, 5, 6\n\
             # Color Increment: 3\n\
             # Angle Increment: 2.5\n\
             let SCALE = 1.75;\n",
        );
        assert_eq!(
            configured.initial_color,
            StrokeColor::Rgb([0x12, 0x34, 0x56])
        );
        assert_eq!(configured.palette, [[0x10, 0x20, 0x30], [0xab, 0xcd, 0xef]]);
        assert_eq!(configured.background, Some([4, 5, 6]));
        assert_eq!(configured.color_increment, 3);
        assert_eq!(configured.turn_angle_increment, 2.5_f64.to_radians());
        assert_eq!(configured.scale_multiplier, 1.75);
    }

    #[test]
    fn svg_uses_explicit_turtle_colors_and_backgrounds() {
        let scene = Scene2d {
            primitives: vec![Primitive2d::Line(StyledLine2d {
                line: Line2d((0.0, 0.0), (1.0, 1.0)),
                width: 1.0,
                color: StrokeColor::Rgb([18, 52, 86]),
            })],
            background: Some([1, 2, 3]),
        };
        let svg = targets::svg::encode(&scene, targets::Palette::Dark);
        assert!(svg.contains("fill=\"#010203\""), "{svg}");
        assert!(svg.contains("stroke=\"#123456\""), "{svg}");
    }
}
