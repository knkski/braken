//! Structured-clone payloads shared by the browser UI and render worker.
//!
//! The worker owns the expensive derivation, one exact generation cache entry,
//! and visualization. Completed workers remain idle for reuse. The UI sends an
//! explicit cancellation command for superseded work; asynchronous derivation
//! observes it cooperatively after yielding at safe boundaries. The current
//! synchronous visualization streams still prevent the browser event loop from
//! dispatching a newly posted cancellation until that phase returns, so the UI
//! watchdog may recreate an unresponsive worker. Replacement does not normally
//! discard the worker or its reusable backend state.
//!
//! Derivation, visualization, and display remain distinct: the worker may use
//! WebGPU or CPU to derive a generation, currently emits backend-neutral scene
//! data, and leaves rasterization to the UI renderer. Each backend string names
//! its explicit derivation or visualization stage; display remains UI-owned.
//! Adjacent iteration requests transfer the exact target lines plus compact
//! morph records; each request completes before the GUI starts the next step,
//! bounding the worker to one source/target generation pair.
//! Render results also carry a bounded shared-IR tooling snapshot. This keeps
//! grammar compilation and disassembly inside the Worker rather than the UI
//! event loop; JSON exceeding the tooling transfer limit is omitted without
//! changing derivation.

use braken_viz::{MorphCoordinateSpace, StrokeColor};
use serde::{Deserialize, Serialize};

pub(crate) const LINE_TRANSFER_CHUNK_LINES: usize = 65_536;
pub(crate) const LINE_2D_TRANSFER_VALUES: usize = 6;
pub(crate) const LINE_3D_TRANSFER_VALUES: usize = 8;
/// Backwards-compatible name used by the 2D transition and line builders.
pub(crate) const LINE_TRANSFER_VALUES: usize = LINE_2D_TRANSFER_VALUES;
pub(crate) const MORPH_TRANSFER_CHUNK_LINES: usize = 32_768;
pub(crate) const MORPH_TRANSFER_VALUES: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerTransferLayout {
    pub line_chunks: usize,
    pub line_values: usize,
    pub morph_chunks: usize,
    pub morph_values: usize,
}

/// Compact exact token for the typed-array scene protocol. Palette indices are
/// non-negative, packed RGB values are negative, and NaN retains themed
/// default styling. All packed integers fit exactly in an IEEE-754 `f32`.
pub(crate) fn encode_stroke_color(color: StrokeColor) -> f32 {
    match color {
        StrokeColor::ThemeDefault => f32::NAN,
        StrokeColor::PaletteIndex(index) => f32::from(index),
        StrokeColor::Rgb([red, green, blue]) => {
            let packed = (u32::from(red) << 16) | (u32::from(green) << 8) | u32::from(blue);
            -((packed + 1) as f32)
        }
    }
}

#[allow(dead_code)] // The Worker binary only encodes; the UI binary decodes.
pub(crate) fn decode_stroke_color(token: f32) -> StrokeColor {
    if token.is_nan() {
        return StrokeColor::ThemeDefault;
    }
    if token >= 0.0 {
        return StrokeColor::PaletteIndex(token.min(f32::from(u16::MAX)) as u16);
    }
    let packed = ((-token) as u32).saturating_sub(1).min(0x00ff_ffff);
    StrokeColor::Rgb([
        ((packed >> 16) & 0xff) as u8,
        ((packed >> 8) & 0xff) as u8,
        (packed & 0xff) as u8,
    ])
}

pub(crate) const fn encode_morph_space(space: MorphCoordinateSpace) -> f32 {
    match space {
        MorphCoordinateSpace::Source => 0.0,
        MorphCoordinateSpace::Target => 1.0,
    }
}

#[allow(dead_code)] // The Worker binary only encodes; the UI binary decodes.
pub(crate) fn decode_morph_space(token: f32) -> MorphCoordinateSpace {
    if token == 1.0 {
        MorphCoordinateSpace::Target
    } else {
        MorphCoordinateSpace::Source
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WorkerCommand {
    Run(WorkerRenderRequest),
    Cancel { request_id: u64 },
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerRenderRequest {
    pub request_id: u64,
    pub source: String,
    #[serde(default)]
    pub ir_json: Option<String>,
    pub iterations: usize,
    pub angle: f32,
    pub visualizer: String,
    pub turtle: WorkerTurtleConfig,
    pub orientation_anchor: Option<String>,
    pub seed: u64,
    #[serde(default)]
    pub float_width: WorkerFloatWidth,
    #[serde(default)]
    pub ambiguous_rules: WorkerAmbiguousRules,
    pub transition: Option<WorkerIterationTransition>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum WorkerFloatWidth {
    F32,
    #[default]
    F64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum WorkerAmbiguousRules {
    First,
    Error,
    #[default]
    Uniform,
}

/// Maximum IR tooling text accepted across the shared Worker boundary.
pub(crate) const IR_TOOLING_TEXT_LIMIT: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct WorkerIterationTransition {
    pub from_iteration: usize,
    pub to_iteration: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerTurtleConfig {
    pub initial_angle: f64,
    pub default_step: f64,
    pub scale_multiplier: f64,
    pub initial_width: f64,
    pub width_increment: f64,
    pub turn_angle_increment: f64,
    pub initial_color: WorkerStrokeColor,
    pub color_increment: i32,
    pub palette: Vec<[u8; 3]>,
    pub background: Option<[u8; 3]>,
    pub draw_modules: Vec<String>,
    pub move_modules: Vec<String>,
    pub module_aliases: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum WorkerStrokeColor {
    ThemeDefault,
    PaletteIndex(u16),
    Rgb([u8; 3]),
}

impl From<StrokeColor> for WorkerStrokeColor {
    fn from(color: StrokeColor) -> Self {
        match color {
            StrokeColor::ThemeDefault => Self::ThemeDefault,
            StrokeColor::PaletteIndex(index) => Self::PaletteIndex(index),
            StrokeColor::Rgb(rgb) => Self::Rgb(rgb),
        }
    }
}

impl From<WorkerStrokeColor> for StrokeColor {
    fn from(color: WorkerStrokeColor) -> Self {
        match color {
            WorkerStrokeColor::ThemeDefault => Self::ThemeDefault,
            WorkerStrokeColor::PaletteIndex(index) => Self::PaletteIndex(index),
            WorkerStrokeColor::Rgb(rgb) => Self::Rgb(rgb),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WorkerEvent {
    Ready,
    Fatal {
        message: String,
    },
    Progress {
        request_id: u64,
        phase: String,
        phase_completed: usize,
        phase_total: Option<usize>,
        completed_iterations: usize,
        total_iterations: usize,
        modules: usize,
        items: usize,
        elapsed_millis: u64,
    },
    Cancelled {
        request_id: u64,
    },
    Finished {
        request_id: u64,
        result: Result<WorkerRenderResult, String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerRenderResult {
    /// The transferable `Float32Array` chunks accompanying this metadata contain
    /// exactly `line_count * scene.line_transfer_values()` values. Two-
    /// dimensional records are `x1,y1,x2,y2,width,color`; three-dimensional
    /// records are `x1,y1,z1,x2,y2,z2,width,color`.
    pub line_count: usize,
    /// Sum of positive, finite unstyled target segment lengths.
    pub total_line_length: Option<f64>,
    /// Length-weighted source-width reference for automatic 3D display.
    /// Two-dimensional scenes carry `None`; three-dimensional scenes carry a
    /// finite value greater than or equal to one.
    pub width_reference: Option<f64>,
    pub scene: WorkerScene,
    pub background: Option<[u8; 3]>,
    pub elapsed_millis: u64,
    /// Backend that derived the generation.
    pub derivation_backend: String,
    /// Backend that converted the generation into geometry.
    pub visualization_backend: String,
    /// Bounded, deterministic shared-IR disassembly produced inside the Worker.
    pub ir_disassembly: String,
    /// Versioned IR JSON when it fits the GUI tooling transfer limit.
    pub ir_json: Option<String>,
    /// Present when a second transferable `morphs` array accompanies the
    /// target line chunks. Each morph uses [`MORPH_TRANSFER_VALUES`] floats.
    pub transition: Option<WorkerTransitionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WorkerScene {
    TwoD {
        polygons: Vec<WorkerPolygon>,
        texts: Vec<WorkerText>,
        bounds: Option<[f32; 4]>,
    },
    ThreeD {
        polygons: Vec<WorkerPolygon3d>,
        bounds: Option<[f32; 6]>,
    },
}

impl WorkerScene {
    #[allow(dead_code)] // Each frontend validates the stride; the Worker only encodes it.
    pub(crate) const fn line_transfer_values(&self) -> usize {
        match self {
            Self::TwoD { .. } => LINE_2D_TRANSFER_VALUES,
            Self::ThreeD { .. } => LINE_3D_TRANSFER_VALUES,
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::TwoD {
                polygons,
                texts,
                bounds,
            } => {
                validate_bounds(bounds.as_ref().map(|bounds| bounds.as_slice()), 2)?;
                checked_element_count(polygons.len(), texts.len(), "two-dimensional scene")?;
                for polygon in polygons {
                    validate_vertices(&polygon.vertices, "two-dimensional polygon")?;
                }
                for text in texts {
                    if !text.x.is_finite() || !text.y.is_finite() || !text.size.is_finite() {
                        return Err(String::from(
                            "browser Worker produced non-finite text geometry",
                        ));
                    }
                }
            }
            Self::ThreeD { polygons, bounds } => {
                validate_bounds(bounds.as_ref().map(|bounds| bounds.as_slice()), 3)?;
                for polygon in polygons {
                    validate_vertices(&polygon.vertices, "three-dimensional polygon")?;
                }
            }
        }
        Ok(())
    }
}

impl WorkerRenderResult {
    /// Validate metadata and derive the exact transferable array shape before
    /// either buffer is posted. Frontends separately repeat the boundary
    /// checks because transferred arrays are not part of this JSON metadata.
    pub(crate) fn transfer_layout(&self) -> Result<WorkerTransferLayout, String> {
        let tooling_limit = IR_TOOLING_TEXT_LIMIT;
        if self.ir_disassembly.len() > tooling_limit
            || self
                .ir_json
                .as_ref()
                .is_some_and(|json| json.len() > tooling_limit)
        {
            return Err(String::from(
                "browser Worker IR tooling payload exceeds the transfer limit",
            ));
        }
        self.scene.validate()?;
        match &self.scene {
            WorkerScene::TwoD {
                polygons,
                texts,
                bounds,
            } => {
                if self.width_reference.is_some() {
                    return Err(String::from(
                        "browser Worker attached a 3D width reference to a two-dimensional scene",
                    ));
                }
                self.line_count
                    .checked_add(polygons.len())
                    .and_then(|count| count.checked_add(texts.len()))
                    .ok_or_else(|| {
                        String::from("browser Worker two-dimensional element count overflow")
                    })?;
                if self.line_count > 0 && bounds.is_none() {
                    return Err(String::from(
                        "browser Worker omitted bounds for two-dimensional lines",
                    ));
                }
            }
            WorkerScene::ThreeD { polygons, bounds } => {
                if !self
                    .width_reference
                    .is_some_and(|reference| reference.is_finite() && reference >= 1.0)
                {
                    return Err(String::from(
                        "browser Worker produced an invalid 3D width reference",
                    ));
                }
                self.line_count.checked_add(polygons.len()).ok_or_else(|| {
                    String::from("browser Worker three-dimensional element count overflow")
                })?;
                let mut previous_lines_before = 0usize;
                for (index, polygon) in polygons.iter().enumerate() {
                    if polygon.lines_before > self.line_count {
                        return Err(String::from(
                            "browser Worker spatial polygon order exceeds the line count",
                        ));
                    }
                    if index > 0 && polygon.lines_before < previous_lines_before {
                        return Err(String::from(
                            "browser Worker spatial polygon order is not monotonic",
                        ));
                    }
                    previous_lines_before = polygon.lines_before;
                }
                if bounds.is_none()
                    && (self.line_count > 0
                        || polygons.iter().any(|polygon| !polygon.vertices.is_empty()))
                {
                    return Err(String::from(
                        "browser Worker omitted bounds for three-dimensional geometry",
                    ));
                }
            }
        }
        if self
            .total_line_length
            .is_some_and(|length| !length.is_finite() || length <= 0.0)
        {
            return Err(String::from(
                "browser Worker produced an invalid target line-length summary",
            ));
        }
        let (line_chunks, line_values) = checked_transfer_layout(
            self.line_count,
            LINE_TRANSFER_CHUNK_LINES,
            self.scene.line_transfer_values(),
            "line",
        )?;
        let (morph_chunks, morph_values) = match (&self.scene, &self.transition) {
            (WorkerScene::ThreeD { .. }, Some(_)) => {
                return Err(String::from(
                    "three-dimensional Worker scenes cannot contain transition metadata",
                ));
            }
            (_, Some(transition)) => {
                if transition.from_iteration.abs_diff(transition.to_iteration) != 1 {
                    return Err(String::from(
                        "browser Worker transition iterations are not adjacent",
                    ));
                }
                validate_bounds(
                    transition
                        .source_bounds
                        .as_ref()
                        .map(|bounds| bounds.as_slice()),
                    2,
                )?;
                if transition
                    .source_total_line_length
                    .is_some_and(|length| !length.is_finite() || length <= 0.0)
                {
                    return Err(String::from(
                        "browser Worker produced an invalid source line-length summary",
                    ));
                }
                transition
                    .source_line_count
                    .checked_add(self.line_count)
                    .ok_or_else(|| String::from("browser Worker transition line count overflow"))?;
                checked_transfer_layout(
                    transition.morph_count,
                    MORPH_TRANSFER_CHUNK_LINES,
                    MORPH_TRANSFER_VALUES,
                    "morph",
                )?
            }
            (_, None) => (0, 0),
        };
        Ok(WorkerTransferLayout {
            line_chunks,
            line_values,
            morph_chunks,
            morph_values,
        })
    }
}

fn checked_transfer_layout(
    records: usize,
    records_per_chunk: usize,
    values_per_record: usize,
    kind: &str,
) -> Result<(usize, usize), String> {
    if records_per_chunk == 0 || values_per_record == 0 {
        return Err(format!("browser Worker {kind} transfer has a zero stride"));
    }
    let values = records
        .checked_mul(values_per_record)
        .ok_or_else(|| format!("browser Worker {kind} transfer size overflow"))?;
    let chunks = records.div_ceil(records_per_chunk);
    Ok((chunks, values))
}

pub(crate) fn expected_chunk_values(
    records: usize,
    chunk_index: usize,
    records_per_chunk: usize,
    values_per_record: usize,
    kind: &str,
) -> Result<usize, String> {
    let (chunks, _) = checked_transfer_layout(records, records_per_chunk, values_per_record, kind)?;
    if chunk_index >= chunks {
        return Err(format!(
            "browser Worker {kind} transfer contains an unexpected chunk"
        ));
    }
    let consumed = chunk_index
        .checked_mul(records_per_chunk)
        .ok_or_else(|| format!("browser Worker {kind} chunk offset overflow"))?;
    records
        .checked_sub(consumed)
        .and_then(|remaining| {
            remaining
                .min(records_per_chunk)
                .checked_mul(values_per_record)
        })
        .ok_or_else(|| format!("browser Worker {kind} chunk size overflow"))
}

fn checked_element_count(left: usize, right: usize, kind: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("browser Worker {kind} element count overflow"))
}

fn validate_bounds(bounds: Option<&[f32]>, dimensions: usize) -> Result<(), String> {
    let Some(bounds) = bounds else {
        return Ok(());
    };
    if bounds.len() != dimensions.saturating_mul(2)
        || !bounds.iter().copied().all(f32::is_finite)
        || bounds.chunks_exact(2).any(|pair| pair[0] > pair[1])
    {
        return Err(format!(
            "browser Worker produced invalid {dimensions}D bounds"
        ));
    }
    Ok(())
}

fn validate_vertices<const DIMENSIONS: usize>(
    vertices: &[[f64; DIMENSIONS]],
    kind: &str,
) -> Result<(), String> {
    let values = vertices
        .len()
        .checked_mul(DIMENSIONS)
        .ok_or_else(|| format!("browser Worker {kind} vertex count overflow"))?;
    if values > 0
        && vertices
            .iter()
            .flatten()
            .copied()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(format!(
            "browser Worker produced a non-finite {kind} vertex"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerPolygon {
    pub vertices: Vec<[f64; 2]>,
    pub color: WorkerStrokeColor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerPolygon3d {
    pub vertices: Vec<[f64; 3]>,
    pub color: WorkerStrokeColor,
    /// Number of line primitives preceding this polygon in turtle traversal
    /// order. Values are cumulative across streamed batches.
    pub lines_before: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerTransitionResult {
    pub from_iteration: usize,
    pub to_iteration: usize,
    pub morph_count: usize,
    pub source_bounds: Option<[f32; 4]>,
    pub source_line_count: usize,
    /// Sum of positive, finite unstyled source segment lengths.
    pub source_total_line_length: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerText {
    pub x: f64,
    pub y: f64,
    pub content: String,
    pub size: f64,
    pub role: WorkerTextRole,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum WorkerTextRole {
    Title,
    Heading,
    Body,
    Muted,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(
        line_count: usize,
        scene: WorkerScene,
        transition: Option<WorkerTransitionResult>,
    ) -> WorkerRenderResult {
        let width_reference = matches!(&scene, WorkerScene::ThreeD { .. }).then_some(1.0);
        WorkerRenderResult {
            line_count,
            total_line_length: None,
            width_reference,
            scene,
            background: None,
            elapsed_millis: 0,
            derivation_backend: String::from("CPU"),
            visualization_backend: String::from("CPU"),
            ir_disassembly: String::from("test IR"),
            ir_json: None,
            transition,
        }
    }

    #[test]
    fn compact_stroke_tokens_round_trip_every_variant() {
        for color in [
            StrokeColor::ThemeDefault,
            StrokeColor::PaletteIndex(0),
            StrokeColor::PaletteIndex(u16::MAX),
            StrokeColor::Rgb([0, 0, 0]),
            StrokeColor::Rgb([12, 34, 56]),
            StrokeColor::Rgb([255, 255, 255]),
        ] {
            assert_eq!(decode_stroke_color(encode_stroke_color(color)), color);
        }
    }

    #[test]
    fn oversized_ir_tooling_payload_is_rejected_at_the_transfer_boundary() {
        let mut result = result(
            0,
            WorkerScene::TwoD {
                polygons: Vec::new(),
                texts: Vec::new(),
                bounds: None,
            },
            None,
        );
        result.ir_disassembly = "x".repeat(IR_TOOLING_TEXT_LIMIT + 1);
        assert!(result.transfer_layout().is_err());
    }

    #[test]
    fn scene_kind_selects_the_transfer_stride() {
        assert_eq!(
            WorkerScene::TwoD {
                polygons: Vec::new(),
                texts: Vec::new(),
                bounds: None,
            }
            .line_transfer_values(),
            LINE_2D_TRANSFER_VALUES,
        );
        assert_eq!(
            WorkerScene::ThreeD {
                polygons: Vec::new(),
                bounds: None,
            }
            .line_transfer_values(),
            LINE_3D_TRANSFER_VALUES,
        );
    }

    #[test]
    fn spatial_layout_uses_exact_eight_value_chunks() {
        let line_count = LINE_TRANSFER_CHUNK_LINES + 1;
        let layout = result(
            line_count,
            WorkerScene::ThreeD {
                polygons: Vec::new(),
                bounds: Some([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0]),
            },
            None,
        )
        .transfer_layout()
        .expect("valid spatial metadata");

        assert_eq!(layout.line_chunks, 2);
        assert_eq!(layout.line_values, line_count * LINE_3D_TRANSFER_VALUES);
        assert_eq!(layout.morph_chunks, 0);
        assert_eq!(layout.morph_values, 0);
        assert_eq!(
            expected_chunk_values(
                line_count,
                0,
                LINE_TRANSFER_CHUNK_LINES,
                LINE_3D_TRANSFER_VALUES,
                "line",
            )
            .expect("first chunk"),
            LINE_TRANSFER_CHUNK_LINES * LINE_3D_TRANSFER_VALUES,
        );
        assert_eq!(
            expected_chunk_values(
                line_count,
                1,
                LINE_TRANSFER_CHUNK_LINES,
                LINE_3D_TRANSFER_VALUES,
                "line",
            )
            .expect("last chunk"),
            LINE_3D_TRANSFER_VALUES,
        );
    }

    #[test]
    fn planar_transition_keeps_existing_six_and_twenty_four_value_records() {
        let layout = result(
            2,
            WorkerScene::TwoD {
                polygons: Vec::new(),
                texts: Vec::new(),
                bounds: Some([-1.0, 1.0, -1.0, 1.0]),
            },
            Some(WorkerTransitionResult {
                from_iteration: 2,
                to_iteration: 3,
                morph_count: 3,
                source_bounds: Some([-1.0, 1.0, -1.0, 1.0]),
                source_line_count: 2,
                source_total_line_length: Some(2.0),
            }),
        )
        .transfer_layout()
        .expect("valid planar transition metadata");

        assert_eq!(layout.line_values, 2 * LINE_2D_TRANSFER_VALUES);
        assert_eq!(layout.morph_values, 3 * MORPH_TRANSFER_VALUES);
        assert_eq!(layout.line_chunks, 1);
        assert_eq!(layout.morph_chunks, 1);
    }

    #[test]
    fn malformed_spatial_metadata_is_rejected() {
        let invalid_bounds = result(
            0,
            WorkerScene::ThreeD {
                polygons: Vec::new(),
                bounds: Some([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            },
            None,
        );
        assert!(invalid_bounds.transfer_layout().is_err());

        let invalid_polygon = result(
            0,
            WorkerScene::ThreeD {
                polygons: vec![WorkerPolygon3d {
                    vertices: vec![[0.0, f64::NAN, 0.0]],
                    color: WorkerStrokeColor::ThemeDefault,
                    lines_before: 0,
                }],
                bounds: None,
            },
            None,
        );
        assert!(invalid_polygon.transfer_layout().is_err());

        let invalid_transition = result(
            0,
            WorkerScene::ThreeD {
                polygons: Vec::new(),
                bounds: None,
            },
            Some(WorkerTransitionResult {
                from_iteration: 0,
                to_iteration: 1,
                morph_count: 0,
                source_bounds: None,
                source_line_count: 0,
                source_total_line_length: None,
            }),
        );
        assert!(invalid_transition.transfer_layout().is_err());

        for width_reference in [None, Some(f64::NAN), Some(0.999)] {
            let mut invalid_reference = result(
                0,
                WorkerScene::ThreeD {
                    polygons: Vec::new(),
                    bounds: None,
                },
                None,
            );
            invalid_reference.width_reference = width_reference;
            assert!(
                invalid_reference
                    .transfer_layout()
                    .expect_err("invalid 3D width reference must be rejected")
                    .contains("invalid 3D width reference")
            );
        }

        let mut planar_reference = result(
            0,
            WorkerScene::TwoD {
                polygons: Vec::new(),
                texts: Vec::new(),
                bounds: None,
            },
            None,
        );
        planar_reference.width_reference = Some(1.0);
        assert!(
            planar_reference
                .transfer_layout()
                .expect_err("2D metadata must not carry a 3D width reference")
                .contains("3D width reference")
        );
    }

    fn ordered_polygon(lines_before: usize) -> WorkerPolygon3d {
        WorkerPolygon3d {
            vertices: Vec::new(),
            color: WorkerStrokeColor::ThemeDefault,
            lines_before,
        }
    }

    #[test]
    fn spatial_polygon_order_allows_consecutive_polygons() {
        let metadata = result(
            2,
            WorkerScene::ThreeD {
                polygons: vec![ordered_polygon(1), ordered_polygon(1), ordered_polygon(2)],
                bounds: Some([0.0; 6]),
            },
            None,
        );

        metadata
            .transfer_layout()
            .expect("equal cumulative line offsets are valid");
    }

    #[test]
    fn malformed_spatial_polygon_order_is_rejected() {
        let decreasing = result(
            2,
            WorkerScene::ThreeD {
                polygons: vec![ordered_polygon(2), ordered_polygon(1)],
                bounds: Some([0.0; 6]),
            },
            None,
        );
        assert!(
            decreasing
                .transfer_layout()
                .expect_err("decreasing offsets must be rejected")
                .contains("not monotonic")
        );

        let past_end = result(
            2,
            WorkerScene::ThreeD {
                polygons: vec![ordered_polygon(3)],
                bounds: Some([0.0; 6]),
            },
            None,
        );
        assert!(
            past_end
                .transfer_layout()
                .expect_err("offsets beyond the line payload must be rejected")
                .contains("exceeds the line count")
        );
    }

    #[test]
    fn transfer_size_overflow_and_extra_chunks_are_rejected() {
        let overflowing = result(
            usize::MAX,
            WorkerScene::ThreeD {
                polygons: Vec::new(),
                bounds: Some([0.0; 6]),
            },
            None,
        );
        assert!(
            overflowing
                .transfer_layout()
                .expect_err("line transfer multiplication must overflow")
                .contains("transfer size overflow")
        );
        assert!(
            expected_chunk_values(
                0,
                0,
                LINE_TRANSFER_CHUNK_LINES,
                LINE_3D_TRANSFER_VALUES,
                "line",
            )
            .is_err()
        );
    }
}
