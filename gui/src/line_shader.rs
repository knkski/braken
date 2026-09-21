//! Device-limit-aware, index-free rendering for large turtle scenes.
//!
//! Iced's `Canvas` renderer combines all tessellated paths in a layer into one
//! index buffer. Large L-systems can therefore cross the device buffer limit
//! even though each individual line is tiny. This renderer expands a compact
//! line instance into two triangles in the vertex shader and never allocates an
//! index buffer. When a scene contains more detail than the viewport can show,
//! it picks a deterministic, resolution-derived preview while retaining the
//! complete scene on the CPU.
//!
//! WGPU is a display backend in this module: it neither derives an L-system nor
//! creates its visualization geometry. Live camera changes only update the
//! composite transform. A settled camera and raster epoch identify the exact
//! raster being refined; pointer interaction pauses that refinement, settling
//! resumes it, and an explicit cancellation remains sticky across later views.

use super::{CANVAS_BACKGROUND, DARK_CANVAS_BACKGROUND, Primitive2d, RenderBounds, Scene2d};
use crate::camera::{Camera2d, ViewBounds, ViewTransform, ViewportSize, WorldPoint};
use braken_gui::orientation::{OrientationAnchor, OrientationLandmarks, OrientationTransform};
use braken_gui::theme_palette::{theme_default_bucket, theme_default_color};
use braken_viz::{
    Line2d, MorphCoordinateSpace, MorphLine2d, StrokeColor, StyledLine2d,
    targets::{
        Palette, StrokeWidthEstimator, adaptive_scene_stroke_width,
        adaptive_scene_svg_stroke_width, turtle_stroke_rgb,
    },
};
use bytemuck::{Pod, Zeroable};
use iced::wgpu;
use iced::widget::shader;
use iced::{Color, Rectangle, Size};
use std::fmt::Write as _;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::{error::Error, fmt};

const INSTANCE_BYTES_PER_LOGICAL_PIXEL: f64 = 80.0;
const MIN_PREVIEW_SEGMENTS: usize = 4_096;
#[cfg(not(target_arch = "wasm32"))]
const UPLOAD_SEGMENTS_PER_FRAME: usize = 16_384;
// Transferred browser scenes perform five JS/WASM typed-array reads per
// segment. Bound each refinement slice tightly enough to preserve input frames.
#[cfg(target_arch = "wasm32")]
const UPLOAD_SEGMENTS_PER_FRAME: usize = 2_048;
#[cfg(target_arch = "wasm32")]
const TRANSFER_LINES_PER_CHUNK: usize = super::worker_protocol::LINE_TRANSFER_CHUNK_LINES;
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Each of the preview/exact accumulation textures stays within this scratch
/// budget. Very large or HiDPI windows are rendered at a proportionally lower
/// internal resolution and composited at full size; no scene lines are lost.
const MAX_ACCUMULATION_TEXTURE_BYTES: u64 = 64 * 1024 * 1024;

static NEXT_SCENE_ID: AtomicU64 = AtomicU64::new(1);

fn valid_line_width_scale(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

#[derive(Debug, Clone, Copy)]
struct SourceLine {
    start: [f32; 2],
    end: [f32; 2],
    width: f32,
    color: StrokeColor,
}

/// Immutable geometry for one adjacent-generation deformation.
///
/// Both fitted projections are retained so animation can interpolate in screen
/// space and arrive at the ordinary fitted source and target views exactly.
#[derive(Debug)]
pub(super) struct TransitionScene {
    id: u64,
    lines: MorphStorage,
    source_bounds: RenderBounds,
    target_bounds: RenderBounds,
    source_line_count: usize,
    target_line_count: usize,
    source_total_line_length: Option<f64>,
    target_total_line_length: Option<f64>,
    background: Option<[u8; 3]>,
}

#[derive(Debug)]
enum MorphStorage {
    Owned(Box<[MorphLine2d]>),
    #[cfg(target_arch = "wasm32")]
    Transferred {
        chunks: Box<[js_sys::Float32Array]>,
        line_count: usize,
    },
}

impl MorphStorage {
    fn len(&self) -> usize {
        match self {
            Self::Owned(lines) => lines.len(),
            #[cfg(target_arch = "wasm32")]
            Self::Transferred { line_count, .. } => *line_count,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> MorphLine2d {
        match self {
            Self::Owned(lines) => lines[index],
            #[cfg(target_arch = "wasm32")]
            Self::Transferred { chunks, .. } => {
                const VALUES: usize = super::worker_protocol::MORPH_TRANSFER_VALUES;
                let chunk_index = index / super::worker_protocol::MORPH_TRANSFER_CHUNK_LINES;
                let line_in_chunk = index % super::worker_protocol::MORPH_TRANSFER_CHUNK_LINES;
                let values = &chunks[chunk_index];
                let offset =
                    u32::try_from(line_in_chunk.saturating_mul(VALUES)).unwrap_or(u32::MAX);
                let value = |index: usize| {
                    values
                        .get_index(offset.saturating_add(u32::try_from(index).unwrap_or(u32::MAX)))
                };
                MorphLine2d {
                    source: StyledLine2d {
                        line: Line2d(
                            (f64::from(value(0)), f64::from(value(1))),
                            (f64::from(value(2)), f64::from(value(3))),
                        ),
                        width: f64::from(value(4)),
                        color: super::worker_protocol::decode_stroke_color(value(20)),
                    },
                    target: StyledLine2d {
                        line: Line2d(
                            (f64::from(value(5)), f64::from(value(6))),
                            (f64::from(value(7)), f64::from(value(8))),
                        ),
                        width: f64::from(value(9)),
                        color: super::worker_protocol::decode_stroke_color(value(21)),
                    },
                    source_space: super::worker_protocol::decode_morph_space(value(22)),
                    target_space: super::worker_protocol::decode_morph_space(value(23)),
                    source_opacity: value(10),
                    target_opacity: value(11),
                    source_palette_line: Line2d(
                        (f64::from(value(12)), f64::from(value(13))),
                        (f64::from(value(14)), f64::from(value(15))),
                    ),
                    target_palette_line: Line2d(
                        (f64::from(value(16)), f64::from(value(17))),
                        (f64::from(value(18)), f64::from(value(19))),
                    ),
                }
            }
        }
    }
}

impl TransitionScene {
    pub(super) fn new(
        lines: Vec<MorphLine2d>,
        source_bounds: RenderBounds,
        target_bounds: RenderBounds,
        source_metrics: (usize, Option<f64>),
        target_metrics: (usize, Option<f64>),
        background: Option<[u8; 3]>,
    ) -> Arc<Self> {
        Arc::new(Self {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: MorphStorage::Owned(lines.into_boxed_slice()),
            source_bounds,
            target_bounds,
            source_line_count: source_metrics.0,
            target_line_count: target_metrics.0,
            source_total_line_length: source_metrics.1,
            target_total_line_length: target_metrics.1,
            background,
        })
    }

    #[cfg(test)]
    pub(super) fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub(super) fn background(&self) -> Option<[u8; 3]> {
        self.background
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn from_transferred(
        values: js_sys::Array,
        line_count: usize,
        source_bounds: RenderBounds,
        target_bounds: RenderBounds,
        source_metrics: (usize, Option<f64>),
        target_metrics: (usize, Option<f64>),
        background: Option<[u8; 3]>,
    ) -> Result<Arc<Self>, LineSceneError> {
        use wasm_bindgen::JsCast;

        let expected_values = line_count
            .checked_mul(super::worker_protocol::MORPH_TRANSFER_VALUES)
            .ok_or(LineSceneError::InvalidTransfer {
                expected_values: usize::MAX,
                actual_values: 0,
            })?;
        let expected_chunks =
            line_count.div_ceil(super::worker_protocol::MORPH_TRANSFER_CHUNK_LINES);
        if values.length() as usize != expected_chunks {
            return Err(LineSceneError::InvalidTransfer {
                expected_values,
                actual_values: 0,
            });
        }
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(expected_chunks).map_err(|_| {
            LineSceneError::ResourceExhausted {
                requested_lines: line_count,
            }
        })?;
        let mut actual_values = 0usize;
        for (chunk_index, value) in values.iter().enumerate() {
            let chunk = value.dyn_into::<js_sys::Float32Array>().map_err(|_| {
                LineSceneError::InvalidTransfer {
                    expected_values,
                    actual_values,
                }
            })?;
            let remaining = line_count.saturating_sub(
                chunk_index.saturating_mul(super::worker_protocol::MORPH_TRANSFER_CHUNK_LINES),
            );
            let expected = remaining
                .min(super::worker_protocol::MORPH_TRANSFER_CHUNK_LINES)
                .saturating_mul(super::worker_protocol::MORPH_TRANSFER_VALUES);
            if chunk.length() as usize != expected {
                return Err(LineSceneError::InvalidTransfer {
                    expected_values,
                    actual_values: actual_values.saturating_add(chunk.length() as usize),
                });
            }
            actual_values = actual_values.saturating_add(chunk.length() as usize);
            chunks.push(chunk);
        }
        if actual_values != expected_values {
            return Err(LineSceneError::InvalidTransfer {
                expected_values,
                actual_values,
            });
        }
        Ok(Arc::new(Self {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: MorphStorage::Transferred {
                chunks: chunks.into_boxed_slice(),
                line_count,
            },
            source_bounds,
            target_bounds,
            source_line_count: source_metrics.0,
            target_line_count: target_metrics.0,
            source_total_line_length: source_metrics.1,
            target_total_line_length: target_metrics.1,
            background,
        }))
    }

    pub(super) fn software_preview(
        &self,
        maximum_lines: usize,
        size: Size,
        camera: Camera2d,
        dark: bool,
        progress: f32,
        line_width_scale: f32,
    ) -> Vec<ProjectedMorphLine> {
        projected_transition_preview(
            self,
            size,
            camera,
            dark,
            progress,
            self.lines.len().min(maximum_lines),
            line_width_scale,
        )
        .into_iter()
        .map(|line| ProjectedMorphLine {
            start: line.start,
            end: line.end,
            width: line.width,
            color: line.color,
        })
        .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ProjectedMorphLine {
    pub(super) start: [f32; 2],
    pub(super) end: [f32; 2],
    pub(super) width: f32,
    pub(super) color: [f32; 4],
}

/// A compact, immutable copy of the line data rendered by [`LineProgram`].
///
/// Text and other primitives intentionally remain on the regular Canvas path.
#[derive(Debug)]
pub(super) struct LineScene {
    id: u64,
    lines: LineStorage,
    bounds: RenderBounds,
    total_line_length: Option<f64>,
    background: Option<[u8; 3]>,
    refined_lines: AtomicUsize,
    refinement_cancelled: AtomicBool,
}

#[derive(Debug)]
enum LineStorage {
    Owned(Box<[SourceLine]>),
    #[cfg(target_arch = "wasm32")]
    Transferred {
        chunks: Box<[js_sys::Float32Array]>,
        line_count: usize,
    },
}

impl LineStorage {
    fn len(&self) -> usize {
        match self {
            Self::Owned(lines) => lines.len(),
            #[cfg(target_arch = "wasm32")]
            Self::Transferred { line_count, .. } => *line_count,
        }
    }

    fn get(&self, index: usize) -> SourceLine {
        match self {
            Self::Owned(lines) => lines[index],
            #[cfg(target_arch = "wasm32")]
            Self::Transferred { chunks, .. } => {
                let chunk_index = index / TRANSFER_LINES_PER_CHUNK;
                let line_in_chunk = index % TRANSFER_LINES_PER_CHUNK;
                let values = &chunks[chunk_index];
                let offset = u32::try_from(line_in_chunk.saturating_mul(6)).unwrap_or(u32::MAX);
                SourceLine {
                    start: [
                        values.get_index(offset),
                        values.get_index(offset.saturating_add(1)),
                    ],
                    end: [
                        values.get_index(offset.saturating_add(2)),
                        values.get_index(offset.saturating_add(3)),
                    ],
                    width: values.get_index(offset.saturating_add(4)),
                    color: super::worker_protocol::decode_stroke_color(
                        values.get_index(offset.saturating_add(5)),
                    ),
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn copy_range_values(&self, range: std::ops::Range<usize>, output: &mut Vec<f32>) -> bool {
        let value_count = match range.len().checked_mul(6) {
            Some(count) => count,
            None => return false,
        };
        output.clear();
        if output.try_reserve(value_count).is_err() {
            return false;
        }

        match self {
            Self::Owned(lines) => {
                for line in &lines[range] {
                    output.extend_from_slice(&[
                        line.start[0],
                        line.start[1],
                        line.end[0],
                        line.end[1],
                        line.width,
                        super::worker_protocol::encode_stroke_color(line.color),
                    ]);
                }
            }
            Self::Transferred { chunks, .. } => {
                output.resize(value_count, 0.0);
                let mut source_line = range.start;
                let mut destination = 0usize;
                while source_line < range.end {
                    let chunk_index = source_line / TRANSFER_LINES_PER_CHUNK;
                    let local_start = source_line % TRANSFER_LINES_PER_CHUNK;
                    let lines_from_chunk =
                        (range.end - source_line).min(TRANSFER_LINES_PER_CHUNK - local_start);
                    let value_start = u32::try_from(local_start * 6).unwrap_or(u32::MAX);
                    let value_end =
                        u32::try_from((local_start + lines_from_chunk) * 6).unwrap_or(u32::MAX);
                    let value_len = lines_from_chunk * 6;
                    chunks[chunk_index]
                        .subarray(value_start, value_end)
                        .copy_to(&mut output[destination..destination + value_len]);
                    source_line += lines_from_chunk;
                    destination += value_len;
                }
            }
        }
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LineSceneError {
    ResourceExhausted {
        requested_lines: usize,
    },
    Cancelled,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    InvalidTransfer {
        expected_values: usize,
        actual_values: usize,
    },
}

#[derive(Debug, Default)]
pub(super) struct LineSceneBuilder {
    lines: Vec<SourceLine>,
    width_estimator: StrokeWidthEstimator,
    background: Option<[u8; 3]>,
    orientation_landmarks: OrientationLandmarks,
}

impl LineSceneBuilder {
    pub(super) fn set_background(&mut self, background: Option<[u8; 3]>) {
        self.background = background;
    }

    pub(super) fn extend(
        &mut self,
        lines: impl IntoIterator<Item = StyledLine2d>,
    ) -> Result<(), LineSceneError> {
        let lines = lines.into_iter();
        let (lower, upper) = lines.size_hint();
        let additional = upper.unwrap_or(lower);
        self.lines
            .try_reserve(additional)
            .map_err(|_| LineSceneError::ResourceExhausted {
                requested_lines: self.lines.len().saturating_add(additional),
            })?;
        for line in lines {
            if self.lines.len() == self.lines.capacity() {
                self.lines
                    .try_reserve(1)
                    .map_err(|_| LineSceneError::ResourceExhausted {
                        requested_lines: self.lines.len().saturating_add(1),
                    })?;
            }
            self.width_estimator.observe(line.line);
            self.orientation_landmarks.observe(line.line.0, line.line.1);
            self.lines.push(SourceLine {
                start: [line.line.0.0 as f32, line.line.0.1 as f32],
                end: [line.line.1.0 as f32, line.line.1.1 as f32],
                width: line.width as f32,
                color: line.color,
            });
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn finish(self, bounds: RenderBounds) -> Arc<LineScene> {
        self.finish_oriented(bounds, None, 0.0).0
    }

    pub(super) fn finish_oriented(
        mut self,
        bounds: RenderBounds,
        anchor: Option<OrientationAnchor>,
        reference_angle: f64,
    ) -> (Arc<LineScene>, RenderBounds, OrientationTransform) {
        let transform = self
            .orientation_landmarks
            .transform(anchor, reference_angle);
        if !transform.is_identity() {
            for line in &mut self.lines {
                let start = transform.apply((f64::from(line.start[0]), f64::from(line.start[1])));
                let end = transform.apply((f64::from(line.end[0]), f64::from(line.end[1])));
                line.start = [start.0 as f32, start.1 as f32];
                line.end = [end.0 as f32, end.1 as f32];
            }
        }
        let oriented_bounds = if transform.is_identity() {
            bounds
        } else {
            source_line_bounds(&self.lines).unwrap_or_default()
        };
        let scene = Arc::new(LineScene {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: LineStorage::Owned(self.lines.into_boxed_slice()),
            bounds: oriented_bounds,
            total_line_length: self.width_estimator.total_line_length(),
            background: self.background,
            refined_lines: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
        });
        (scene, oriented_bounds, transform)
    }
}

fn source_line_bounds(lines: &[SourceLine]) -> Option<RenderBounds> {
    lines.iter().fold(None::<RenderBounds>, |bounds, line| {
        Some(match bounds {
            Some(bounds) => RenderBounds {
                min_x: bounds.min_x.min(line.start[0]).min(line.end[0]),
                max_x: bounds.max_x.max(line.start[0]).max(line.end[0]),
                min_y: bounds.min_y.min(line.start[1]).min(line.end[1]),
                max_y: bounds.max_y.max(line.start[1]).max(line.end[1]),
            },
            None => RenderBounds {
                min_x: line.start[0].min(line.end[0]),
                max_x: line.start[0].max(line.end[0]),
                min_y: line.start[1].min(line.end[1]),
                max_y: line.start[1].max(line.end[1]),
            },
        })
    })
}

impl fmt::Display for LineSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceExhausted { requested_lines } => write!(
                formatter,
                "not enough memory to prepare {requested_lines} lines for the GPU renderer"
            ),
            Self::Cancelled => formatter.write_str("line-scene operation cancelled"),
            Self::InvalidTransfer {
                expected_values,
                actual_values,
            } => write!(
                formatter,
                "render worker transferred {actual_values} line values; expected {expected_values}"
            ),
        }
    }
}

impl Error for LineSceneError {}

impl LineScene {
    pub(super) fn from_scene(
        scene: &Scene2d,
        bounds: RenderBounds,
    ) -> Result<Option<Arc<Self>>, LineSceneError> {
        if scene.primitives.is_empty()
            || scene
                .primitives
                .iter()
                .any(|primitive| !matches!(primitive, Primitive2d::Line(_)))
        {
            return Ok(None);
        }

        let mut lines = Vec::new();
        let mut width_estimator = StrokeWidthEstimator::default();
        lines
            .try_reserve_exact(scene.primitives.len())
            .map_err(|_| LineSceneError::ResourceExhausted {
                requested_lines: scene.primitives.len(),
            })?;
        for primitive in &scene.primitives {
            let Primitive2d::Line(line) = primitive else {
                unreachable!("non-line primitives were rejected above")
            };
            width_estimator.observe(line.line);
            lines.push(SourceLine {
                start: [line.line.0.0 as f32, line.line.0.1 as f32],
                end: [line.line.1.0 as f32, line.line.1.1 as f32],
                width: line.width as f32,
                color: line.color,
            });
        }

        Ok(Some(Arc::new(Self {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: LineStorage::Owned(lines.into_boxed_slice()),
            bounds,
            total_line_length: width_estimator.total_line_length(),
            background: scene.background,
            refined_lines: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
        })))
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn from_transferred(
        values: js_sys::Array,
        line_count: usize,
        bounds: RenderBounds,
        total_line_length: Option<f64>,
        background: Option<[u8; 3]>,
    ) -> Result<Arc<Self>, LineSceneError> {
        use wasm_bindgen::JsCast;

        let expected_values = line_count
            .checked_mul(6)
            .ok_or(LineSceneError::InvalidTransfer {
                expected_values: usize::MAX,
                actual_values: 0,
            })?;
        let chunk_count = values.length() as usize;
        let expected_chunks = line_count.div_ceil(TRANSFER_LINES_PER_CHUNK);
        if chunk_count != expected_chunks {
            return Err(LineSceneError::InvalidTransfer {
                expected_values,
                actual_values: 0,
            });
        }
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(chunk_count)
            .map_err(|_| LineSceneError::ResourceExhausted {
                requested_lines: line_count,
            })?;
        let mut actual_values = 0usize;
        for (chunk_index, value) in values.iter().enumerate() {
            let chunk = value.dyn_into::<js_sys::Float32Array>().map_err(|_| {
                LineSceneError::InvalidTransfer {
                    expected_values,
                    actual_values,
                }
            })?;
            let remaining_lines =
                line_count.saturating_sub(chunk_index.saturating_mul(TRANSFER_LINES_PER_CHUNK));
            let expected_chunk_values = remaining_lines
                .min(TRANSFER_LINES_PER_CHUNK)
                .saturating_mul(6);
            if chunk.length() as usize != expected_chunk_values {
                return Err(LineSceneError::InvalidTransfer {
                    expected_values,
                    actual_values: actual_values.saturating_add(chunk.length() as usize),
                });
            }
            actual_values = actual_values.saturating_add(chunk.length() as usize);
            chunks.push(chunk);
        }
        if actual_values != expected_values {
            return Err(LineSceneError::InvalidTransfer {
                expected_values,
                actual_values,
            });
        }
        Ok(Arc::new(Self {
            id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: LineStorage::Transferred {
                chunks: chunks.into_boxed_slice(),
                line_count,
            },
            bounds,
            total_line_length,
            background,
            refined_lines: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
        }))
    }

    pub(super) fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub(super) fn total_line_length(&self) -> Option<f64> {
        self.total_line_length
    }

    pub(super) fn fallback_preview(&self, maximum_lines: usize) -> Result<Scene2d, LineSceneError> {
        let selected = self.lines.len().min(maximum_lines);
        let mut primitives = Vec::new();
        primitives
            .try_reserve_exact(selected)
            .map_err(|_| LineSceneError::ResourceExhausted {
                requested_lines: selected,
            })?;
        for output_index in 0..selected {
            let source_index = if selected == self.lines.len() {
                output_index
            } else if selected <= 1 {
                self.lines.len() / 2
            } else {
                ((output_index as u128 * (self.lines.len() - 1) as u128) / (selected - 1) as u128)
                    as usize
            };
            let line = self.lines.get(source_index);
            primitives.push(Primitive2d::Line(StyledLine2d {
                line: Line2d(
                    (f64::from(line.start[0]), f64::from(line.start[1])),
                    (f64::from(line.end[0]), f64::from(line.end[1])),
                ),
                width: f64::from(line.width),
                color: line.color,
            }));
        }
        Ok(Scene2d {
            primitives,
            background: self.background,
        })
    }

    pub(super) fn refinement_progress(&self) -> (usize, usize) {
        (
            self.refined_lines
                .load(Ordering::Acquire)
                .min(self.lines.len()),
            self.lines.len(),
        )
    }

    pub(super) fn is_refining(&self) -> bool {
        let (completed, total) = self.refinement_progress();
        !self.refinement_cancelled.load(Ordering::Acquire) && completed < total
    }

    pub(super) fn refinement_was_cancelled(&self) -> bool {
        self.refinement_cancelled.load(Ordering::Acquire)
    }

    pub(super) fn cancel_refinement(&self) {
        self.refinement_cancelled.store(true, Ordering::Release);
    }

    /// Marks a newly settled camera as needing a fresh exact display pass.
    /// A user's explicit cancellation is sticky across navigation.
    pub(super) fn restart_view_refinement(&self) {
        if !self.refinement_cancelled.load(Ordering::Acquire) {
            self.refined_lines.store(0, Ordering::Release);
        }
    }

    /// Records completion of the bounded Canvas preview when Iced has selected
    /// its software renderer. Custom shader primitives are unavailable on that
    /// path, so there is no exact GPU refinement to keep scheduling.
    pub(super) fn mark_software_preview_complete(&self) {
        self.refined_lines
            .store(self.lines.len(), Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn set_refinement_progress_for_test(&self, completed: usize) {
        self.refined_lines.store(completed, Ordering::Release);
    }

    pub(super) fn encode_svg_with_cancel(
        &self,
        palette: Palette,
        line_width_scale: f32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String, LineSceneError> {
        let (mut output, foreground, stroke_width) = self.svg_header(palette, line_width_scale)?;
        for start in (0..self.lines.len()).step_by(SVG_LINES_PER_BATCH) {
            let end = start
                .saturating_add(SVG_LINES_PER_BATCH)
                .min(self.lines.len());
            self.append_svg_lines(
                &mut output,
                foreground,
                palette,
                stroke_width,
                start..end,
                is_cancelled,
            )?;
        }
        if is_cancelled() {
            return Err(LineSceneError::Cancelled);
        }
        output.push_str("</svg>\n");
        Ok(output)
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) async fn encode_svg_yielding(
        &self,
        palette: Palette,
        line_width_scale: f32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String, LineSceneError> {
        let (mut output, foreground, stroke_width) = self.svg_header(palette, line_width_scale)?;
        for start in (0..self.lines.len()).step_by(SVG_LINES_PER_BATCH) {
            let end = start
                .saturating_add(SVG_LINES_PER_BATCH)
                .min(self.lines.len());
            self.append_svg_lines(
                &mut output,
                foreground,
                palette,
                stroke_width,
                start..end,
                is_cancelled,
            )?;
            yield_to_browser().await;
        }
        if is_cancelled() {
            return Err(LineSceneError::Cancelled);
        }
        output.push_str("</svg>\n");
        Ok(output)
    }

    fn svg_header(
        &self,
        palette: Palette,
        line_width_scale: f32,
    ) -> Result<(String, &'static str, f32), LineSceneError> {
        let (themed_background, foreground) = match palette {
            Palette::Light => ("#f2f5f9", "#263247"),
            Palette::Dark => ("#0f1420", "#eef2f8"),
        };
        let background = self
            .background
            .map(|[red, green, blue]| format!("#{red:02x}{green:02x}{blue:02x}"))
            .unwrap_or_else(|| themed_background.to_owned());
        let drawing_extent = render_bounds_extent(self.bounds);
        let drawing_width = drawing_extent.0.max(0.1) as f32;
        let drawing_height = drawing_extent.1.max(0.1) as f32;
        let margin = drawing_width.max(drawing_height) * 0.04;
        let view_x = self.bounds.min_x - margin;
        let view_y = -self.bounds.max_y - margin;
        let view_width = drawing_width + margin * 2.0;
        let view_height = drawing_height + margin * 2.0;
        let stroke_width = adaptive_scene_svg_stroke_width(
            self.total_line_length,
            self.lines.len(),
            drawing_extent,
            f64::from(view_width.max(view_height)),
        ) as f32
            * valid_line_width_scale(line_width_scale);
        let mut output = String::new();
        output
            .try_reserve(512)
            .map_err(|_| LineSceneError::ResourceExhausted {
                requested_lines: self.lines.len(),
            })?;
        let _ = writeln!(
            output,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{view_x} {view_y} {view_width} {view_height}\"><rect x=\"{view_x}\" y=\"{view_y}\" width=\"{view_width}\" height=\"{view_height}\" fill=\"{background}\"/>"
        );
        Ok((output, foreground, stroke_width))
    }

    fn append_svg_lines(
        &self,
        output: &mut String,
        foreground: &str,
        palette: Palette,
        stroke_width: f32,
        range: std::ops::Range<usize>,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<(), LineSceneError> {
        if is_cancelled() {
            return Err(LineSceneError::Cancelled);
        }
        let requested = range
            .len()
            .checked_mul(320)
            .ok_or(LineSceneError::ResourceExhausted {
                requested_lines: self.lines.len(),
            })?;
        output
            .try_reserve(requested)
            .map_err(|_| LineSceneError::ResourceExhausted {
                requested_lines: self.lines.len(),
            })?;
        for index in range {
            let line = self.lines.get(index);
            let width = stroke_width * line.width;
            let stroke = turtle_stroke_rgb(line.color, palette)
                .map(|[red, green, blue]| format!("#{red:02x}{green:02x}{blue:02x}"))
                .unwrap_or_else(|| foreground.to_owned());
            let _ = writeln!(
                output,
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>",
                line.start[0], -line.start[1], line.end[0], -line.end[1]
            );
        }
        Ok(())
    }

    /// Number of segments selected before applying the adapter's stricter
    /// buffer limit. The returned value is useful for honest preview status.
    pub(super) fn viewport_preview_count(&self, size: Size) -> usize {
        self.lines.len().min(preview_budget_for_viewport(size))
    }
}

const SVG_LINES_PER_BATCH: usize = 2 * 1024;

#[cfg(target_arch = "wasm32")]
async fn yield_to_browser() {
    use wasm_bindgen::JsValue;

    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if web_sys::window().is_some_and(|window| {
            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
                .is_ok()
        }) {
            return;
        }
        let _result = resolve.call0(&JsValue::UNDEFINED);
    });
    let _result = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[derive(Debug, Clone)]
pub(super) struct LineProgram {
    scene: Arc<LineScene>,
    dark: bool,
    // Live input is composited immediately over the latest settled raster.
    live_camera: Camera2d,
    // Exact line batches are projected for this camera only.
    settled_camera: Camera2d,
    // Interaction pauses exact batches without invalidating the front texture.
    interacting: bool,
    // Display-only invalidation; derivation and visualization own other keys.
    raster_epoch: u64,
    line_width_scale: f32,
}

impl LineProgram {
    pub(super) fn new(
        scene: Arc<LineScene>,
        dark: bool,
        live_camera: Camera2d,
        settled_camera: Camera2d,
        interacting: bool,
        raster_epoch: u64,
        line_width_scale: f32,
    ) -> Self {
        Self {
            scene,
            dark,
            live_camera,
            settled_camera,
            interacting,
            raster_epoch,
            line_width_scale: valid_line_width_scale(line_width_scale),
        }
    }
}

impl<Message> shader::Program<Message> for LineProgram {
    type State = ();
    type Primitive = LinePrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: iced::mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        LinePrimitive {
            scene: Arc::clone(&self.scene),
            dark: self.dark,
            size: bounds.size(),
            live_camera: self.live_camera,
            settled_camera: self.settled_camera,
            interacting: self.interacting,
            raster_epoch: self.raster_epoch,
            line_width_scale: self.line_width_scale,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct LinePrimitive {
    scene: Arc<LineScene>,
    dark: bool,
    size: Size,
    live_camera: Camera2d,
    settled_camera: Camera2d,
    interacting: bool,
    raster_epoch: u64,
    line_width_scale: f32,
}

/// Bounded preview renderer for a deformation transition. Unlike the exact
/// static renderer, this path intentionally samples to viewport/device limits:
/// its lifetime is only the short animation, after which the exact target
/// [`LineScene`] resumes ordinary refinement.
#[derive(Debug, Clone)]
pub(super) struct TransitionProgram {
    scene: Arc<TransitionScene>,
    progress: f32,
    dark: bool,
    camera: Camera2d,
    line_width_scale: f32,
}

impl TransitionProgram {
    pub(super) fn new(
        scene: Arc<TransitionScene>,
        progress: f32,
        dark: bool,
        camera: Camera2d,
        line_width_scale: f32,
    ) -> Self {
        Self {
            scene,
            progress: progress.clamp(0.0, 1.0),
            dark,
            camera,
            line_width_scale: valid_line_width_scale(line_width_scale),
        }
    }
}

impl<Message> shader::Program<Message> for TransitionProgram {
    type State = ();
    type Primitive = TransitionPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: iced::mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        TransitionPrimitive {
            scene: Arc::clone(&self.scene),
            progress: self.progress,
            dark: self.dark,
            camera: self.camera,
            size: bounds.size(),
            line_width_scale: self.line_width_scale,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TransitionPrimitive {
    scene: Arc<TransitionScene>,
    progress: f32,
    dark: bool,
    camera: Camera2d,
    size: Size,
    line_width_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransitionPreparedKey {
    scene_id: u64,
    progress: u32,
    width: u32,
    height: u32,
    dark: bool,
    camera: CameraKey,
    line_width_scale: u32,
}

pub(super) struct TransitionPipeline {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instance_count: u32,
    prepared: Option<TransitionPreparedKey>,
    target: PhysicalRect,
}

impl shader::Pipeline for TransitionPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("braken.transition.shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "line_shader.wgsl"
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("braken.transition.pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("braken.transition.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuLine>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Float32,
                        3 => Float32,
                        4 => Float32x4,
                        5 => Float32x2,
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            instance_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("braken.transition.instances"),
                size: wgpu::COPY_BUFFER_ALIGNMENT,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            instance_capacity: 0,
            instance_count: 0,
            prepared: None,
            target: PhysicalRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        }
    }
}

impl shader::Primitive for TransitionPrimitive {
    type Pipeline = TransitionPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        pipeline.target = physical_target(bounds, viewport.scale_factor());
        let key = TransitionPreparedKey {
            scene_id: self.scene.id,
            progress: self.progress.to_bits(),
            width: bounds.width.to_bits(),
            height: bounds.height.to_bits(),
            dark: self.dark,
            camera: self.camera.into(),
            line_width_scale: self.line_width_scale.to_bits(),
        };
        if pipeline.prepared == Some(key) {
            return;
        }
        let device_capacity = (device.limits().max_buffer_size
            / std::mem::size_of::<GpuLine>() as u64)
            .min(u32::MAX as u64) as usize;
        let selected = self
            .scene
            .lines
            .len()
            .min(preview_budget_for_viewport(self.size))
            .min(device_capacity.saturating_sub(1));
        let required = selected.saturating_add(1).min(device_capacity);
        if required > pipeline.instance_capacity {
            let bytes = (required as u64)
                .saturating_mul(std::mem::size_of::<GpuLine>() as u64)
                .max(wgpu::COPY_BUFFER_ALIGNMENT)
                .min(device.limits().max_buffer_size);
            pipeline.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("braken.transition.instances"),
                size: bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            pipeline.instance_capacity = (bytes / std::mem::size_of::<GpuLine>() as u64) as usize;
        }
        let projected = projected_transition_preview(
            &self.scene,
            self.size,
            self.camera,
            self.dark,
            self.progress,
            selected.min(pipeline.instance_capacity),
            self.line_width_scale,
        );
        let mut lines = Vec::new();
        if lines
            .try_reserve_exact(projected.len().saturating_add(1))
            .is_ok()
        {
            let background = display_background(self.scene.background, self.dark);
            let height = self.size.height.max(1.0);
            lines.push(GpuLine {
                start: [-height, height * 0.5],
                end: [self.size.width.max(1.0) + height, height * 0.5],
                width: height * 2.0,
                _padding: 0.0,
                color: [background.r, background.g, background.b, 1.0],
                viewport: [self.size.width.max(1.0), height],
            });
            lines.extend(projected);
        }
        if !lines.is_empty() {
            queue.write_buffer(&pipeline.instance_buffer, 0, bytemuck::cast_slice(&lines));
        }
        pipeline.instance_count = u32::try_from(lines.len()).unwrap_or(u32::MAX);
        pipeline.prepared = Some(key);
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        if pipeline.instance_count == 0 || clip_bounds.width == 0 || clip_bounds.height == 0 {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("braken.transition.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_viewport(
            pipeline.target.x,
            pipeline.target.y,
            pipeline.target.width,
            pipeline.target.height,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_vertex_buffer(0, pipeline.instance_buffer.slice(..));
        pass.draw(0..6, 0..pipeline.instance_count);
    }
}

impl LinePrimitive {
    fn prepared_key(&self, bounds: &Rectangle, scale_factor: f32) -> PreparedKey {
        PreparedKey {
            scene_id: self.scene.id,
            raster_epoch: self.raster_epoch,
            width: bounds.width.to_bits(),
            height: bounds.height.to_bits(),
            scale: scale_factor.to_bits(),
            dark: self.dark,
            camera: self.settled_camera.into(),
            line_width_scale: self.line_width_scale.to_bits(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CameraKey {
    zoom: u64,
    focus_x: u64,
    focus_y: u64,
}

impl From<Camera2d> for CameraKey {
    fn from(camera: Camera2d) -> Self {
        Self {
            zoom: camera.zoom.to_bits(),
            focus_x: camera.focus[0].to_bits(),
            focus_y: camera.focus[1].to_bits(),
        }
    }
}

// Only inputs that require rerasterizing line geometry belong here. The live
// camera is a composite-pass uniform, interaction only controls scheduling,
// and absolute widget position is tracked by `Offscreen::target`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedKey {
    scene_id: u64,
    raster_epoch: u64,
    width: u32,
    height: u32,
    scale: u32,
    dark: bool,
    camera: CameraKey,
    line_width_scale: u32,
}

#[derive(Debug)]
pub(super) struct LinePipeline {
    line_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    composite_uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    prepared: Option<PreparedKey>,
    offscreen: Option<Offscreen>,
    pending_count: u32,
    pending_clear: bool,
    pending_target: AccumulationTarget,
    refinement: Refinement,
    #[cfg(target_arch = "wasm32")]
    transferred_scratch: Vec<f32>,
    #[cfg(target_arch = "wasm32")]
    transferred_instances: Vec<GpuLine>,
}

// `preview` is the sampled front texture. During exact refinement, `exact` is
// the same-sized back texture; completion swaps them so the old front can be
// reused as scratch for a later settled view. Incompatible extents are dropped
// before replacement allocation.
#[derive(Debug)]
struct Offscreen {
    preview: Option<AccumulationTexture>,
    exact: Option<AccumulationTexture>,
    width: u32,
    height: u32,
    target: PhysicalRect,
}

#[derive(Debug)]
struct AccumulationTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone, Copy)]
struct PhysicalRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

#[derive(Debug, Clone, Copy, Default)]
enum Refinement {
    #[default]
    Empty,
    PreviewPending {
        preview_count: usize,
    },
    ExactPending {
        end: usize,
    },
    Complete,
}

#[derive(Debug, Clone, Copy, Default)]
enum AccumulationTarget {
    #[default]
    Preview,
    Exact,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuLine {
    start: [f32; 2],
    end: [f32; 2],
    width: f32,
    _padding: f32,
    color: [f32; 4],
    viewport: [f32; 2],
}

/// Maps the live viewport back into the camera used to rasterize the currently
/// sampled accumulation texture. Keeping this transform in the composite pass
/// makes wheel, drag, and pinch updates constant-time even for enormous scenes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CompositeUniform {
    sample_scale: [f32; 2],
    sample_offset: [f32; 2],
    background: [f32; 4],
}

impl CompositeUniform {
    fn new(
        bounds: ViewBounds,
        size: ViewportSize,
        raster_camera: Camera2d,
        live_camera: Camera2d,
        dark: bool,
        source_background: Option<[u8; 3]>,
    ) -> Self {
        let raster = ViewTransform::new(bounds, size, raster_camera);
        let live = ViewTransform::new(bounds, size, live_camera);
        let mapping = live.screen_affine_to(&raster);
        let width = size.width.max(1.0);
        let height = size.height.max(1.0);
        let background = display_background(source_background, dark);
        Self {
            sample_scale: [mapping.scale as f32, mapping.scale as f32],
            sample_offset: [
                (mapping.translation.x / width) as f32,
                (mapping.translation.y / height) as f32,
            ],
            background: [background.r, background.g, background.b, background.a],
        }
    }
}

impl shader::Pipeline for LinePipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("braken.line.shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "line_shader.wgsl"
            ))),
        });
        let line_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("braken.line.pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("braken.line.pipeline"),
            layout: Some(&line_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuLine>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Float32,
                        3 => Float32,
                        4 => Float32x4,
                        5 => Float32x2,
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OFFSCREEN_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("braken.line.composite_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("braken.line.composite_pipeline_layout"),
                bind_group_layouts: &[&composite_layout],
                push_constant_ranges: &[],
            });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("braken.line.composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("braken.line.composite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("braken.line.instances"),
            size: wgpu::COPY_BUFFER_ALIGNMENT,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("braken.line.composite_uniform"),
            size: std::mem::size_of::<CompositeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            line_pipeline,
            composite_pipeline,
            composite_layout,
            composite_uniform_buffer,
            sampler,
            instance_buffer,
            instance_capacity: 0,
            prepared: None,
            offscreen: None,
            pending_count: 0,
            pending_clear: false,
            pending_target: AccumulationTarget::Preview,
            refinement: Refinement::Empty,
            #[cfg(target_arch = "wasm32")]
            transferred_scratch: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            transferred_instances: Vec::new(),
        }
    }
}

impl shader::Primitive for LinePrimitive {
    type Pipeline = LinePipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        let key = self.prepared_key(bounds, viewport.scale_factor());
        let view_bounds = view_bounds(self.scene.bounds);
        let viewport_size = viewport_size(self.size);
        let composite = CompositeUniform::new(
            view_bounds,
            viewport_size,
            self.settled_camera,
            self.live_camera,
            self.dark,
            self.scene.background,
        );
        queue.write_buffer(
            &pipeline.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&composite),
        );

        if let Some(offscreen) = pipeline.offscreen.as_mut() {
            update_offscreen_target(offscreen, bounds, viewport);
        }
        if pipeline.prepared != Some(key) {
            let capacity = upload_capacity(device.limits().max_buffer_size);
            if capacity > pipeline.instance_capacity {
                let allocation = (capacity as u64)
                    .saturating_mul(std::mem::size_of::<GpuLine>() as u64)
                    .max(wgpu::COPY_BUFFER_ALIGNMENT)
                    .min(device.limits().max_buffer_size);
                pipeline.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("braken.line.instances"),
                    size: allocation,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                pipeline.instance_capacity =
                    (allocation / std::mem::size_of::<GpuLine>() as u64) as usize;
            }

            prepare_offscreen(
                &mut pipeline.offscreen,
                device,
                &pipeline.composite_layout,
                &pipeline.sampler,
                &pipeline.composite_uniform_buffer,
                bounds,
                viewport,
            );
            pipeline.pending_count = 0;
            pipeline.pending_clear = true;
            pipeline.pending_target = AccumulationTarget::Preview;
            self.scene.refined_lines.store(0, Ordering::Release);
            let preview_count = self
                .scene
                .viewport_preview_count(self.size)
                .min(pipeline.instance_capacity);
            let instances = projected_preview_range(
                &self.scene,
                self.size,
                self.dark,
                self.settled_camera,
                preview_count,
                0..preview_count,
                self.line_width_scale,
            );
            write_instances(pipeline, queue, &instances);
            pipeline.refinement = Refinement::PreviewPending {
                preview_count: instances.len(),
            };
            pipeline.prepared = Some(key);
            return;
        }

        // The previous pending batch has been submitted by the preceding
        // frame. Keep the accumulated texture, but do not submit it twice.
        pipeline.pending_count = 0;
        pipeline.pending_clear = false;
        if let Refinement::ExactPending { end } = pipeline.refinement {
            self.scene.refined_lines.store(end, Ordering::Release);
        }
        if let Refinement::PreviewPending { preview_count } = pipeline.refinement
            && preview_count >= self.scene.lines.len()
        {
            self.scene
                .refined_lines
                .store(self.scene.lines.len(), Ordering::Release);
            pipeline.refinement = Refinement::Complete;
            if let Some(offscreen) = pipeline.offscreen.as_mut() {
                // The preview already contains the complete scene. Do not
                // retain an obsolete exact texture from the previous view.
                offscreen.exact = None;
            }
            return;
        }
        if let Refinement::ExactPending { end } = pipeline.refinement
            && end >= self.scene.lines.len()
        {
            pipeline.refinement = Refinement::Complete;
            if let Some(offscreen) = pipeline.offscreen.as_mut() {
                // Promote the finished back texture and retain the old front
                // as same-sized scratch. Large scenes can then settle repeated
                // camera gestures without allocating another viewport texture.
                if offscreen.exact.is_some() {
                    std::mem::swap(&mut offscreen.preview, &mut offscreen.exact);
                }
            }
            return;
        }
        if self.scene.refinement_cancelled.load(Ordering::Acquire) {
            pipeline.refinement = Refinement::Complete;
            if let Some(offscreen) = pipeline.offscreen.as_mut() {
                offscreen.exact = None;
            }
            return;
        }
        if self.interacting {
            return;
        }
        match pipeline.refinement {
            Refinement::PreviewPending { .. } => {
                ensure_exact_texture(pipeline, device);
                schedule_exact_batch(pipeline, queue, self, 0);
            }
            Refinement::ExactPending { end } => {
                schedule_exact_batch(pipeline, queue, self, end);
            }
            Refinement::Empty | Refinement::Complete => {}
        }
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let Some(offscreen) = &pipeline.offscreen else {
            return;
        };

        if pipeline.pending_count > 0 || pipeline.pending_clear {
            let background = display_background(self.scene.background, self.dark);
            let accumulation = match pipeline.pending_target {
                AccumulationTarget::Preview => offscreen.preview.as_ref(),
                AccumulationTarget::Exact => offscreen.exact.as_ref(),
            };
            let Some(accumulation) = accumulation else {
                return;
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("braken.line.offscreen_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &accumulation.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if pipeline.pending_clear {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: f64::from(background.r),
                                g: f64::from(background.g),
                                b: f64::from(background.b),
                                a: f64::from(background.a),
                            })
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if pipeline.pending_count > 0 {
                pass.set_pipeline(&pipeline.line_pipeline);
                pass.set_vertex_buffer(0, pipeline.instance_buffer.slice(..));
                pass.draw(0..6, 0..pipeline.pending_count);
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("braken.line.composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_viewport(
            offscreen.target.x,
            offscreen.target.y,
            offscreen.target.width,
            offscreen.target.height,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pass.set_pipeline(&pipeline.composite_pipeline);
        let accumulation = offscreen.preview.as_ref();
        let Some(accumulation) = accumulation else {
            return;
        };
        pass.set_bind_group(0, &accumulation.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn upload_capacity(max_buffer_size: u64) -> usize {
    let device_capacity = max_buffer_size / std::mem::size_of::<GpuLine>() as u64;
    device_capacity.min(UPLOAD_SEGMENTS_PER_FRAME as u64) as usize
}

fn prepare_offscreen(
    offscreen: &mut Option<Offscreen>,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform_buffer: &wgpu::Buffer,
    bounds: &Rectangle,
    viewport: &shader::Viewport,
) {
    let scale = viewport.scale_factor();
    let limits = device.limits();
    let requested_width = ((bounds.width * scale).ceil() as u32).max(1);
    let requested_height = ((bounds.height * scale).ceil() as u32).max(1);
    let (width, height) = accumulation_extent(
        requested_width,
        requested_height,
        limits.max_texture_dimension_2d,
        MAX_ACCUMULATION_TEXTURE_BYTES / 4,
    );
    if width == 0 || height == 0 {
        *offscreen = None;
        return;
    }
    let target = physical_target(bounds, scale);
    if let Some(current) = offscreen
        && current.width == width
        && current.height == height
    {
        current.target = target;
        if current.preview.is_none() {
            current.preview = Some(create_accumulation_texture(
                device,
                layout,
                sampler,
                uniform_buffer,
                width,
                height,
                true,
            ));
        }
        return;
    }
    // Release incompatible viewport textures before allocating their
    // replacement. During a resize a refining scene may otherwise retain two
    // old accumulation textures while creating the new preview.
    *offscreen = None;
    let preview =
        create_accumulation_texture(device, layout, sampler, uniform_buffer, width, height, true);
    *offscreen = Some(Offscreen {
        preview: Some(preview),
        exact: None,
        width,
        height,
        target,
    });
}

fn physical_target(bounds: &Rectangle, scale: f32) -> PhysicalRect {
    PhysicalRect {
        x: bounds.x * scale,
        y: bounds.y * scale,
        width: bounds.width * scale,
        height: bounds.height * scale,
    }
}

fn update_offscreen_target(
    offscreen: &mut Offscreen,
    bounds: &Rectangle,
    viewport: &shader::Viewport,
) {
    offscreen.target = physical_target(bounds, viewport.scale_factor());
}

fn accumulation_extent(
    requested_width: u32,
    requested_height: u32,
    max_dimension: u32,
    max_pixels: u64,
) -> (u32, u32) {
    let requested_width = requested_width.max(1);
    let requested_height = requested_height.max(1);
    let max_dimension = max_dimension.max(1);
    let pixels = u64::from(requested_width).saturating_mul(u64::from(requested_height));
    let dimension_ratio = (f64::from(max_dimension) / f64::from(requested_width))
        .min(f64::from(max_dimension) / f64::from(requested_height));
    let pixel_ratio = (max_pixels.max(1) as f64 / pixels as f64).sqrt();
    let ratio = dimension_ratio.min(pixel_ratio).min(1.0);
    (
        ((f64::from(requested_width) * ratio).floor() as u32)
            .max(1)
            .min(max_dimension),
        ((f64::from(requested_height) * ratio).floor() as u32)
            .max(1)
            .min(max_dimension),
    )
}

fn create_accumulation_texture(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform_buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    preview: bool,
) -> AccumulationTexture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(if preview {
            "braken.line.preview_texture"
        } else {
            "braken.line.exact_texture"
        }),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(if preview {
            "braken.line.preview_bind_group"
        } else {
            "braken.line.exact_bind_group"
        }),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    AccumulationTexture {
        _texture: texture,
        view,
        bind_group,
    }
}

fn ensure_exact_texture(pipeline: &mut LinePipeline, device: &wgpu::Device) {
    let Some((width, height)) = pipeline
        .offscreen
        .as_ref()
        .filter(|offscreen| offscreen.exact.is_none())
        .map(|offscreen| (offscreen.width, offscreen.height))
    else {
        return;
    };
    let exact = create_accumulation_texture(
        device,
        &pipeline.composite_layout,
        &pipeline.sampler,
        &pipeline.composite_uniform_buffer,
        width,
        height,
        false,
    );
    if let Some(offscreen) = pipeline.offscreen.as_mut() {
        offscreen.exact = Some(exact);
    }
}

fn write_instances(pipeline: &mut LinePipeline, queue: &wgpu::Queue, instances: &[GpuLine]) {
    let count = instances.len().min(pipeline.instance_capacity);
    if count > 0 {
        queue.write_buffer(
            &pipeline.instance_buffer,
            0,
            bytemuck::cast_slice(&instances[..count]),
        );
    }
    pipeline.pending_count = u32::try_from(count).unwrap_or(u32::MAX);
}

fn schedule_exact_batch(
    pipeline: &mut LinePipeline,
    queue: &wgpu::Queue,
    primitive: &LinePrimitive,
    start: usize,
) {
    let scene = &primitive.scene;
    let end = start
        .saturating_add(pipeline.instance_capacity)
        .min(scene.lines.len());
    #[cfg(not(target_arch = "wasm32"))]
    let allocation_failed = {
        let instances = projected_exact_range(
            scene,
            primitive.size,
            primitive.dark,
            primitive.settled_camera,
            start..end,
            primitive.line_width_scale,
        );
        write_instances(pipeline, queue, &instances);
        instances.is_empty() && start < end
    };
    #[cfg(target_arch = "wasm32")]
    let allocation_failed = if projected_exact_range(
        primitive,
        start..end,
        &mut pipeline.transferred_scratch,
        &mut pipeline.transferred_instances,
    ) {
        let count = pipeline
            .transferred_instances
            .len()
            .min(pipeline.instance_capacity);
        if count > 0 {
            queue.write_buffer(
                &pipeline.instance_buffer,
                0,
                bytemuck::cast_slice(&pipeline.transferred_instances[..count]),
            );
        }
        pipeline.pending_count = u32::try_from(count).unwrap_or(u32::MAX);
        false
    } else {
        pipeline.pending_count = 0;
        start < end
    };
    pipeline.pending_clear = start == 0 && !allocation_failed;
    pipeline.pending_target = AccumulationTarget::Exact;
    pipeline.refinement = if allocation_failed {
        // A bounded host allocation failed. Keep the already visible preview
        // instead of risking an unbounded retry loop. Release the back texture
        // immediately: refinement is now complete, so another frame is not
        // guaranteed to run the ordinary cancellation cleanup path.
        scene.refinement_cancelled.store(true, Ordering::Release);
        if let Some(offscreen) = pipeline.offscreen.as_mut() {
            offscreen.exact = None;
        }
        Refinement::Complete
    } else {
        Refinement::ExactPending { end }
    };
}

fn preview_budget_for_viewport(size: Size) -> usize {
    let pixels = f64::from(size.width.max(1.0)) * f64::from(size.height.max(1.0));
    ((pixels * INSTANCE_BYTES_PER_LOGICAL_PIXEL / std::mem::size_of::<GpuLine>() as f64).ceil()
        as usize)
        .max(MIN_PREVIEW_SEGMENTS)
}

#[cfg(test)]
fn preview_selection_count(total: usize, size: Size, max_buffer_size: u64) -> usize {
    let device_capacity =
        (max_buffer_size / std::mem::size_of::<GpuLine>() as u64).min(u32::MAX as u64) as usize;
    total
        .min(preview_budget_for_viewport(size))
        .min(device_capacity)
}

#[cfg(not(target_arch = "wasm32"))]
fn projected_exact_range(
    scene: &LineScene,
    size: Size,
    dark: bool,
    camera: Camera2d,
    range: std::ops::Range<usize>,
    line_width_scale: f32,
) -> Vec<GpuLine> {
    projected_range(
        scene,
        size,
        dark,
        camera,
        range.clone(),
        line_width_scale,
        |output_index| output_index,
    )
}

#[cfg(target_arch = "wasm32")]
fn projected_exact_range(
    primitive: &LinePrimitive,
    range: std::ops::Range<usize>,
    scratch: &mut Vec<f32>,
    instances: &mut Vec<GpuLine>,
) -> bool {
    let scene = &primitive.scene;
    instances.clear();
    if range.is_empty()
        || scene.lines.len() == 0
        || !scene.lines.copy_range_values(range.clone(), scratch)
    {
        return range.is_empty() || scene.lines.len() == 0;
    }

    if instances.try_reserve_exact(range.len()).is_err() {
        return false;
    }
    let bounds = view_bounds(scene.bounds);
    let transform = ViewTransform::new(
        bounds,
        viewport_size(primitive.size),
        primitive.settled_camera,
    );
    let base_width = adaptive_scene_stroke_width(
        scene.total_line_length,
        scene.lines.len(),
        render_bounds_extent(scene.bounds),
        transform.fit_scale(),
    ) as f32
        * primitive.settled_camera.zoom as f32
        * primitive.line_width_scale;
    for values in scratch.chunks_exact(6) {
        let line = SourceLine {
            start: [values[0], values[1]],
            end: [values[2], values[3]],
            width: values[4],
            color: super::worker_protocol::decode_stroke_color(values[5]),
        };
        let start = project(line.start, &transform);
        let end = project(line.end, &transform);
        let rgb = resolve_line_color(line.color, line, &transform, primitive.dark);
        let alpha = if line.width > 0.0 { 1.0 } else { 0.0 };
        instances.push(GpuLine {
            start,
            end,
            width: base_width * line.width.max(0.0),
            _padding: 0.0,
            color: [rgb[0], rgb[1], rgb[2], alpha],
            viewport: [
                primitive.size.width.max(1.0),
                primitive.size.height.max(1.0),
            ],
        });
    }
    true
}

fn projected_preview_range(
    scene: &LineScene,
    size: Size,
    dark: bool,
    camera: Camera2d,
    selected: usize,
    range: std::ops::Range<usize>,
    line_width_scale: f32,
) -> Vec<GpuLine> {
    if selected == 0 || range.is_empty() || scene.lines.len() == 0 {
        return Vec::new();
    }

    projected_range(
        scene,
        size,
        dark,
        camera,
        range,
        line_width_scale,
        |output_index| {
            // This quotient samples the whole stream (including its tail) without
            // depending on a floating-point stride or a semantic line ceiling.
            if selected == 1 {
                scene.lines.len() / 2
            } else {
                ((output_index as u128 * (scene.lines.len() - 1) as u128) / (selected - 1) as u128)
                    as usize
            }
        },
    )
}

fn projected_range(
    scene: &LineScene,
    size: Size,
    dark: bool,
    camera: Camera2d,
    range: std::ops::Range<usize>,
    line_width_scale: f32,
    mut source_index: impl FnMut(usize) -> usize,
) -> Vec<GpuLine> {
    if range.is_empty() || scene.lines.len() == 0 {
        return Vec::new();
    }

    let mut instances = Vec::new();
    if instances.try_reserve_exact(range.len()).is_err() {
        return instances;
    }

    let bounds = view_bounds(scene.bounds);
    let transform = ViewTransform::new(bounds, viewport_size(size), camera);
    let base_width = adaptive_scene_stroke_width(
        scene.total_line_length,
        scene.lines.len(),
        render_bounds_extent(scene.bounds),
        transform.fit_scale(),
    ) as f32
        * camera.zoom as f32
        * valid_line_width_scale(line_width_scale);
    for output_index in range {
        let line = scene.lines.get(source_index(output_index));
        let start = project(line.start, &transform);
        let end = project(line.end, &transform);
        let rgb = resolve_line_color(line.color, line, &transform, dark);
        let alpha = if line.width > 0.0 { 1.0 } else { 0.0 };
        instances.push(GpuLine {
            start,
            end,
            width: base_width * line.width.max(0.0),
            _padding: 0.0,
            color: [rgb[0], rgb[1], rgb[2], alpha],
            viewport: [size.width.max(1.0), size.height.max(1.0)],
        });
    }

    instances
}

fn projected_transition_preview(
    scene: &TransitionScene,
    size: Size,
    camera: Camera2d,
    dark: bool,
    progress: f32,
    selected: usize,
    line_width_scale: f32,
) -> Vec<GpuLine> {
    if selected == 0 || scene.lines.is_empty() {
        return Vec::new();
    }
    let mut instances = Vec::new();
    if instances.try_reserve_exact(selected).is_err() {
        return instances;
    }

    let source_bounds = if scene.source_line_count == 0 {
        scene.target_bounds
    } else {
        scene.source_bounds
    };
    let target_bounds = if scene.target_line_count == 0 {
        scene.source_bounds
    } else {
        scene.target_bounds
    };
    let viewport = viewport_size(size);
    let source_transform = ViewTransform::new(view_bounds(source_bounds), viewport, camera);
    let target_transform = ViewTransform::new(view_bounds(target_bounds), viewport, camera);
    let source_width = adaptive_scene_stroke_width(
        scene.source_total_line_length,
        scene.source_line_count,
        render_bounds_extent(source_bounds),
        source_transform.fit_scale(),
    ) as f32
        * camera.zoom as f32;
    let target_width = adaptive_scene_stroke_width(
        scene.target_total_line_length,
        scene.target_line_count,
        render_bounds_extent(target_bounds),
        target_transform.fit_scale(),
    ) as f32
        * camera.zoom as f32;
    let progress = progress.clamp(0.0, 1.0);
    let line_width_scale = valid_line_width_scale(line_width_scale);

    for output_index in 0..selected {
        let source_index = if selected == scene.lines.len() {
            output_index
        } else if selected == 1 {
            scene.lines.len() / 2
        } else {
            ((output_index as u128 * (scene.lines.len() - 1) as u128) / (selected - 1) as u128)
                as usize
        };
        let line = scene.lines.get(source_index);
        let source_endpoint_transform = match line.source_space {
            MorphCoordinateSpace::Source => &source_transform,
            MorphCoordinateSpace::Target => &target_transform,
        };
        let target_endpoint_transform = match line.target_space {
            MorphCoordinateSpace::Source => &source_transform,
            MorphCoordinateSpace::Target => &target_transform,
        };
        let source_endpoint_width = match line.source_space {
            MorphCoordinateSpace::Source => source_width,
            MorphCoordinateSpace::Target => target_width,
        };
        let target_endpoint_width = match line.target_space {
            MorphCoordinateSpace::Source => source_width,
            MorphCoordinateSpace::Target => target_width,
        };
        let source_start = project_line_point(line.source.line.0, source_endpoint_transform);
        let source_end = project_line_point(line.source.line.1, source_endpoint_transform);
        let target_start = project_line_point(line.target.line.0, target_endpoint_transform);
        let target_end = project_line_point(line.target.line.1, target_endpoint_transform);
        let source_color = resolve_styled_color(
            line.source.color,
            line.source_palette_line,
            source_endpoint_transform,
            dark,
        );
        let target_color = resolve_styled_color(
            line.target.color,
            line.target_palette_line,
            target_endpoint_transform,
            dark,
        );
        instances.push(GpuLine {
            start: lerp2(source_start, target_start, progress),
            end: lerp2(source_end, target_end, progress),
            width: lerp(
                source_endpoint_width * line.source.width.max(0.0) as f32,
                target_endpoint_width * line.target.width.max(0.0) as f32,
                progress,
            ) * line_width_scale,
            _padding: 0.0,
            color: [
                lerp(source_color[0], target_color[0], progress),
                lerp(source_color[1], target_color[1], progress),
                lerp(source_color[2], target_color[2], progress),
                lerp(line.source_opacity, line.target_opacity, progress),
            ],
            viewport: [size.width.max(1.0), size.height.max(1.0)],
        });
    }
    instances
}

fn project_line_point(point: (f64, f64), transform: &ViewTransform) -> [f32; 2] {
    let point = transform.project(WorldPoint::new(point.0, point.1));
    [point.x as f32, point.y as f32]
}

fn palette_for_line(line: Line2d, transform: &ViewTransform, dark: bool) -> [f32; 3] {
    let position = transform.world_palette_position(
        WorldPoint::new(line.0.0, line.0.1),
        WorldPoint::new(line.1.0, line.1.1),
    ) as f32;
    theme_default_color(
        theme_default_bucket(f64::from(position)),
        if dark { Palette::Dark } else { Palette::Light },
    )
}

fn display_background(source: Option<[u8; 3]>, dark: bool) -> Color {
    source
        .map(|[red, green, blue]| Color::from_rgb8(red, green, blue))
        .unwrap_or_else(|| {
            if dark {
                DARK_CANVAS_BACKGROUND
            } else {
                CANVAS_BACKGROUND
            }
        })
}

fn resolve_line_color(
    color: StrokeColor,
    line: SourceLine,
    transform: &ViewTransform,
    dark: bool,
) -> [f32; 3] {
    resolve_styled_color(
        color,
        Line2d(
            (f64::from(line.start[0]), f64::from(line.start[1])),
            (f64::from(line.end[0]), f64::from(line.end[1])),
        ),
        transform,
        dark,
    )
}

fn resolve_styled_color(
    color: StrokeColor,
    palette_line: Line2d,
    transform: &ViewTransform,
    dark: bool,
) -> [f32; 3] {
    let target_palette = if dark { Palette::Dark } else { Palette::Light };
    turtle_stroke_rgb(color, target_palette).map_or_else(
        || palette_for_line(palette_line, transform, dark),
        |[red, green, blue]| {
            [
                f32::from(red) / 255.0,
                f32::from(green) / 255.0,
                f32::from(blue) / 255.0,
            ]
        },
    )
}

fn lerp(left: f32, right: f32, progress: f32) -> f32 {
    left + (right - left) * progress
}

fn lerp2(left: [f32; 2], right: [f32; 2], progress: f32) -> [f32; 2] {
    [
        lerp(left[0], right[0], progress),
        lerp(left[1], right[1], progress),
    ]
}

fn project(point: [f32; 2], transform: &ViewTransform) -> [f32; 2] {
    let point = transform.project(WorldPoint::new(f64::from(point[0]), f64::from(point[1])));
    [point.x as f32, point.y as f32]
}

fn view_bounds(bounds: RenderBounds) -> ViewBounds {
    ViewBounds::new(
        f64::from(bounds.min_x),
        f64::from(bounds.max_x),
        f64::from(bounds.min_y),
        f64::from(bounds.max_y),
    )
}

fn render_bounds_extent(bounds: RenderBounds) -> (f64, f64) {
    (
        f64::from((bounds.max_x - bounds.min_x).max(0.0)),
        f64::from((bounds.max_y - bounds.min_y).max(0.0)),
    )
}

fn viewport_size(size: Size) -> ViewportSize {
    ViewportSize::new(
        f64::from(size.width.max(1.0)),
        f64::from(size.height.max(1.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_line_scene(line_count: usize) -> Arc<LineScene> {
        let mut builder = LineSceneBuilder::default();
        builder
            .extend((0..line_count).map(|index| StyledLine2d {
                line: Line2d((index as f64, 0.0), (index as f64 + 1.0, 1.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            }))
            .unwrap();
        builder.finish(RenderBounds {
            min_x: 0.0,
            max_x: line_count.max(1) as f32,
            min_y: 0.0,
            max_y: 1.0,
        })
    }

    fn test_line_primitive() -> LinePrimitive {
        LinePrimitive {
            scene: test_line_scene(8),
            dark: false,
            size: Size::new(640.0, 480.0),
            live_camera: Camera2d::fit(),
            settled_camera: Camera2d::fit(),
            interacting: false,
            raster_epoch: 1,
            line_width_scale: 1.0,
        }
    }

    #[test]
    fn line_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(include_str!("line_shader.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn host_gpu_struct_layouts_match_the_wgsl_contract() {
        assert_eq!(std::mem::size_of::<CompositeUniform>(), 32);
        assert_eq!(std::mem::offset_of!(CompositeUniform, sample_scale), 0);
        assert_eq!(std::mem::offset_of!(CompositeUniform, sample_offset), 8);
        assert_eq!(std::mem::offset_of!(CompositeUniform, background), 16);

        assert_eq!(std::mem::size_of::<GpuLine>(), 48);
        assert_eq!(std::mem::offset_of!(GpuLine, start), 0);
        assert_eq!(std::mem::offset_of!(GpuLine, end), 8);
        assert_eq!(std::mem::offset_of!(GpuLine, width), 16);
        assert_eq!(std::mem::offset_of!(GpuLine, _padding), 20);
        assert_eq!(std::mem::offset_of!(GpuLine, color), 24);
        assert_eq!(std::mem::offset_of!(GpuLine, viewport), 40);
    }

    #[test]
    fn transition_projection_matches_both_independent_fitted_endpoints() {
        let source_bounds = RenderBounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 2.0,
        };
        let target_bounds = RenderBounds {
            min_x: -2.0,
            max_x: 2.0,
            min_y: -8.0,
            max_y: 8.0,
        };
        let morph = MorphLine2d {
            source: StyledLine2d {
                line: Line2d((0.0, 0.0), (10.0, 2.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            },
            target: StyledLine2d {
                line: Line2d((-2.0, -8.0), (2.0, 8.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            },
            source_space: MorphCoordinateSpace::Source,
            target_space: MorphCoordinateSpace::Target,
            source_opacity: 1.0,
            target_opacity: 1.0,
            source_palette_line: Line2d((0.0, 0.0), (10.0, 2.0)),
            target_palette_line: Line2d((-2.0, -8.0), (2.0, 8.0)),
        };
        let scene = TransitionScene::new(
            vec![morph],
            source_bounds,
            target_bounds,
            (1, Some(100.0)),
            (1, Some(200.0)),
            None,
        );
        let size = Size::new(800.0, 600.0);
        let source =
            projected_transition_preview(&scene, size, Camera2d::fit(), false, 0.0, 1, 1.0);
        let target =
            projected_transition_preview(&scene, size, Camera2d::fit(), false, 1.0, 1, 1.0);
        let thin = projected_transition_preview(&scene, size, Camera2d::fit(), false, 0.5, 1, 0.05);
        let automatic =
            projected_transition_preview(&scene, size, Camera2d::fit(), false, 0.5, 1, 1.0);
        let source_transform = ViewTransform::new(
            view_bounds(source_bounds),
            viewport_size(size),
            Camera2d::fit(),
        );
        let target_transform = ViewTransform::new(
            view_bounds(target_bounds),
            viewport_size(size),
            Camera2d::fit(),
        );
        assert_eq!(
            source[0].start,
            project_line_point((0.0, 0.0), &source_transform)
        );
        assert_eq!(
            source[0].end,
            project_line_point((10.0, 2.0), &source_transform)
        );
        assert!((thin[0].width / automatic[0].width - 0.05).abs() < 1.0e-6);
        assert_eq!(
            target[0].start,
            project_line_point((-2.0, -8.0), &target_transform)
        );
        assert_eq!(
            target[0].end,
            project_line_point((2.0, 8.0), &target_transform)
        );
        let source_width = adaptive_scene_stroke_width(
            Some(100.0),
            1,
            render_bounds_extent(source_bounds),
            source_transform.fit_scale(),
        ) as f32;
        let target_width = adaptive_scene_stroke_width(
            Some(200.0),
            1,
            render_bounds_extent(target_bounds),
            target_transform.fit_scale(),
        ) as f32;
        assert!((source[0].width - source_width).abs() < 1.0e-5);
        assert!((target[0].width - target_width).abs() < 1.0e-5);
        assert_ne!(source[0].width, target[0].width);
    }

    #[test]
    fn disappearing_endpoint_stays_in_the_source_coordinate_space() {
        let source_bounds = RenderBounds {
            min_x: 0.0,
            max_x: 10.0,
            min_y: 0.0,
            max_y: 10.0,
        };
        let target_bounds = RenderBounds {
            min_x: -100.0,
            max_x: 100.0,
            min_y: -100.0,
            max_y: 100.0,
        };
        let midpoint = (2.0, 8.0);
        let scene = TransitionScene::new(
            vec![MorphLine2d {
                source: StyledLine2d {
                    line: Line2d((1.0, 8.0), (3.0, 8.0)),
                    width: 1.0,
                    color: StrokeColor::ThemeDefault,
                },
                target: StyledLine2d {
                    line: Line2d(midpoint, midpoint),
                    width: 0.0,
                    color: StrokeColor::ThemeDefault,
                },
                source_space: MorphCoordinateSpace::Source,
                target_space: MorphCoordinateSpace::Source,
                source_opacity: 1.0,
                target_opacity: 0.0,
                source_palette_line: Line2d((1.0, 8.0), (3.0, 8.0)),
                target_palette_line: Line2d(midpoint, midpoint),
            }],
            source_bounds,
            target_bounds,
            (1, None),
            (1, None),
            None,
        );
        let size = Size::new(800.0, 600.0);
        let projected =
            projected_transition_preview(&scene, size, Camera2d::fit(), false, 1.0, 1, 1.0);
        let source_transform = ViewTransform::new(
            view_bounds(source_bounds),
            viewport_size(size),
            Camera2d::fit(),
        );
        let target_transform = ViewTransform::new(
            view_bounds(target_bounds),
            viewport_size(size),
            Camera2d::fit(),
        );

        assert_eq!(
            projected[0].start,
            project_line_point(midpoint, &source_transform)
        );
        assert_ne!(
            projected[0].start,
            project_line_point(midpoint, &target_transform)
        );
    }

    #[test]
    fn transition_preview_sampling_is_bounded_and_spans_the_stream() {
        let morphs = (0..10_000)
            .map(|index| {
                let line = Line2d((index as f64, 0.0), (index as f64 + 1.0, 0.0));
                MorphLine2d {
                    source: StyledLine2d {
                        line,
                        width: 1.0,
                        color: StrokeColor::ThemeDefault,
                    },
                    target: StyledLine2d {
                        line,
                        width: 1.0,
                        color: StrokeColor::ThemeDefault,
                    },
                    source_space: MorphCoordinateSpace::Source,
                    target_space: MorphCoordinateSpace::Target,
                    source_opacity: 1.0,
                    target_opacity: 1.0,
                    source_palette_line: line,
                    target_palette_line: line,
                }
            })
            .collect();
        let bounds = RenderBounds {
            min_x: 0.0,
            max_x: 10_000.0,
            min_y: 0.0,
            max_y: 1.0,
        };
        let scene =
            TransitionScene::new(morphs, bounds, bounds, (10_000, None), (10_000, None), None);
        let selected = 257;
        let preview = projected_transition_preview(
            &scene,
            Size::new(640.0, 480.0),
            Camera2d::fit(),
            false,
            0.5,
            selected,
            1.0,
        );
        assert_eq!(scene.line_count(), 10_000);
        assert_eq!(preview.len(), selected);
        assert_ne!(
            preview.first().unwrap().start,
            preview.last().unwrap().start
        );
    }

    #[test]
    fn prepared_key_ignores_live_camera_interaction_and_widget_origin() {
        let primitive = test_line_primitive();
        let bounds = Rectangle {
            x: 12.0,
            y: 34.0,
            width: 640.0,
            height: 480.0,
        };
        let moved_bounds = Rectangle {
            x: 321.0,
            y: 123.0,
            ..bounds
        };
        let baseline = primitive.prepared_key(&bounds, 2.0);

        let mut live = primitive.clone();
        live.live_camera = Camera2d {
            zoom: 7.0,
            focus: [0.25, 0.75],
        };
        live.interacting = true;

        assert_eq!(live.prepared_key(&moved_bounds, 2.0), baseline);
    }

    #[test]
    fn prepared_key_changes_for_settled_camera_and_raster_inputs() {
        let primitive = test_line_primitive();
        let bounds = Rectangle {
            x: 0.0,
            y: 0.0,
            width: 640.0,
            height: 480.0,
        };
        let baseline = primitive.prepared_key(&bounds, 1.0);

        let mut settled = primitive.clone();
        settled.settled_camera = Camera2d {
            zoom: 2.0,
            focus: [0.25, 0.75],
        };
        assert_ne!(settled.prepared_key(&bounds, 1.0), baseline);

        let mut invalidated = primitive.clone();
        invalidated.raster_epoch = invalidated.raster_epoch.wrapping_add(1);
        assert_ne!(invalidated.prepared_key(&bounds, 1.0), baseline);

        let mut thinner = primitive.clone();
        thinner.line_width_scale = 0.05;
        assert_ne!(thinner.prepared_key(&bounds, 1.0), baseline);

        let resized = Rectangle {
            width: 800.0,
            ..bounds
        };
        assert_ne!(primitive.prepared_key(&resized, 1.0), baseline);
        assert_ne!(primitive.prepared_key(&bounds, 2.0), baseline);

        let mut recolored = primitive.clone();
        recolored.dark = true;
        assert_ne!(recolored.prepared_key(&bounds, 1.0), baseline);
    }

    #[test]
    fn physical_target_tracks_absolute_widget_position_and_dpi() {
        let target = physical_target(
            &Rectangle {
                x: 12.0,
                y: 34.0,
                width: 640.0,
                height: 480.0,
            },
            1.5,
        );

        assert_eq!(target.x, 18.0);
        assert_eq!(target.y, 51.0);
        assert_eq!(target.width, 960.0);
        assert_eq!(target.height, 720.0);
    }

    #[test]
    fn composite_uniform_is_identity_for_the_raster_camera() {
        let bounds = ViewBounds::new(-10.0, 30.0, -20.0, 20.0);
        let size = ViewportSize::new(800.0, 600.0);
        let camera = Camera2d {
            zoom: 3.0,
            focus: [0.25, 0.75],
        };
        let uniform = CompositeUniform::new(bounds, size, camera, camera, false, None);

        assert!((uniform.sample_scale[0] - 1.0).abs() < 1.0e-6);
        assert!((uniform.sample_scale[1] - 1.0).abs() < 1.0e-6);
        assert!(uniform.sample_offset[0].abs() < 1.0e-6);
        assert!(uniform.sample_offset[1].abs() < 1.0e-6);
    }

    #[test]
    fn composite_uniform_maps_live_pixels_back_to_the_raster_camera() {
        let bounds = ViewBounds::new(-10.0, 30.0, -20.0, 20.0);
        let size = ViewportSize::new(800.0, 600.0);
        let raster_camera = Camera2d::fit();
        let live_camera = Camera2d {
            zoom: 4.0,
            focus: [-0.5, 1.25],
        };
        let uniform = CompositeUniform::new(bounds, size, raster_camera, live_camera, true, None);
        let screen = crate::camera::ScreenPoint::new(137.0, 419.0);
        let raster = ViewTransform::new(bounds, size, raster_camera);
        let live = ViewTransform::new(bounds, size, live_camera);
        let expected = raster.project(live.unproject(screen));
        let actual_x = (screen.x / size.width * f64::from(uniform.sample_scale[0])
            + f64::from(uniform.sample_offset[0]))
            * size.width;
        let actual_y = (screen.y / size.height * f64::from(uniform.sample_scale[1])
            + f64::from(uniform.sample_offset[1]))
            * size.height;

        assert!((actual_x - expected.x).abs() < 1.0e-4);
        assert!((actual_y - expected.y).abs() < 1.0e-4);
    }

    #[test]
    fn preview_is_resolution_derived_and_never_exceeds_scene() {
        let small = preview_budget_for_viewport(Size::new(100.0, 100.0));
        let large = preview_budget_for_viewport(Size::new(1_000.0, 1_000.0));
        assert!(small >= MIN_PREVIEW_SEGMENTS);
        assert!(large > small);
    }

    #[test]
    fn selection_respects_mocked_device_buffer_limit() {
        let instance_size = std::mem::size_of::<GpuLine>() as u64;
        let selected =
            preview_selection_count(usize::MAX, Size::new(8_000.0, 8_000.0), instance_size * 17);
        assert_eq!(selected, 17);
        assert!(selected as u64 * instance_size <= instance_size * 17);
    }

    #[test]
    fn progressive_ranges_cover_the_selected_stream_without_overlap() {
        let scene = LineScene {
            id: 1,
            lines: LineStorage::Owned(
                (0..10)
                    .map(|index| SourceLine {
                        start: [index as f32, 0.0],
                        end: [index as f32 + 0.5, 0.0],
                        width: 1.0,
                        color: StrokeColor::ThemeDefault,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            bounds: RenderBounds {
                min_x: 0.0,
                max_x: 10.0,
                min_y: 0.0,
                max_y: 1.0,
            },
            total_line_length: Some(5.0),
            background: None,
            refined_lines: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
        };
        let size = Size::new(100.0, 100.0);
        let first = projected_preview_range(&scene, size, false, Camera2d::default(), 4, 0..2, 1.0);
        let second =
            projected_preview_range(&scene, size, false, Camera2d::default(), 4, 2..4, 1.0);

        assert_eq!(first.len() + second.len(), 4);
        assert!((first[0].start[0] - 12.0).abs() < 0.001);
        assert!((second[1].start[0] - 80.4).abs() < 0.001);
    }

    #[test]
    fn camera_changes_geometry_and_width_without_recoloring_lines() {
        let scene = LineScene {
            id: 1,
            lines: LineStorage::Owned(
                [SourceLine {
                    start: [2.0, 3.0],
                    end: [7.0, 8.0],
                    width: 1.0,
                    color: StrokeColor::ThemeDefault,
                }]
                .into(),
            ),
            bounds: RenderBounds {
                min_x: 0.0,
                max_x: 10.0,
                min_y: 0.0,
                max_y: 10.0,
            },
            total_line_length: Some(std::f64::consts::SQRT_2 * 5.0),
            background: None,
            refined_lines: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
        };
        let size = Size::new(500.0, 300.0);
        let fitted = projected_exact_range(&scene, size, false, Camera2d::fit(), 0..1, 1.0);
        let navigated = projected_exact_range(
            &scene,
            size,
            false,
            Camera2d {
                zoom: 4.0,
                focus: [0.1, 0.9],
            },
            0..1,
            1.0,
        );

        assert_ne!(fitted[0].start, navigated[0].start);
        assert_eq!(fitted[0].color, navigated[0].color);
        assert!((navigated[0].width / fitted[0].width - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn exact_projection_applies_the_line_width_scale_only_to_width() {
        let scene = test_line_scene(1);
        let size = Size::new(500.0, 300.0);
        let automatic = projected_exact_range(&scene, size, false, Camera2d::fit(), 0..1, 1.0);
        let thin = projected_exact_range(&scene, size, false, Camera2d::fit(), 0..1, 0.05);

        assert_eq!(thin[0].start, automatic[0].start);
        assert_eq!(thin[0].end, automatic[0].end);
        assert_eq!(thin[0].color, automatic[0].color);
        assert!((thin[0].width / automatic[0].width - 0.05).abs() < 1.0e-6);
    }

    #[test]
    fn accumulation_texture_preserves_aspect_ratio_within_budget() {
        let (width, height) = accumulation_extent(8_000, 4_000, 16_384, 4_000_000);
        assert!(u64::from(width) * u64::from(height) <= 4_000_000);
        assert!((width as f32 / height as f32 - 2.0).abs() < 0.01);
    }

    #[test]
    fn accumulation_texture_preserves_aspect_ratio_at_dimension_limit() {
        let (wide_width, wide_height) = accumulation_extent(32_768, 4_096, 8_192, u64::MAX);
        assert_eq!((wide_width, wide_height), (8_192, 1_024));

        let (tall_width, tall_height) = accumulation_extent(4_096, 32_768, 8_192, u64::MAX);
        assert_eq!((tall_width, tall_height), (1_024, 8_192));
    }

    #[test]
    fn line_scene_builder_keeps_exact_lines_without_a_canvas_scene() {
        let scene = test_line_scene(100);
        assert_eq!(scene.line_count(), 100);
        assert_eq!(scene.refinement_progress(), (0, 100));
        assert!(
            (scene.total_line_length().unwrap() - 100.0 * std::f64::consts::SQRT_2).abs() < 1.0e-12
        );
    }

    #[test]
    fn settled_view_restart_resets_active_refinement_progress() {
        let scene = test_line_scene(100);
        scene.set_refinement_progress_for_test(17);
        scene.restart_view_refinement();
        assert_eq!(scene.refinement_progress(), (0, 100));
    }

    #[test]
    fn explicit_refinement_cancel_is_sticky_across_view_restarts() {
        let scene = test_line_scene(100);
        scene.cancel_refinement();
        assert!(scene.refinement_was_cancelled());
        scene.set_refinement_progress_for_test(17);
        scene.restart_view_refinement();
        assert_eq!(scene.refinement_progress(), (17, 100));
        assert!(!scene.is_refining());
    }

    #[test]
    fn software_fallback_is_bounded_and_spans_the_exact_scene() {
        let mut builder = LineSceneBuilder::default();
        builder
            .extend((0..10_000).map(|index| StyledLine2d {
                line: Line2d((index as f64, 0.0), (index as f64 + 1.0, 0.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            }))
            .unwrap();
        let scene = builder.finish(RenderBounds {
            min_x: 0.0,
            max_x: 10_000.0,
            min_y: 0.0,
            max_y: 0.0,
        });

        let preview = scene.fallback_preview(257).unwrap();
        assert_eq!(preview.primitives.len(), 257);
        let Primitive2d::Line(first) = &preview.primitives[0] else {
            panic!("fallback emitted a non-line primitive")
        };
        let Primitive2d::Line(last) = &preview.primitives[256] else {
            panic!("fallback emitted a non-line primitive")
        };
        assert_eq!(first.line.0.0, 0.0);
        assert_eq!(last.line.1.0, 10_000.0);
    }

    #[test]
    fn svg_encoding_observes_cancellation_between_bounded_batches() {
        let mut builder = LineSceneBuilder::default();
        builder
            .extend((0..(SVG_LINES_PER_BATCH * 3)).map(|index| StyledLine2d {
                line: Line2d((index as f64, 0.0), (index as f64 + 1.0, 0.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            }))
            .unwrap();
        let scene = builder.finish(RenderBounds {
            min_x: 0.0,
            max_x: (SVG_LINES_PER_BATCH * 3) as f32,
            min_y: 0.0,
            max_y: 0.0,
        });
        let probes = std::cell::Cell::new(0usize);
        let result = scene.encode_svg_with_cancel(Palette::Light, 1.0, &|| {
            probes.set(probes.get() + 1);
            probes.get() >= 3
        });

        assert_eq!(result, Err(LineSceneError::Cancelled));
    }

    #[test]
    fn svg_encoding_applies_the_line_width_scale() {
        let scene = test_line_scene(1);
        let automatic = scene
            .encode_svg_with_cancel(Palette::Light, 1.0, &|| false)
            .unwrap();
        let thin = scene
            .encode_svg_with_cancel(Palette::Light, 0.05, &|| false)
            .unwrap();
        let stroke_width = |svg: &str| {
            svg.split("stroke-width=\"")
                .nth(1)
                .and_then(|value| value.split('"').next())
                .unwrap()
                .parse::<f32>()
                .unwrap()
        };

        assert!((stroke_width(&thin) / stroke_width(&automatic) - 0.05).abs() < 1.0e-6);
    }
}
