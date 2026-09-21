//! Retained XYZ display geometry for the Iced frontend.
//!
//! Rotation stays entirely in this display stage: the visualization result is
//! retained in turtle-world coordinates, while the shared camera module owns
//! the orthographic sphere fit used by the WGPU and Canvas paths.
//! Source widths are normalized only when displayed. Theme-default geometry
//! uses the shared emerald spatial palette, and both explicit and default
//! albedos receive the same camera-space rod/surface lighting and opaque depth
//! cue in accelerated, software, and SVG paths.

use super::{CANVAS_BACKGROUND, DARK_CANVAS_BACKGROUND, SOFTWARE_FALLBACK_PREVIEW_LINES};
use crate::camera::{Orbit3d, ViewBounds3d, ViewTransform3d, ViewportSize, WorldPoint3d};
use braken_viz::targets::{
    Palette, adaptive_spatial_scene_stroke_width, spatial_lit_color_bounded,
    spatial_theme_default_color_at, turtle_stroke_rgb,
};
#[cfg(target_arch = "wasm32")]
use braken_viz::targets::{StrokeWidthEstimator, adaptive_spatial_scene_svg_stroke_width};
use braken_viz::{
    Line2d, Polygon2d, Polygon3d, Primitive2d, Primitive3d, Scene2d, Scene3d, StrokeColor,
    StyledLine2d, StyledLine3d, Turtle3dPrimitiveKind, Turtle3dWidthReferenceEstimator,
    normalized_turtle_3d_width,
};
use bytemuck::{Pod, Zeroable};
use iced::wgpu;
use iced::widget::shader;
use iced::{Color, Point, Rectangle, Size};
use std::error::Error;
use std::fmt;
#[cfg(target_arch = "wasm32")]
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn spatial_depth_state(
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled,
        depth_compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}
const MOVING_GPU_PREVIEW_LINES: usize = 32 * 1024;
const GPU_UPLOAD_CHUNK_ITEMS: usize = 64 * 1024;
#[cfg(target_arch = "wasm32")]
const SVG_WORK_PER_BATCH: usize = 2 * 1024;
static NEXT_SPATIAL_SCENE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct RenderBounds3d {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
}

impl From<RenderBounds3d> for ViewBounds3d {
    fn from(bounds: RenderBounds3d) -> Self {
        Self::new(
            f64::from(bounds.min_x),
            f64::from(bounds.max_x),
            f64::from(bounds.min_y),
            f64::from(bounds.max_y),
            f64::from(bounds.min_z),
            f64::from(bounds.max_z),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct SourceLine3d {
    start: [f32; 3],
    end: [f32; 3],
    width: f32,
    color: StrokeColor,
}

#[derive(Debug)]
enum LineStorage3d {
    Owned(Box<[SourceLine3d]>),
    #[cfg(target_arch = "wasm32")]
    Transferred {
        chunks: Box<[js_sys::Float32Array]>,
        line_count: usize,
    },
}

/// Polygon payload plus its compact position in the mixed turtle stream.
/// `lines_before` is cumulative across all source batches.
#[derive(Debug)]
pub(super) struct SpatialPolygon3d {
    vertices: Vec<(f64, f64, f64)>,
    color: StrokeColor,
    lines_before: usize,
}

impl SpatialPolygon3d {
    pub(super) fn new(polygon: Polygon3d, lines_before: usize) -> Self {
        Self {
            vertices: polygon.vertices,
            color: polygon.color,
            lines_before,
        }
    }
}

impl LineStorage3d {
    fn len(&self) -> usize {
        match self {
            Self::Owned(lines) => lines.len(),
            #[cfg(target_arch = "wasm32")]
            Self::Transferred { line_count, .. } => *line_count,
        }
    }

    fn get(&self, index: usize) -> SourceLine3d {
        match self {
            Self::Owned(lines) => lines[index],
            #[cfg(target_arch = "wasm32")]
            Self::Transferred { chunks, .. } => {
                const VALUES: usize = super::worker_protocol::LINE_3D_TRANSFER_VALUES;
                let chunk_index = index / super::worker_protocol::LINE_TRANSFER_CHUNK_LINES;
                let line_in_chunk = index % super::worker_protocol::LINE_TRANSFER_CHUNK_LINES;
                let values = &chunks[chunk_index];
                let offset =
                    u32::try_from(line_in_chunk.saturating_mul(VALUES)).unwrap_or(u32::MAX);
                let value = |value_index: usize| {
                    values.get_index(
                        offset.saturating_add(u32::try_from(value_index).unwrap_or(u32::MAX)),
                    )
                };
                SourceLine3d {
                    start: [value(0), value(1), value(2)],
                    end: [value(3), value(4), value(5)],
                    width: value(6),
                    color: super::worker_protocol::decode_stroke_color(value(7)),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpatialSceneError {
    ResourceExhausted {
        requested: usize,
    },
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    InvalidTransfer {
        expected_values: usize,
        actual_values: usize,
    },
    InvalidGeometry(&'static str),
    Cancelled,
}

impl fmt::Display for SpatialSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceExhausted { requested } => {
                write!(
                    formatter,
                    "not enough memory for {requested} spatial primitives"
                )
            }
            Self::InvalidTransfer {
                expected_values,
                actual_values,
            } => write!(
                formatter,
                "render worker transferred {actual_values} spatial line values; expected {expected_values}",
            ),
            Self::InvalidGeometry(detail) => {
                write!(formatter, "invalid spatial geometry: {detail}")
            }
            Self::Cancelled => formatter.write_str("spatial scene operation cancelled"),
        }
    }
}

impl Error for SpatialSceneError {}

/// Immutable source-space geometry shared by the software and WGPU displays.
#[derive(Debug)]
pub(super) struct SpatialScene {
    id: u64,
    lines: LineStorage3d,
    polygons: Box<[SpatialPolygon3d]>,
    bounds: RenderBounds3d,
    total_line_length: Option<f64>,
    width_reference: f64,
    polygon_triangle_count: usize,
    background: Option<[u8; 3]>,
    refinement_total: usize,
    refined_work: AtomicUsize,
    refinement_cancelled: AtomicBool,
    gpu_refinement_failed: AtomicBool,
    gpu_failure_requested: AtomicUsize,
}

/// Incrementally collects bounded turtle batches directly into retained XYZ
/// storage. This avoids materializing a second, mixed `Scene3d` allocation
/// before the display can take ownership of the completed geometry.
#[derive(Debug, Default)]
pub(super) struct SpatialSceneBuilder {
    lines: Vec<SourceLine3d>,
    polygons: Vec<SpatialPolygon3d>,
    total_line_length: f64,
    has_line_length: bool,
    width_reference: Turtle3dWidthReferenceEstimator,
    bounds: Option<RenderBounds3d>,
    background: Option<[u8; 3]>,
}

impl SpatialSceneBuilder {
    pub(super) fn new(background: Option<[u8; 3]>) -> Self {
        Self {
            background,
            ..Self::default()
        }
    }

    pub(super) fn extend(
        &mut self,
        lines: Vec<StyledLine3d>,
        polygons: Vec<Polygon3d>,
        primitive_order: Vec<Turtle3dPrimitiveKind>,
    ) -> Result<(), SpatialSceneError> {
        let primitive_count = lines.len().checked_add(polygons.len()).ok_or(
            SpatialSceneError::ResourceExhausted {
                requested: usize::MAX,
            },
        )?;
        let tagged_lines = primitive_order
            .iter()
            .filter(|kind| matches!(kind, Turtle3dPrimitiveKind::Line))
            .count();
        if primitive_order.len() != primitive_count
            || tagged_lines != lines.len()
            || primitive_order.len().saturating_sub(tagged_lines) != polygons.len()
        {
            return Err(SpatialSceneError::InvalidGeometry(
                "turtle batch primitive order does not match its payloads",
            ));
        }
        self.lines
            .try_reserve(lines.len())
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: self.lines.len().saturating_add(lines.len()),
            })?;
        self.polygons.try_reserve(polygons.len()).map_err(|_| {
            SpatialSceneError::ResourceExhausted {
                requested: self.polygons.len().saturating_add(polygons.len()),
            }
        })?;
        let mut lines = lines.into_iter();
        let mut polygons = polygons.into_iter();
        for kind in primitive_order {
            match kind {
                Turtle3dPrimitiveKind::Line => {
                    let line = lines.next().ok_or(SpatialSceneError::InvalidGeometry(
                        "turtle batch line order exhausted its payload",
                    ))?;
                    self.push_line(line)?;
                }
                Turtle3dPrimitiveKind::Polygon => {
                    let polygon = polygons.next().ok_or(SpatialSceneError::InvalidGeometry(
                        "turtle batch polygon order exhausted its payload",
                    ))?;
                    self.push_polygon(polygon)?;
                }
            }
        }
        debug_assert!(lines.next().is_none());
        debug_assert!(polygons.next().is_none());
        Ok(())
    }

    fn push_line(&mut self, line: StyledLine3d) -> Result<(), SpatialSceneError> {
        let source_start = line.line.0;
        let source_end = line.line.1;
        let start = checked_point(source_start)?;
        let end = checked_point(source_end)?;
        let width = line.width as f32;
        if !line.width.is_finite() || !width.is_finite() || width < 0.0 {
            return Err(SpatialSceneError::InvalidGeometry(
                "line width is negative or outside the finite f32 range",
            ));
        }
        let length = ((source_end.0 - source_start.0).powi(2)
            + (source_end.1 - source_start.1).powi(2)
            + (source_end.2 - source_start.2).powi(2))
        .sqrt();
        if length.is_finite() && length > 0.0 {
            self.total_line_length += length;
            self.has_line_length = self.total_line_length.is_finite();
        }
        self.width_reference.observe(length, line.width);
        self.include_point(start);
        self.include_point(end);
        self.lines.push(SourceLine3d {
            start,
            end,
            width,
            color: line.color,
        });
        Ok(())
    }

    fn push_polygon(&mut self, polygon: Polygon3d) -> Result<(), SpatialSceneError> {
        for &point in &polygon.vertices {
            checked_point(point)?;
        }
        self.polygons
            .try_reserve(1)
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: self.polygons.len().saturating_add(1),
            })?;
        for &point in &polygon.vertices {
            self.include_point(checked_point(point)?);
        }
        self.polygons
            .push(SpatialPolygon3d::new(polygon, self.lines.len()));
        Ok(())
    }

    fn include_point(&mut self, [x, y, z]: [f32; 3]) {
        let bounds = self.bounds.get_or_insert(RenderBounds3d {
            min_x: x,
            max_x: x,
            min_y: y,
            max_y: y,
            min_z: z,
            max_z: z,
        });
        bounds.min_x = bounds.min_x.min(x);
        bounds.max_x = bounds.max_x.max(x);
        bounds.min_y = bounds.min_y.min(y);
        bounds.max_y = bounds.max_y.max(y);
        bounds.min_z = bounds.min_z.min(z);
        bounds.max_z = bounds.max_z.max(z);
    }

    pub(super) fn finish(self) -> Result<Arc<SpatialScene>, SpatialSceneError> {
        let polygon_triangle_count =
            checked_triangle_count(self.polygons.iter().map(|polygon| polygon.vertices.len()))?;
        let refinement_total = checked_refinement_total(self.lines.len(), polygon_triangle_count)?;
        Ok(Arc::new(SpatialScene {
            id: NEXT_SPATIAL_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: LineStorage3d::Owned(self.lines.into_boxed_slice()),
            polygons: self.polygons.into_boxed_slice(),
            bounds: self.bounds.unwrap_or_default(),
            total_line_length: self.has_line_length.then_some(self.total_line_length),
            width_reference: self.width_reference.width_reference(),
            polygon_triangle_count,
            background: self.background,
            refinement_total,
            refined_work: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
            gpu_refinement_failed: AtomicBool::new(false),
            gpu_failure_requested: AtomicUsize::new(0),
        }))
    }
}

impl SpatialScene {
    pub(super) fn from_scene(scene: Scene3d) -> Result<Arc<Self>, SpatialSceneError> {
        let mut builder = SpatialSceneBuilder::new(scene.background);
        builder
            .lines
            .try_reserve(scene.primitives.len())
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: scene.primitives.len(),
            })?;
        builder
            .polygons
            .try_reserve(scene.primitives.len().min(1024))
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: scene.primitives.len(),
            })?;
        for primitive in scene.primitives {
            match primitive {
                Primitive3d::Line(line) => builder.push_line(line)?,
                Primitive3d::Polygon(polygon) => builder.push_polygon(polygon)?,
            }
        }
        builder.finish()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn from_transferred(
        values: js_sys::Array,
        line_count: usize,
        polygons: Vec<SpatialPolygon3d>,
        bounds: RenderBounds3d,
        total_line_length: Option<f64>,
        width_reference: f64,
        background: Option<[u8; 3]>,
    ) -> Result<Arc<Self>, SpatialSceneError> {
        use wasm_bindgen::JsCast;

        const VALUES: usize = super::worker_protocol::LINE_3D_TRANSFER_VALUES;
        let expected_values =
            line_count
                .checked_mul(VALUES)
                .ok_or(SpatialSceneError::InvalidTransfer {
                    expected_values: usize::MAX,
                    actual_values: 0,
                })?;
        let expected_chunks =
            line_count.div_ceil(super::worker_protocol::LINE_TRANSFER_CHUNK_LINES);
        if values.length() as usize != expected_chunks {
            return Err(SpatialSceneError::InvalidTransfer {
                expected_values,
                actual_values: 0,
            });
        }
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(expected_chunks).map_err(|_| {
            SpatialSceneError::ResourceExhausted {
                requested: line_count,
            }
        })?;
        let mut actual_values = 0usize;
        for (chunk_index, value) in values.iter().enumerate() {
            let chunk = value.dyn_into::<js_sys::Float32Array>().map_err(|_| {
                SpatialSceneError::InvalidTransfer {
                    expected_values,
                    actual_values,
                }
            })?;
            let remaining = line_count.saturating_sub(
                chunk_index.saturating_mul(super::worker_protocol::LINE_TRANSFER_CHUNK_LINES),
            );
            let expected = remaining
                .min(super::worker_protocol::LINE_TRANSFER_CHUNK_LINES)
                .saturating_mul(VALUES);
            if chunk.length() as usize != expected {
                return Err(SpatialSceneError::InvalidTransfer {
                    expected_values,
                    actual_values: actual_values.saturating_add(chunk.length() as usize),
                });
            }
            actual_values = actual_values.saturating_add(chunk.length() as usize);
            chunks.push(chunk);
        }
        if actual_values != expected_values {
            return Err(SpatialSceneError::InvalidTransfer {
                expected_values,
                actual_values,
            });
        }
        if !bounds.is_finite_ordered() {
            return Err(SpatialSceneError::InvalidGeometry(
                "worker bounds are non-finite or inverted",
            ));
        }
        if total_line_length.is_some_and(|length| !length.is_finite() || length < 0.0) {
            return Err(SpatialSceneError::InvalidGeometry(
                "worker line-length metric is non-finite or negative",
            ));
        }
        if !width_reference.is_finite() || width_reference < 1.0 {
            return Err(SpatialSceneError::InvalidGeometry(
                "worker width reference is non-finite or less than one",
            ));
        }
        validate_spatial_polygon_order(&polygons, line_count)?;
        for polygon in &polygons {
            for &point in &polygon.vertices {
                let point = checked_point(point)?;
                if !bounds.contains(point) {
                    return Err(SpatialSceneError::InvalidGeometry(
                        "worker bounds do not contain a polygon vertex",
                    ));
                }
            }
        }
        let polygon_triangle_count =
            checked_triangle_count(polygons.iter().map(|polygon| polygon.vertices.len()))?;
        let refinement_total = checked_refinement_total(line_count, polygon_triangle_count)?;
        Ok(Arc::new(Self {
            id: NEXT_SPATIAL_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            lines: LineStorage3d::Transferred {
                chunks: chunks.into_boxed_slice(),
                line_count,
            },
            polygons: polygons.into_boxed_slice(),
            bounds,
            total_line_length,
            width_reference,
            polygon_triangle_count,
            background,
            refinement_total,
            refined_work: AtomicUsize::new(0),
            refinement_cancelled: AtomicBool::new(false),
            gpu_refinement_failed: AtomicBool::new(false),
            gpu_failure_requested: AtomicUsize::new(0),
        }))
    }

    pub(super) fn background(&self) -> Option<[u8; 3]> {
        self.background
    }

    pub(super) fn width_reference(&self) -> f64 {
        self.width_reference
    }

    fn refinement_total(&self) -> usize {
        self.refinement_total
    }

    pub(super) fn refinement_progress(&self) -> (usize, usize) {
        let total = self.refinement_total();
        (self.refined_work.load(Ordering::Acquire).min(total), total)
    }

    pub(super) fn is_refining(&self) -> bool {
        let (completed, total) = self.refinement_progress();
        !self.refinement_cancelled.load(Ordering::Acquire)
            && self.gpu_refinement_failure().is_none()
            && completed < total
    }

    pub(super) fn refinement_was_cancelled(&self) -> bool {
        self.refinement_cancelled.load(Ordering::Acquire)
    }

    pub(super) fn cancel_refinement(&self) {
        self.refinement_cancelled.store(true, Ordering::Release);
    }

    pub(super) fn restart_refinement(&self) {
        if !self.refinement_cancelled.load(Ordering::Acquire)
            && self.gpu_refinement_failure().is_none()
        {
            self.refined_work.store(0, Ordering::Release);
        }
    }

    pub(super) fn gpu_refinement_failure(&self) -> Option<SpatialSceneError> {
        self.gpu_refinement_failed.load(Ordering::Acquire).then(|| {
            SpatialSceneError::ResourceExhausted {
                requested: self.gpu_failure_requested.load(Ordering::Relaxed),
            }
        })
    }

    pub(super) fn record_gpu_refinement_failure(&self, error: SpatialSceneError) {
        let requested = match error {
            SpatialSceneError::ResourceExhausted { requested } => requested,
            _ => self.refinement_total(),
        };
        self.gpu_failure_requested
            .store(requested, Ordering::Relaxed);
        self.gpu_refinement_failed.store(true, Ordering::Release);
    }

    pub(super) fn mark_software_preview_complete(&self) {
        self.refined_work
            .store(self.refinement_total(), Ordering::Release);
    }

    fn source_line(&self, selected_index: usize, selected: usize) -> SourceLine3d {
        self.lines.get(representative_index(
            selected_index,
            selected,
            self.lines.len(),
        ))
    }

    fn base_stroke_width(&self, transform: ViewTransform3d, line_width_scale: f32) -> f32 {
        let diameter = self.bounds.into_view_bounds().fit_radius() * 2.0;
        let scale = if line_width_scale.is_finite() && line_width_scale > 0.0 {
            line_width_scale
        } else {
            1.0
        };
        (adaptive_spatial_scene_stroke_width(
            self.total_line_length,
            self.lines.len(),
            (diameter, diameter),
            transform.stroke_scale(),
        ) as f32
            * scale)
            .max(0.0)
    }

    pub(super) fn software_preview(
        &self,
        size: Size,
        orbit: Orbit3d,
        dark: bool,
        line_width_scale: f32,
    ) -> Result<Vec<ProjectedPrimitive3d>, SpatialSceneError> {
        let selected = self.lines.len().min(SOFTWARE_FALLBACK_PREVIEW_LINES);
        let mut projected = Vec::new();
        projected
            .try_reserve(selected.saturating_add(self.polygons.len()))
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: selected.saturating_add(self.polygons.len()),
            })?;
        let transform =
            ViewTransform3d::new(self.bounds.into_view_bounds(), viewport_size(size), orbit);
        let base_width = self.base_stroke_width(transform, line_width_scale);
        for index in 0..selected {
            let source_index = representative_index(index, selected, self.lines.len());
            let line = self.lines.get(source_index);
            let start = world_point(line.start);
            let end = world_point(line.end);
            let projected_start = transform.project(start);
            let projected_end = transform.project(end);
            let Some(width) = effective_width(base_width, line.width, self.width_reference())
            else {
                continue;
            };
            let color = resolved_lit_color(
                line.color,
                transform.world_palette_position(start, end),
                dark,
                self.background,
                transform.rod_light(start, end),
                transform.normalized_midpoint_depth(start, end),
            );
            projected.push(ProjectedPrimitive3d::Line {
                start: Point::new(projected_start.x as f32, projected_start.y as f32),
                end: Point::new(projected_end.x as f32, projected_end.y as f32),
                width,
                color,
                depth: (projected_start.depth + projected_end.depth) * 0.5,
                source_position: spatial_line_source_position(source_index, &self.polygons),
            });
        }
        for (index, polygon) in self.polygons.iter().enumerate() {
            let Some(first) = polygon.vertices.first().copied() else {
                continue;
            };
            let mut vertices = Vec::new();
            vertices
                .try_reserve_exact(polygon.vertices.len())
                .map_err(|_| SpatialSceneError::ResourceExhausted {
                    requested: polygon.vertices.len(),
                })?;
            let mut depth = 0.0;
            for &(x, y, z) in &polygon.vertices {
                let point = transform.project(WorldPoint3d::new(x, y, z));
                vertices.push(Point::new(point.x as f32, point.y as f32));
                depth += point.depth;
            }
            let center = polygon_center(&polygon.vertices)
                .unwrap_or_else(|| WorldPoint3d::new(first.0, first.1, first.2));
            let color = resolved_lit_color(
                polygon.color,
                transform.world_palette_position(center, center),
                dark,
                self.background,
                transform.surface_light(
                    polygon
                        .vertices
                        .iter()
                        .map(|&(x, y, z)| WorldPoint3d::new(x, y, z)),
                ),
                transform.normalized_view_depth(depth / polygon.vertices.len().max(1) as f64),
            );
            projected.push(ProjectedPrimitive3d::Polygon {
                vertices,
                color,
                depth: depth / polygon.vertices.len().max(1) as f64,
                source_position: spatial_polygon_source_position(index, polygon),
            });
        }
        projected.sort_by(|left, right| {
            left.depth()
                .total_cmp(&right.depth())
                .then_with(|| left.source_position().cmp(&right.source_position()))
        });
        Ok(projected)
    }

    /// Projects the complete retained scene at the orientation captured when
    /// SVG export starts. Depth order is fixed before the async file/download
    /// portion begins.
    pub(super) fn project_for_svg(
        &self,
        orbit: Orbit3d,
        palette: Palette,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Scene2d, SpatialSceneError> {
        let transform = ViewTransform3d::new(
            self.bounds.into_view_bounds(),
            ViewportSize::new(1000.0, 1000.0),
            orbit,
        );
        let mut primitives = Vec::new();
        primitives
            .try_reserve(self.lines.len().saturating_add(self.polygons.len()))
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: self.lines.len().saturating_add(self.polygons.len()),
            })?;
        for index in 0..self.lines.len() {
            if is_cancelled() {
                return Err(SpatialSceneError::Cancelled);
            }
            let line = self.lines.get(index);
            let world_start = world_point(line.start);
            let world_end = world_point(line.end);
            let start = transform.project(world_start);
            let end = transform.project(world_end);
            primitives.push((
                (start.depth + end.depth) * 0.5,
                spatial_line_source_position(index, &self.polygons),
                Primitive2d::Line(StyledLine2d {
                    line: Line2d((start.x, -start.y), (end.x, -end.y)),
                    width: normalized_turtle_3d_width(
                        f64::from(line.width),
                        self.width_reference(),
                    ),
                    color: resolved_lit_stroke(
                        line.color,
                        transform.world_palette_position(world_start, world_end),
                        palette,
                        self.background,
                        transform.rod_light(world_start, world_end),
                        transform.normalized_midpoint_depth(world_start, world_end),
                    ),
                }),
            ));
        }
        for (index, polygon) in self.polygons.iter().enumerate() {
            if is_cancelled() {
                return Err(SpatialSceneError::Cancelled);
            }
            let mut vertices = Vec::new();
            vertices
                .try_reserve_exact(polygon.vertices.len())
                .map_err(|_| SpatialSceneError::ResourceExhausted {
                    requested: polygon.vertices.len(),
                })?;
            let mut depth = 0.0;
            for &(x, y, z) in &polygon.vertices {
                let point = transform.project(WorldPoint3d::new(x, y, z));
                vertices.push((point.x, -point.y));
                depth += point.depth;
            }
            let mean_depth = depth / polygon.vertices.len().max(1) as f64;
            let center = polygon_center(&polygon.vertices).unwrap_or_else(|| {
                let &(x, y, z) = polygon.vertices.first().unwrap_or(&(0.0, 0.0, 0.0));
                WorldPoint3d::new(x, y, z)
            });
            primitives.push((
                mean_depth,
                spatial_polygon_source_position(index, polygon),
                Primitive2d::Polygon(Polygon2d {
                    vertices,
                    color: resolved_lit_stroke(
                        polygon.color,
                        transform.world_palette_position(center, center),
                        palette,
                        self.background,
                        transform.surface_light(
                            polygon
                                .vertices
                                .iter()
                                .map(|&(x, y, z)| WorldPoint3d::new(x, y, z)),
                        ),
                        transform.normalized_view_depth(mean_depth),
                    ),
                }),
            ));
        }
        primitives.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        Ok(Scene2d {
            primitives: primitives
                .into_iter()
                .map(|(_, _, primitive)| primitive)
                .collect(),
            background: self.background,
        })
    }

    /// Projects and encodes the complete retained scene in bounded browser
    /// work quanta. Both the projection/measurement pass and the SVG-writing
    /// pass yield, so a cancel message can run even for very large 3D curves.
    #[cfg(target_arch = "wasm32")]
    pub(super) async fn encode_svg_yielding(
        &self,
        orbit: Orbit3d,
        palette: Palette,
        line_width_scale: f32,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<String, SpatialSceneError> {
        let transform = ViewTransform3d::new(
            self.bounds.into_view_bounds(),
            ViewportSize::new(1000.0, 1000.0),
            orbit,
        );
        let mut metrics = SpatialSvgMetrics::default();
        let primitive_count = self.lines.len().checked_add(self.polygons.len()).ok_or(
            SpatialSceneError::ResourceExhausted {
                requested: usize::MAX,
            },
        )?;
        let mut order = Vec::new();
        order.try_reserve_exact(primitive_count).map_err(|_| {
            SpatialSceneError::ResourceExhausted {
                requested: primitive_count,
            }
        })?;

        for start in (0..self.lines.len()).step_by(SVG_WORK_PER_BATCH) {
            if is_cancelled() {
                return Err(SpatialSceneError::Cancelled);
            }
            let end = start
                .saturating_add(SVG_WORK_PER_BATCH)
                .min(self.lines.len());
            for index in start..end {
                let line = self.lines.get(index);
                let projected_start = transform.project(world_point(line.start));
                let projected_end = transform.project(world_point(line.end));
                metrics.observe(
                    (projected_start.x, projected_start.y),
                    (projected_end.x, projected_end.y),
                );
                order.push(SpatialDepthPrimitive {
                    depth: (projected_start.depth + projected_end.depth) * 0.5,
                    source_position: spatial_line_source_position(index, &self.polygons),
                    primitive: SpatialPrimitiveIndex::Line(index),
                });
            }
            yield_to_browser().await;
        }

        let mut work_since_yield = 0usize;
        for (index, polygon) in self.polygons.iter().enumerate() {
            let Some(&first) = polygon.vertices.first() else {
                order.push(SpatialDepthPrimitive {
                    depth: 0.0,
                    source_position: spatial_polygon_source_position(index, polygon),
                    primitive: SpatialPrimitiveIndex::Polygon(index),
                });
                work_since_yield = work_since_yield.saturating_add(1);
                if work_since_yield >= SVG_WORK_PER_BATCH {
                    if is_cancelled() {
                        return Err(SpatialSceneError::Cancelled);
                    }
                    yield_to_browser().await;
                    work_since_yield = 0;
                }
                continue;
            };
            let first_projected = transform.project(WorldPoint3d::new(first.0, first.1, first.2));
            let first = (first_projected.x, first_projected.y);
            let mut depth = first_projected.depth;
            let mut previous = first;
            for &(x, y, z) in polygon.vertices.iter().skip(1) {
                let point = transform.project(WorldPoint3d::new(x, y, z));
                depth += point.depth;
                let point = (point.x, point.y);
                metrics.observe(previous, point);
                previous = point;
                work_since_yield = work_since_yield.saturating_add(1);
                if work_since_yield >= SVG_WORK_PER_BATCH {
                    if is_cancelled() {
                        return Err(SpatialSceneError::Cancelled);
                    }
                    yield_to_browser().await;
                    work_since_yield = 0;
                }
            }
            metrics.observe(previous, first);
            work_since_yield = work_since_yield.saturating_add(1);
            order.push(SpatialDepthPrimitive {
                depth: depth / polygon.vertices.len() as f64,
                source_position: spatial_polygon_source_position(index, polygon),
                primitive: SpatialPrimitiveIndex::Polygon(index),
            });
            if work_since_yield >= SVG_WORK_PER_BATCH {
                if is_cancelled() {
                    return Err(SpatialSceneError::Cancelled);
                }
                yield_to_browser().await;
                work_since_yield = 0;
            }
        }
        if work_since_yield > 0 {
            yield_to_browser().await;
        }
        if is_cancelled() {
            return Err(SpatialSceneError::Cancelled);
        }
        sort_spatial_depth_primitives_yielding(&mut order, is_cancelled).await?;

        let (view, drawing_extent) = metrics.view();
        let stroke_scale = if line_width_scale.is_finite() && line_width_scale > 0.0 {
            f64::from(line_width_scale)
        } else {
            1.0
        };
        let stroke_width = adaptive_spatial_scene_svg_stroke_width(
            metrics.width_estimator.total_line_length(),
            metrics.width_estimator.line_count(),
            drawing_extent,
            view.2.max(view.3),
        ) * stroke_scale;
        let themed_background = match palette {
            Palette::Light => "#f2f5f9",
            Palette::Dark => "#0f1420",
        };
        let background = self
            .background
            .map(rgb_hex)
            .unwrap_or_else(|| themed_background.to_owned());
        let mut output = String::new();
        output
            .try_reserve(512)
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: self.lines.len().saturating_add(self.polygons.len()),
            })?;
        let _ = writeln!(
            output,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{background}\"/>",
            view.0, view.1, view.2, view.3, view.0, view.1, view.2, view.3
        );

        work_since_yield = 0;
        for entry in order {
            if is_cancelled() {
                return Err(SpatialSceneError::Cancelled);
            }
            match entry.primitive {
                SpatialPrimitiveIndex::Line(index) => {
                    output
                        .try_reserve(320)
                        .map_err(|_| SpatialSceneError::ResourceExhausted {
                            requested: self.lines.len(),
                        })?;
                    let line = self.lines.get(index);
                    let world_start = world_point(line.start);
                    let world_end = world_point(line.end);
                    let projected_start = transform.project(world_start);
                    let projected_end = transform.project(world_end);
                    let width = stroke_width
                        * normalized_turtle_3d_width(f64::from(line.width), self.width_reference());
                    let stroke = rgb_hex(rgb8(resolved_lit_rgb(
                        line.color,
                        transform.world_palette_position(world_start, world_end),
                        palette,
                        self.background,
                        transform.rod_light(world_start, world_end),
                        transform.normalized_midpoint_depth(world_start, world_end),
                    )));
                    let _ = writeln!(
                        output,
                        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>",
                        projected_start.x, projected_start.y, projected_end.x, projected_end.y,
                    );
                    work_since_yield = work_since_yield.saturating_add(1);
                }
                SpatialPrimitiveIndex::Polygon(index) => {
                    let polygon = &self.polygons[index];
                    let requested = polygon
                        .vertices
                        .len()
                        .checked_mul(64)
                        .and_then(|size| size.checked_add(320))
                        .ok_or(SpatialSceneError::ResourceExhausted {
                            requested: polygon.vertices.len(),
                        })?;
                    output.try_reserve(requested).map_err(|_| {
                        SpatialSceneError::ResourceExhausted {
                            requested: polygon.vertices.len(),
                        }
                    })?;
                    let center = polygon_center(&polygon.vertices)
                        .unwrap_or_else(|| WorldPoint3d::new(0.0, 0.0, 0.0));
                    let fill = rgb_hex(rgb8(resolved_lit_rgb(
                        polygon.color,
                        transform.world_palette_position(center, center),
                        palette,
                        self.background,
                        transform.surface_light(
                            polygon
                                .vertices
                                .iter()
                                .map(|&(x, y, z)| WorldPoint3d::new(x, y, z)),
                        ),
                        transform.normalized_view_depth(entry.depth),
                    )));
                    output.push_str("<polygon points=\"");
                    for &(x, y, z) in &polygon.vertices {
                        let point = transform.project(WorldPoint3d::new(x, y, z));
                        let _ = write!(output, "{},{} ", point.x, point.y);
                        work_since_yield = work_since_yield.saturating_add(1);
                        if work_since_yield >= SVG_WORK_PER_BATCH {
                            if is_cancelled() {
                                return Err(SpatialSceneError::Cancelled);
                            }
                            yield_to_browser().await;
                            work_since_yield = 0;
                        }
                    }
                    let _ = writeln!(output, "\" fill=\"{fill}\" stroke=\"none\"/>");
                    if polygon.vertices.is_empty() {
                        work_since_yield = work_since_yield.saturating_add(1);
                    }
                }
            }
            if work_since_yield >= SVG_WORK_PER_BATCH {
                if is_cancelled() {
                    return Err(SpatialSceneError::Cancelled);
                }
                yield_to_browser().await;
                work_since_yield = 0;
            }
        }
        if work_since_yield > 0 {
            yield_to_browser().await;
        }
        if is_cancelled() {
            return Err(SpatialSceneError::Cancelled);
        }
        output.push_str("</svg>\n");
        Ok(output)
    }
}

impl RenderBounds3d {
    fn into_view_bounds(self) -> ViewBounds3d {
        self.into()
    }

    #[cfg(target_arch = "wasm32")]
    fn is_finite_ordered(self) -> bool {
        [
            self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z,
        ]
        .into_iter()
        .all(f32::is_finite)
            && self.min_x <= self.max_x
            && self.min_y <= self.max_y
            && self.min_z <= self.max_z
    }

    #[cfg(target_arch = "wasm32")]
    fn contains(self, [x, y, z]: [f32; 3]) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }
}

#[derive(Debug)]
pub(super) enum ProjectedPrimitive3d {
    Line {
        start: Point,
        end: Point,
        width: f32,
        color: Color,
        depth: f64,
        source_position: usize,
    },
    Polygon {
        vertices: Vec<Point>,
        color: Color,
        depth: f64,
        source_position: usize,
    },
}

impl ProjectedPrimitive3d {
    fn depth(&self) -> f64 {
        match self {
            Self::Line { depth, .. } | Self::Polygon { depth, .. } => *depth,
        }
    }

    fn source_position(&self) -> usize {
        match self {
            Self::Line {
                source_position, ..
            }
            | Self::Polygon {
                source_position, ..
            } => *source_position,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy)]
enum SpatialPrimitiveIndex {
    Line(usize),
    Polygon(usize),
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy)]
struct SpatialDepthPrimitive {
    depth: f64,
    source_position: usize,
    primitive: SpatialPrimitiveIndex,
}

#[cfg(target_arch = "wasm32")]
fn compare_spatial_depth_primitives(
    left: &SpatialDepthPrimitive,
    right: &SpatialDepthPrimitive,
) -> std::cmp::Ordering {
    left.depth
        .total_cmp(&right.depth)
        .then_with(|| left.source_position.cmp(&right.source_position))
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
struct SpatialSvgMetrics {
    width_estimator: StrokeWidthEstimator,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

#[cfg(target_arch = "wasm32")]
impl Default for SpatialSvgMetrics {
    fn default() -> Self {
        Self {
            width_estimator: StrokeWidthEstimator::default(),
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl SpatialSvgMetrics {
    fn observe(&mut self, start: (f64, f64), end: (f64, f64)) {
        self.width_estimator.observe(Line2d(start, end));
        self.min_x = self.min_x.min(start.0).min(end.0);
        self.max_x = self.max_x.max(start.0).max(end.0);
        self.min_y = self.min_y.min(start.1).min(end.1);
        self.max_y = self.max_y.max(start.1).max(end.1);
    }

    fn view(&self) -> ((f64, f64, f64, f64), (f64, f64)) {
        if ![self.min_x, self.max_x, self.min_y, self.max_y]
            .into_iter()
            .all(f64::is_finite)
        {
            return ((0.0, 0.0, 1000.0, 700.0), (0.0, 0.0));
        }
        let drawing_width = (self.max_x - self.min_x).max(0.0);
        let drawing_height = (self.max_y - self.min_y).max(0.0);
        let width = drawing_width.max(0.1);
        let height = drawing_height.max(0.1);
        let margin = width.max(height) * 0.04;
        (
            (
                self.min_x - margin,
                self.min_y - margin,
                width + margin * 2.0,
                height + margin * 2.0,
            ),
            (drawing_width, drawing_height),
        )
    }
}

#[cfg(target_arch = "wasm32")]
fn rgb_hex([red, green, blue]: [u8; 3]) -> String {
    format!("#{red:02x}{green:02x}{blue:02x}")
}

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

/// Cooperatively sorts bounded runs, then merges them while continuing to
/// service cancellation between browser tasks.
#[cfg(target_arch = "wasm32")]
async fn sort_spatial_depth_primitives_yielding(
    order: &mut Vec<SpatialDepthPrimitive>,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<(), SpatialSceneError> {
    for chunk in order.chunks_mut(SVG_WORK_PER_BATCH) {
        if is_cancelled() {
            return Err(SpatialSceneError::Cancelled);
        }
        chunk.sort_unstable_by(compare_spatial_depth_primitives);
        yield_to_browser().await;
    }
    if order.len() <= SVG_WORK_PER_BATCH {
        return if is_cancelled() {
            Err(SpatialSceneError::Cancelled)
        } else {
            Ok(())
        };
    }

    let item_count = order.len();
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(item_count)
        .map_err(|_| SpatialSceneError::ResourceExhausted {
            requested: item_count,
        })?;
    let mut run_length = SVG_WORK_PER_BATCH;
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
                        && compare_spatial_depth_primitives(&order[left], &order[right])
                            != std::cmp::Ordering::Greater);
                if take_left {
                    scratch.push(order[left]);
                    left += 1;
                } else {
                    scratch.push(order[right]);
                    right += 1;
                }
                work += 1;
                if work >= SVG_WORK_PER_BATCH {
                    work = 0;
                    if is_cancelled() {
                        return Err(SpatialSceneError::Cancelled);
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
        Err(SpatialSceneError::Cancelled)
    } else {
        Ok(())
    }
}

fn representative_index(index: usize, selected: usize, total: usize) -> usize {
    if selected == total {
        index
    } else if selected <= 1 {
        total / 2
    } else {
        ((index as u128 * total.saturating_sub(1) as u128) / selected.saturating_sub(1) as u128)
            as usize
    }
}

fn spatial_line_source_position(index: usize, polygons: &[SpatialPolygon3d]) -> usize {
    let polygons_before = polygons.partition_point(|polygon| polygon.lines_before <= index);
    index.saturating_add(polygons_before)
}

fn spatial_polygon_source_position(index: usize, polygon: &SpatialPolygon3d) -> usize {
    polygon.lines_before.saturating_add(index)
}

fn checked_triangle_count(
    vertex_counts: impl IntoIterator<Item = usize>,
) -> Result<usize, SpatialSceneError> {
    vertex_counts
        .into_iter()
        .try_fold(0usize, |total, vertices| {
            total.checked_add(vertices.saturating_sub(2)).ok_or(
                SpatialSceneError::ResourceExhausted {
                    requested: usize::MAX,
                },
            )
        })
}

fn checked_refinement_total(
    line_count: usize,
    polygon_triangle_count: usize,
) -> Result<usize, SpatialSceneError> {
    line_count
        .checked_add(polygon_triangle_count)
        .ok_or(SpatialSceneError::ResourceExhausted {
            requested: usize::MAX,
        })
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_spatial_polygon_order(
    polygons: &[SpatialPolygon3d],
    line_count: usize,
) -> Result<(), SpatialSceneError> {
    let mut previous = 0usize;
    for (index, polygon) in polygons.iter().enumerate() {
        if polygon.lines_before > line_count {
            return Err(SpatialSceneError::InvalidGeometry(
                "polygon source order exceeds the transferred line count",
            ));
        }
        if index > 0 && polygon.lines_before < previous {
            return Err(SpatialSceneError::InvalidGeometry(
                "polygon source order is not monotonic",
            ));
        }
        previous = polygon.lines_before;
    }
    Ok(())
}

fn checked_point((x, y, z): (f64, f64, f64)) -> Result<[f32; 3], SpatialSceneError> {
    let point = [x as f32, y as f32, z as f32];
    if [x, y, z].into_iter().all(f64::is_finite) && point.into_iter().all(f32::is_finite) {
        Ok(point)
    } else {
        Err(SpatialSceneError::InvalidGeometry(
            "coordinate is outside the finite f32 range",
        ))
    }
}

fn world_point(point: [f32; 3]) -> WorldPoint3d {
    WorldPoint3d::new(
        f64::from(point[0]),
        f64::from(point[1]),
        f64::from(point[2]),
    )
}

fn viewport_size(size: Size) -> ViewportSize {
    ViewportSize::new(
        f64::from(size.width.max(1.0)),
        f64::from(size.height.max(1.0)),
    )
}

fn effective_width(base: f32, source_width: f32, width_reference: f64) -> Option<f32> {
    let multiplier = normalized_turtle_3d_width(f64::from(source_width), width_reference) as f32;
    let width = base * multiplier;
    (width.is_finite() && width > 0.0).then_some(width)
}

fn spatial_base_color(color: StrokeColor, position: f64, palette: Palette) -> [f32; 3] {
    if let Some([red, green, blue]) = turtle_stroke_rgb(color, palette) {
        return [
            f32::from(red) / 255.0,
            f32::from(green) / 255.0,
            f32::from(blue) / 255.0,
        ];
    }
    spatial_theme_default_color_at(position, palette)
}

fn palette_for_dark(dark: bool) -> Palette {
    if dark { Palette::Dark } else { Palette::Light }
}

fn spatial_background_rgb(source: Option<[u8; 3]>, palette: Palette) -> [f32; 3] {
    let [red, green, blue] = source.unwrap_or(match palette {
        Palette::Light => [242, 245, 249],
        Palette::Dark => [15, 20, 32],
    });
    [
        f32::from(red) / 255.0,
        f32::from(green) / 255.0,
        f32::from(blue) / 255.0,
    ]
}

fn resolved_base_color(color: StrokeColor, position: f64, dark: bool) -> Color {
    let [red, green, blue] = spatial_base_color(color, position, palette_for_dark(dark));
    Color::from_rgb(red, green, blue)
}

fn resolved_lit_rgb(
    color: StrokeColor,
    position: f64,
    palette: Palette,
    background: Option<[u8; 3]>,
    light: f64,
    near_depth: f64,
) -> [f32; 3] {
    spatial_lit_color_bounded(
        spatial_base_color(color, position, palette),
        spatial_background_rgb(background, palette),
        light,
        near_depth,
    )
}

fn resolved_lit_color(
    color: StrokeColor,
    position: f64,
    dark: bool,
    background: Option<[u8; 3]>,
    light: f64,
    near_depth: f64,
) -> Color {
    let [red, green, blue] = resolved_lit_rgb(
        color,
        position,
        palette_for_dark(dark),
        background,
        light,
        near_depth,
    );
    Color::from_rgb(red, green, blue)
}

fn resolved_lit_stroke(
    color: StrokeColor,
    position: f64,
    palette: Palette,
    background: Option<[u8; 3]>,
    light: f64,
    near_depth: f64,
) -> StrokeColor {
    StrokeColor::Rgb(rgb8(resolved_lit_rgb(
        color, position, palette, background, light, near_depth,
    )))
}

fn rgb8(color: [f32; 3]) -> [u8; 3] {
    color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn polygon_center(vertices: &[(f64, f64, f64)]) -> Option<WorldPoint3d> {
    let (&first, remaining) = vertices.split_first()?;
    let mut center = WorldPoint3d::new(first.0, first.1, first.2);
    for (index, &(x, y, z)) in remaining.iter().enumerate() {
        let weight = 1.0 / (index + 2) as f64;
        center.x += (x - center.x) * weight;
        center.y += (y - center.y) * weight;
        center.z += (z - center.z) * weight;
    }
    Some(center)
}

fn display_background(source: Option<[u8; 3]>, dark: bool) -> Color {
    source
        .map(|[red, green, blue]| Color::from_rgb8(red, green, blue))
        .unwrap_or(if dark {
            DARK_CANVAS_BACKGROUND
        } else {
            CANVAS_BACKGROUND
        })
}

#[derive(Debug, Clone)]
pub(super) struct SpatialProgram {
    scene: Arc<SpatialScene>,
    orbit: Orbit3d,
    dark: bool,
    moving: bool,
    line_width_scale: f32,
}

impl SpatialProgram {
    pub(super) fn new(
        scene: Arc<SpatialScene>,
        orbit: Orbit3d,
        dark: bool,
        moving: bool,
        line_width_scale: f32,
    ) -> Self {
        Self {
            scene,
            orbit,
            dark,
            moving,
            line_width_scale,
        }
    }
}

impl<Message> shader::Program<Message> for SpatialProgram {
    type State = ();
    type Primitive = SpatialPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: iced::mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        SpatialPrimitive {
            scene: Arc::clone(&self.scene),
            orbit: self.orbit,
            dark: self.dark,
            moving: self.moving,
            line_width_scale: self.line_width_scale,
            size: bounds.size(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SpatialPrimitive {
    scene: Arc<SpatialScene>,
    orbit: Orbit3d,
    dark: bool,
    moving: bool,
    line_width_scale: f32,
    size: Size,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SpatialUniform {
    viewport_and_scale: [f32; 4],
    center_and_radius: [f32; 4],
    quaternion: [f32; 4],
    background: [f32; 4],
    base_width: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuSpatialLine {
    start: [f32; 3],
    end: [f32; 3],
    width: f32,
    _padding: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuSpatialVertex {
    position: [f32; 3],
    _padding: f32,
    color: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedKey {
    scene_id: u64,
    dark: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct SpatialExactBuild {
    next_line: usize,
    polygon: PolygonCursor,
    completed_triangles: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct PolygonCursor {
    polygon_index: usize,
    edge_index: usize,
}

struct DepthTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Copy)]
struct PhysicalRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

struct SpatialBufferBatch {
    buffer: wgpu::Buffer,
    capacity: usize,
    count: u32,
}

pub(super) struct SpatialPipeline {
    background_pipeline: wgpu::RenderPipeline,
    polygon_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    line_batches: Vec<SpatialBufferBatch>,
    polygon_batches: Vec<SpatialBufferBatch>,
    building_line_batches: Vec<SpatialBufferBatch>,
    building_polygon_batches: Vec<SpatialBufferBatch>,
    exact_build: Option<SpatialExactBuild>,
    prepared: Option<PreparedKey>,
    depth: Option<DepthTarget>,
    target: PhysicalRect,
}

impl shader::Pipeline for SpatialPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("braken.spatial.shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "spatial_scene.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("braken.spatial.bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("braken.spatial.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let color_target = || {
            Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("braken.spatial.background_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_background"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_background"),
                targets: &[color_target()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            // Every pipeline bound in one render pass must declare the same
            // depth attachment format. The background must participate in
            // that contract without populating depth ahead of the geometry.
            depth_stencil: Some(spatial_depth_state(false, wgpu::CompareFunction::Always)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let polygon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("braken.spatial.polygon_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_polygon"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuSpatialVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32,
                        2 => Float32x4,
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_polygon"),
                targets: &[color_target()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(spatial_depth_state(true, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("braken.spatial.line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuSpatialLine>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32,
                        3 => Float32,
                        4 => Float32x4,
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[color_target()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(spatial_depth_state(true, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("braken.spatial.uniform"),
            size: std::mem::size_of::<SpatialUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("braken.spatial.bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        Self {
            background_pipeline,
            polygon_pipeline,
            line_pipeline,
            uniform_buffer,
            uniform_bind_group,
            line_batches: Vec::new(),
            polygon_batches: Vec::new(),
            building_line_batches: Vec::new(),
            building_polygon_batches: Vec::new(),
            exact_build: None,
            prepared: None,
            depth: None,
            target: PhysicalRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        }
    }
}

impl shader::Primitive for SpatialPrimitive {
    type Pipeline = SpatialPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        if self.scene.gpu_refinement_failure().is_some() {
            return;
        }
        let prepared = (|| -> Result<(), SpatialSceneError> {
            let transform = ViewTransform3d::new(
                self.scene.bounds.into_view_bounds(),
                viewport_size(self.size),
                self.orbit,
            );
            let center = self.scene.bounds.into_view_bounds().center();
            let screen_center = transform.viewport_center();
            let background = display_background(self.scene.background, self.dark);
            let quaternion = self.orbit.components();
            let uniform = SpatialUniform {
                viewport_and_scale: [
                    self.size.width.max(1.0),
                    self.size.height.max(1.0),
                    transform.fit_scale() as f32,
                    0.0,
                ],
                center_and_radius: [
                    center.x as f32,
                    center.y as f32,
                    center.z as f32,
                    self.scene.bounds.into_view_bounds().fit_radius() as f32,
                ],
                quaternion: [
                    quaternion[0] as f32,
                    quaternion[1] as f32,
                    quaternion[2] as f32,
                    quaternion[3] as f32,
                ],
                background: [background.r, background.g, background.b, 1.0],
                base_width: [
                    self.scene
                        .base_stroke_width(transform, self.line_width_scale),
                    screen_center.x as f32,
                    screen_center.y as f32,
                    0.0,
                ],
            };
            queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
            pipeline.target = physical_target(bounds, viewport.scale_factor());
            ensure_depth_target(pipeline, device, viewport.physical_size())?;

            let key = PreparedKey {
                scene_id: self.scene.id,
                dark: self.dark,
            };
            let line_chunk_capacity = gpu_chunk_capacity::<GpuSpatialLine>(device)?;
            let polygon_chunk_capacity = gpu_polygon_chunk_capacity(device)?;

            if pipeline.prepared != Some(key) {
                let selected = self
                    .scene
                    .lines
                    .len()
                    .min(MOVING_GPU_PREVIEW_LINES)
                    .min(line_chunk_capacity);
                let lines =
                    gpu_spatial_lines(&self.scene, transform, self.dark, selected, 0..selected)?;
                if lines.is_empty() {
                    pipeline.line_batches.clear();
                } else {
                    upload_spatial_batch(
                        &mut pipeline.line_batches,
                        0,
                        &lines,
                        "braken.spatial.lines.preview",
                        device,
                        queue,
                    )?;
                    pipeline.line_batches.truncate(1);
                }

                let mut polygon_cursor = PolygonCursor::default();
                let (vertices, preview_triangles) = gpu_spatial_polygon_chunk(
                    &self.scene,
                    transform,
                    self.dark,
                    &mut polygon_cursor,
                    polygon_chunk_capacity,
                )?;
                if vertices.is_empty() {
                    pipeline.polygon_batches.clear();
                } else {
                    upload_spatial_batch(
                        &mut pipeline.polygon_batches,
                        0,
                        &vertices,
                        "braken.spatial.polygons.preview",
                        device,
                        queue,
                    )?;
                    pipeline.polygon_batches.truncate(1);
                }

                pipeline.building_line_batches.clear();
                pipeline.building_polygon_batches.clear();
                let preview_is_exact = selected == self.scene.lines.len()
                    && preview_triangles == self.scene.polygon_triangle_count;
                if preview_is_exact {
                    self.scene
                        .refined_work
                        .store(self.scene.refinement_total(), Ordering::Release);
                    pipeline.exact_build = None;
                } else {
                    self.scene.refined_work.store(0, Ordering::Release);
                    pipeline.exact_build = Some(SpatialExactBuild::default());
                }
                pipeline.prepared = Some(key);
                return Ok(());
            }

            if self.scene.refinement_cancelled.load(Ordering::Acquire) {
                pipeline.building_line_batches.clear();
                pipeline.building_polygon_batches.clear();
                pipeline.exact_build = None;
                return Ok(());
            }
            if self.moving {
                return Ok(());
            }
            let Some(build) = pipeline.exact_build.as_mut() else {
                return Ok(());
            };

            if build.next_line < self.scene.lines.len() {
                let end = build
                    .next_line
                    .saturating_add(line_chunk_capacity)
                    .min(self.scene.lines.len());
                let lines = gpu_spatial_lines(
                    &self.scene,
                    transform,
                    self.dark,
                    self.scene.lines.len(),
                    build.next_line..end,
                )?;
                let batch_index = pipeline.building_line_batches.len();
                upload_spatial_batch(
                    &mut pipeline.building_line_batches,
                    batch_index,
                    &lines,
                    "braken.spatial.lines.exact",
                    device,
                    queue,
                )?;
                build.next_line = end;
                let completed_work = build
                    .next_line
                    .checked_add(build.completed_triangles)
                    .ok_or(SpatialSceneError::ResourceExhausted {
                        requested: usize::MAX,
                    })?;
                self.scene
                    .refined_work
                    .store(completed_work, Ordering::Release);
                if build.next_line < self.scene.lines.len() || self.scene.polygon_triangle_count > 0
                {
                    return Ok(());
                }
            }

            if build.completed_triangles < self.scene.polygon_triangle_count {
                let (vertices, completed) = gpu_spatial_polygon_chunk(
                    &self.scene,
                    transform,
                    self.dark,
                    &mut build.polygon,
                    polygon_chunk_capacity,
                )?;
                if vertices.is_empty() || completed == 0 {
                    return Err(SpatialSceneError::ResourceExhausted {
                        requested: self.scene.polygon_triangle_count,
                    });
                }
                let batch_index = pipeline.building_polygon_batches.len();
                upload_spatial_batch(
                    &mut pipeline.building_polygon_batches,
                    batch_index,
                    &vertices,
                    "braken.spatial.polygons.exact",
                    device,
                    queue,
                )?;
                build.completed_triangles = build
                    .completed_triangles
                    .checked_add(completed)
                    .ok_or(SpatialSceneError::ResourceExhausted {
                        requested: usize::MAX,
                    })?;
                let completed_work = build
                    .next_line
                    .checked_add(build.completed_triangles)
                    .ok_or(SpatialSceneError::ResourceExhausted {
                        requested: usize::MAX,
                    })?;
                self.scene
                    .refined_work
                    .store(completed_work, Ordering::Release);
                if build.completed_triangles < self.scene.polygon_triangle_count {
                    return Ok(());
                }
            }

            std::mem::swap(
                &mut pipeline.line_batches,
                &mut pipeline.building_line_batches,
            );
            std::mem::swap(
                &mut pipeline.polygon_batches,
                &mut pipeline.building_polygon_batches,
            );
            pipeline.building_line_batches.clear();
            pipeline.building_polygon_batches.clear();
            pipeline.exact_build = None;
            self.scene
                .refined_work
                .store(self.scene.refinement_total(), Ordering::Release);
            Ok(())
        })();
        if let Err(error) = prepared {
            pipeline.line_batches.clear();
            pipeline.polygon_batches.clear();
            pipeline.building_line_batches.clear();
            pipeline.building_polygon_batches.clear();
            pipeline.exact_build = None;
            pipeline.prepared = None;
            self.scene.record_gpu_refinement_failure(error);
        }
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        if self.scene.gpu_refinement_failure().is_some()
            || clip_bounds.width == 0
            || clip_bounds.height == 0
        {
            return;
        }
        let Some(depth) = &pipeline.depth else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("braken.spatial.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
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
        pass.set_bind_group(0, &pipeline.uniform_bind_group, &[]);
        pass.set_pipeline(&pipeline.background_pipeline);
        pass.draw(0..3, 0..1);
        if !pipeline.polygon_batches.is_empty() {
            pass.set_pipeline(&pipeline.polygon_pipeline);
            for batch in &pipeline.polygon_batches {
                pass.set_vertex_buffer(0, batch.buffer.slice(..));
                pass.draw(0..batch.count, 0..1);
            }
        }
        if !pipeline.line_batches.is_empty() {
            pass.set_pipeline(&pipeline.line_pipeline);
            for batch in &pipeline.line_batches {
                pass.set_vertex_buffer(0, batch.buffer.slice(..));
                pass.draw(0..6, 0..batch.count);
            }
        }
    }
}

fn gpu_spatial_lines(
    scene: &SpatialScene,
    transform: ViewTransform3d,
    dark: bool,
    selected: usize,
    range: std::ops::Range<usize>,
) -> Result<Vec<GpuSpatialLine>, SpatialSceneError> {
    if range.end > selected || selected > scene.lines.len() {
        return Err(SpatialSceneError::ResourceExhausted {
            requested: range.end,
        });
    }
    let mut lines = Vec::new();
    lines
        .try_reserve_exact(range.len())
        .map_err(|_| SpatialSceneError::ResourceExhausted {
            requested: range.len(),
        })?;
    for index in range {
        let line = scene.source_line(index, selected);
        let start = world_point(line.start);
        let end = world_point(line.end);
        let color = resolved_base_color(
            line.color,
            transform.world_palette_position(start, end),
            dark,
        );
        lines.push(GpuSpatialLine {
            start: line.start,
            end: line.end,
            width: normalized_turtle_3d_width(f64::from(line.width), scene.width_reference())
                as f32,
            _padding: 0.0,
            color: [color.r, color.g, color.b, color.a],
        });
    }
    Ok(lines)
}

fn gpu_spatial_polygon_chunk(
    scene: &SpatialScene,
    transform: ViewTransform3d,
    dark: bool,
    cursor: &mut PolygonCursor,
    max_vertices: usize,
) -> Result<(Vec<GpuSpatialVertex>, usize), SpatialSceneError> {
    if max_vertices < 3 {
        return Err(SpatialSceneError::ResourceExhausted { requested: 3 });
    }
    if scene.polygon_triangle_count == 0 || cursor.polygon_index >= scene.polygons.len() {
        return Ok((Vec::new(), 0));
    }
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(max_vertices)
        .map_err(|_| SpatialSceneError::ResourceExhausted {
            requested: max_vertices,
        })?;
    let mut completed = 0usize;
    while cursor.polygon_index < scene.polygons.len()
        && vertices.len().saturating_add(3) <= max_vertices
    {
        let polygon = &scene.polygons[cursor.polygon_index];
        if polygon.vertices.len() < 3 {
            cursor.polygon_index += 1;
            cursor.edge_index = 1;
            continue;
        }
        cursor.edge_index = cursor.edge_index.max(1);
        if cursor.edge_index >= polygon.vertices.len() - 1 {
            cursor.polygon_index += 1;
            cursor.edge_index = 1;
            continue;
        }
        let first = polygon.vertices[0];
        let center = polygon_center(&polygon.vertices)
            .unwrap_or_else(|| WorldPoint3d::new(first.0, first.1, first.2));
        let color = resolved_base_color(
            polygon.color,
            transform.world_palette_position(center, center),
            dark,
        );
        let color = [color.r, color.g, color.b, color.a];
        for &(x, y, z) in &[
            first,
            polygon.vertices[cursor.edge_index],
            polygon.vertices[cursor.edge_index + 1],
        ] {
            vertices.push(GpuSpatialVertex {
                position: [x as f32, y as f32, z as f32],
                _padding: 0.0,
                color,
            });
        }
        cursor.edge_index += 1;
        completed += 1;
    }
    Ok((vertices, completed))
}

fn gpu_chunk_capacity<T>(device: &wgpu::Device) -> Result<usize, SpatialSceneError> {
    let item_size = std::mem::size_of::<T>() as u64;
    checked_gpu_chunk_capacity(item_size, device.limits().max_buffer_size)
}

fn gpu_polygon_chunk_capacity(device: &wgpu::Device) -> Result<usize, SpatialSceneError> {
    let capacity = gpu_chunk_capacity::<GpuSpatialVertex>(device)? / 3 * 3;
    if capacity < 3 {
        Err(SpatialSceneError::ResourceExhausted { requested: 3 })
    } else {
        Ok(capacity)
    }
}

fn checked_gpu_chunk_capacity(
    item_size: u64,
    max_buffer_size: u64,
) -> Result<usize, SpatialSceneError> {
    if item_size == 0 {
        return Err(SpatialSceneError::ResourceExhausted { requested: 1 });
    }
    let device_capacity = max_buffer_size / item_size;
    let bounded = device_capacity
        .min(u32::MAX as u64)
        .min(GPU_UPLOAD_CHUNK_ITEMS as u64);
    if bounded == 0 {
        return Err(SpatialSceneError::ResourceExhausted { requested: 1 });
    }
    usize::try_from(bounded).map_err(|_| SpatialSceneError::ResourceExhausted { requested: 1 })
}

fn upload_spatial_batch<T: Pod>(
    batches: &mut Vec<SpatialBufferBatch>,
    index: usize,
    values: &[T],
    label: &'static str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(), SpatialSceneError> {
    if values.is_empty() {
        return Ok(());
    }
    let required = values.len();
    let (byte_size, draw_count) =
        checked_gpu_batch_layout::<T>(required, device.limits().max_buffer_size)?;
    if index > batches.len() {
        return Err(SpatialSceneError::ResourceExhausted {
            requested: required,
        });
    }
    if index == batches.len() {
        batches
            .try_reserve(1)
            .map_err(|_| SpatialSceneError::ResourceExhausted {
                requested: batches.len().saturating_add(1),
            })?;
        batches.push(SpatialBufferBatch {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: byte_size.max(wgpu::COPY_BUFFER_ALIGNMENT),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity: required,
            count: 0,
        });
    } else if required > batches[index].capacity {
        batches[index].buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size.max(wgpu::COPY_BUFFER_ALIGNMENT),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        batches[index].capacity = required;
    }
    let batch = &mut batches[index];
    queue.write_buffer(&batch.buffer, 0, bytemuck::cast_slice(values));
    batch.count = draw_count;
    Ok(())
}

fn checked_gpu_batch_layout<T>(
    required: usize,
    max_buffer_size: u64,
) -> Result<(u64, u32), SpatialSceneError> {
    let required_u64 =
        u64::try_from(required).map_err(|_| SpatialSceneError::ResourceExhausted {
            requested: required,
        })?;
    let byte_size = required_u64
        .checked_mul(std::mem::size_of::<T>() as u64)
        .filter(|size| *size <= max_buffer_size)
        .ok_or(SpatialSceneError::ResourceExhausted {
            requested: required,
        })?;
    let draw_count = u32::try_from(required).map_err(|_| SpatialSceneError::ResourceExhausted {
        requested: required,
    })?;
    Ok((byte_size, draw_count))
}

fn ensure_depth_target(
    pipeline: &mut SpatialPipeline,
    device: &wgpu::Device,
    size: Size<u32>,
) -> Result<(), SpatialSceneError> {
    let width = size.width.max(1);
    let height = size.height.max(1);
    let maximum = device.limits().max_texture_dimension_2d;
    if width > maximum || height > maximum {
        return Err(SpatialSceneError::ResourceExhausted {
            requested: usize::try_from(u64::from(width) * u64::from(height)).unwrap_or(usize::MAX),
        });
    }
    if pipeline
        .depth
        .as_ref()
        .is_some_and(|depth| depth.width == width && depth.height == height)
    {
        return Ok(());
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("braken.spatial.depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    pipeline.depth = Some(DepthTarget {
        _texture: texture,
        view,
        width,
        height,
    });
    Ok(())
}

fn physical_target(bounds: &Rectangle, scale_factor: f32) -> PhysicalRect {
    let scale = scale_factor.max(f32::EPSILON);
    PhysicalRect {
        x: bounds.x * scale,
        y: bounds.y * scale,
        width: (bounds.width * scale).max(1.0),
        height: (bounds.height * scale).max(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{
        SPATIAL_LIGHT_DIRECTION, SPATIAL_ROD_AMBIENT_LIGHT, SPATIAL_SURFACE_AMBIENT_LIGHT,
    };
    use braken_viz::targets::{SPATIAL_FAR_FOG, SPATIAL_NEAR_FOG};
    use braken_viz::{Line3d, StyledLine3d};

    fn styled_line(index: f64) -> StyledLine3d {
        StyledLine3d {
            line: Line3d((index, 0.0, -index), (index + 1.0, 1.0, index)),
            width: 1.0,
            color: StrokeColor::ThemeDefault,
        }
    }

    fn line(index: f64) -> Primitive3d {
        Primitive3d::Line(styled_line(index))
    }

    fn polygon() -> Polygon3d {
        Polygon3d {
            vertices: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            color: StrokeColor::ThemeDefault,
        }
    }

    #[test]
    fn representative_selection_keeps_endpoints() {
        assert_eq!(representative_index(0, 4, 10), 0);
        assert_eq!(representative_index(3, 4, 10), 9);
    }

    #[test]
    fn spatial_shader_is_valid_wgsl() {
        let source = include_str!("spatial_scene.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        let light_direction = format!(
            "vec3<f32>({:.8}, {:.8}, {:.8})",
            SPATIAL_LIGHT_DIRECTION[0], SPATIAL_LIGHT_DIRECTION[1], SPATIAL_LIGHT_DIRECTION[2],
        );
        assert!(
            source.contains(&light_direction),
            "shader key light must match the shared camera authority"
        );
        for (name, value) in [
            ("SPATIAL_LINE_AMBIENT", SPATIAL_ROD_AMBIENT_LIGHT),
            ("SPATIAL_POLYGON_AMBIENT", SPATIAL_SURFACE_AMBIENT_LIGHT),
            ("SPATIAL_FAR_FOG", SPATIAL_FAR_FOG),
            ("SPATIAL_NEAR_FOG", SPATIAL_NEAR_FOG),
        ] {
            let declaration = format!("const {name}: f32 = {value:.2};");
            assert!(
                source.contains(&declaration),
                "shader {name} must match the shared display authority"
            );
        }
    }

    #[test]
    fn host_layouts_match_spatial_shader_inputs() {
        assert_eq!(std::mem::size_of::<SpatialUniform>(), 80);
        assert_eq!(std::mem::offset_of!(SpatialUniform, viewport_and_scale), 0);
        assert_eq!(std::mem::offset_of!(SpatialUniform, center_and_radius), 16);
        assert_eq!(std::mem::offset_of!(SpatialUniform, quaternion), 32);
        assert_eq!(std::mem::offset_of!(SpatialUniform, background), 48);
        assert_eq!(std::mem::offset_of!(SpatialUniform, base_width), 64);
        assert_eq!(std::mem::size_of::<GpuSpatialLine>(), 48);
        assert_eq!(std::mem::offset_of!(GpuSpatialLine, start), 0);
        assert_eq!(std::mem::offset_of!(GpuSpatialLine, end), 12);
        assert_eq!(std::mem::offset_of!(GpuSpatialLine, width), 24);
        assert_eq!(std::mem::offset_of!(GpuSpatialLine, color), 32);
        assert_eq!(std::mem::size_of::<GpuSpatialVertex>(), 32);
    }

    #[test]
    fn spatial_background_obeys_the_shared_depth_attachment_contract() {
        let background = spatial_depth_state(false, wgpu::CompareFunction::Always);
        assert_eq!(background.format, DEPTH_FORMAT);
        assert!(!background.depth_write_enabled);
        assert_eq!(background.depth_compare, wgpu::CompareFunction::Always);

        let geometry = spatial_depth_state(true, wgpu::CompareFunction::LessEqual);
        assert_eq!(geometry.format, DEPTH_FORMAT);
        assert!(geometry.depth_write_enabled);
        assert_eq!(geometry.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn software_preview_is_bounded_and_depth_sorted() {
        let scene = Scene3d {
            primitives: (0..SOFTWARE_FALLBACK_PREVIEW_LINES + 32)
                .map(|index| line(index as f64))
                .collect(),
            background: None,
        };
        let scene = SpatialScene::from_scene(scene).unwrap();
        let preview = scene
            .software_preview(Size::new(800.0, 600.0), Orbit3d::canonical(), false, 1.0)
            .unwrap();
        assert_eq!(preview.len(), SOFTWARE_FALLBACK_PREVIEW_LINES);
        assert!(
            preview
                .windows(2)
                .all(|pair| pair[0].depth() <= pair[1].depth())
        );
    }

    #[test]
    fn current_orbit_svg_keeps_all_primitives() {
        let scene = Scene3d {
            primitives: vec![line(0.0), line(1.0)],
            background: Some([1, 2, 3]),
        };
        let scene = SpatialScene::from_scene(scene).unwrap();
        let projected = scene
            .project_for_svg(Orbit3d::canonical(), Palette::Light, &|| false)
            .unwrap();
        assert_eq!(projected.primitives.len(), 2);
        assert_eq!(projected.background, Some([1, 2, 3]));
    }

    #[test]
    fn spatial_appearance_is_green_by_default_and_preserves_explicit_hue() {
        for palette in [Palette::Light, Palette::Dark] {
            let green = resolved_lit_rgb(StrokeColor::ThemeDefault, 0.5, palette, None, 0.78, 0.5);
            assert!(green[1] > green[0] && green[1] > green[2]);

            let explicit = resolved_lit_rgb(
                StrokeColor::Rgb([180, 80, 20]),
                0.5,
                palette,
                None,
                0.78,
                0.5,
            );
            assert!(explicit[0] > explicit[1] && explicit[1] > explicit[2]);
        }
    }

    #[test]
    fn spatial_width_reference_tames_large_source_widths_without_flattening_taper() {
        let scene = SpatialScene::from_scene(Scene3d {
            primitives: vec![
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                    width: 78.0,
                    color: StrokeColor::ThemeDefault,
                }),
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                    width: 93.0,
                    color: StrokeColor::ThemeDefault,
                }),
            ],
            background: None,
        })
        .unwrap();
        assert!((scene.width_reference() - 85.5).abs() < 1.0e-12);

        let transform = ViewTransform3d::new(
            scene.bounds.into_view_bounds(),
            ViewportSize::new(800.0, 600.0),
            Orbit3d::canonical(),
        );
        let lines = gpu_spatial_lines(&scene, transform, false, 2, 0..2).unwrap();
        assert!(lines[0].width < 1.0);
        assert!(lines[1].width > 1.0);
        assert!((lines[0].width / lines[1].width - 78.0 / 93.0).abs() < 1.0e-6);
    }

    #[test]
    fn equal_depth_software_and_svg_order_follow_mixed_source_order() {
        let coincident_line = || StyledLine3d {
            line: Line3d((0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
            width: 1.0,
            color: StrokeColor::ThemeDefault,
        };
        let coincident_polygon = || Polygon3d {
            vertices: vec![(0.5, 0.0, 0.0); 3],
            color: StrokeColor::ThemeDefault,
        };
        let scene = SpatialScene::from_scene(Scene3d {
            primitives: vec![
                Primitive3d::Polygon(coincident_polygon()),
                Primitive3d::Line(coincident_line()),
                Primitive3d::Polygon(coincident_polygon()),
                Primitive3d::Line(coincident_line()),
            ],
            background: None,
        })
        .unwrap();

        assert_eq!(
            scene
                .polygons
                .iter()
                .map(|polygon| polygon.lines_before)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        let preview = scene
            .software_preview(Size::new(800.0, 600.0), Orbit3d::canonical(), false, 1.0)
            .unwrap();
        assert_eq!(
            preview
                .iter()
                .map(ProjectedPrimitive3d::source_position)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );

        let svg_scene = scene
            .project_for_svg(Orbit3d::canonical(), Palette::Light, &|| false)
            .unwrap();
        assert!(matches!(svg_scene.primitives[0], Primitive2d::Polygon(_)));
        assert!(matches!(svg_scene.primitives[1], Primitive2d::Line(_)));
        assert!(matches!(svg_scene.primitives[2], Primitive2d::Polygon(_)));
        assert!(matches!(svg_scene.primitives[3], Primitive2d::Line(_)));
    }

    #[test]
    fn streamed_builder_keeps_consecutive_and_cross_batch_polygon_offsets() {
        let mut builder = SpatialSceneBuilder::new(None);
        builder
            .extend(
                vec![styled_line(0.0)],
                vec![polygon()],
                vec![Turtle3dPrimitiveKind::Line, Turtle3dPrimitiveKind::Polygon],
            )
            .unwrap();
        builder
            .extend(
                vec![styled_line(1.0)],
                vec![polygon(), polygon(), polygon()],
                vec![
                    Turtle3dPrimitiveKind::Polygon,
                    Turtle3dPrimitiveKind::Line,
                    Turtle3dPrimitiveKind::Polygon,
                    Turtle3dPrimitiveKind::Polygon,
                ],
            )
            .unwrap();
        let scene = builder.finish().unwrap();

        assert_eq!(
            scene
                .polygons
                .iter()
                .map(|polygon| polygon.lines_before)
                .collect::<Vec<_>>(),
            [1, 1, 2, 2]
        );
        validate_spatial_polygon_order(&scene.polygons, scene.lines.len()).unwrap();
    }

    #[test]
    fn streamed_builder_rejects_malformed_primitive_order() {
        let mut builder = SpatialSceneBuilder::new(None);
        assert!(matches!(
            builder.extend(
                vec![styled_line(0.0)],
                vec![polygon()],
                vec![Turtle3dPrimitiveKind::Line],
            ),
            Err(SpatialSceneError::InvalidGeometry(_))
        ));
    }

    #[test]
    fn native_collection_derives_checked_f32_bounds() {
        let scene = SpatialScene::from_scene(Scene3d {
            primitives: vec![Primitive3d::Line(StyledLine3d {
                line: Line3d((-2.0, 3.0, -4.0), (5.0, -6.0, 7.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        })
        .unwrap();

        assert_eq!(
            scene.bounds,
            RenderBounds3d {
                min_x: -2.0,
                max_x: 5.0,
                min_y: -6.0,
                max_y: 3.0,
                min_z: -4.0,
                max_z: 7.0,
            }
        );
    }

    #[test]
    fn native_collection_rejects_lossy_coordinates_and_widths() {
        let invalid_coordinate = Scene3d {
            primitives: vec![Primitive3d::Line(StyledLine3d {
                line: Line3d((0.0, 0.0, 0.0), (f64::MAX, 1.0, 1.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };
        assert!(matches!(
            SpatialScene::from_scene(invalid_coordinate),
            Err(SpatialSceneError::InvalidGeometry(_))
        ));

        let invalid_width = Scene3d {
            primitives: vec![Primitive3d::Line(StyledLine3d {
                line: Line3d((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                width: -1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };
        assert!(matches!(
            SpatialScene::from_scene(invalid_width),
            Err(SpatialSceneError::InvalidGeometry(_))
        ));
    }

    #[test]
    fn polygon_batches_cover_every_triangle_without_truncation() {
        let scene = SpatialScene::from_scene(Scene3d {
            primitives: vec![Primitive3d::Polygon(Polygon3d {
                vertices: vec![
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.5, 1.5, 0.0),
                    (0.0, 1.0, 0.0),
                    (-0.5, 0.5, 0.0),
                ],
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        })
        .unwrap();
        let transform = ViewTransform3d::new(
            scene.bounds.into_view_bounds(),
            ViewportSize::new(100.0, 100.0),
            Orbit3d::canonical(),
        );
        let mut cursor = PolygonCursor::default();
        let mut completed = 0usize;
        while completed < scene.polygon_triangle_count {
            let (vertices, triangles) =
                gpu_spatial_polygon_chunk(&scene, transform, false, &mut cursor, 6).unwrap();
            assert!(vertices.len() <= 6);
            assert_eq!(vertices.len(), triangles * 3);
            assert!(triangles > 0);
            completed += triangles;
        }
        assert_eq!(completed, 4);
    }

    #[test]
    fn software_preview_completes_spatial_refinement_honestly() {
        let scene = SpatialScene::from_scene(Scene3d {
            primitives: vec![line(0.0), line(1.0)],
            background: None,
        })
        .unwrap();
        assert!(scene.is_refining());
        assert_eq!(scene.refinement_progress(), (0, 2));

        scene.mark_software_preview_complete();

        assert!(!scene.is_refining());
        assert_eq!(scene.refinement_progress(), (2, 2));
    }

    #[test]
    fn gpu_failure_is_sticky_and_disables_refinement() {
        let scene = SpatialScene::from_scene(Scene3d {
            primitives: vec![line(0.0)],
            background: None,
        })
        .unwrap();
        assert_eq!(scene.gpu_refinement_failure(), None);

        let failure = SpatialSceneError::ResourceExhausted { requested: 17 };
        scene.record_gpu_refinement_failure(failure);

        assert_eq!(scene.gpu_refinement_failure(), Some(failure));
        assert!(!scene.is_refining());
        scene.restart_refinement();
        assert_eq!(scene.gpu_refinement_failure(), Some(failure));
        assert_eq!(scene.refinement_progress(), (0, 1));
    }

    #[test]
    fn gpu_layout_helpers_reject_device_and_draw_count_limits() {
        let item_size = std::mem::size_of::<GpuSpatialLine>() as u64;
        assert_eq!(checked_gpu_chunk_capacity(item_size, item_size * 4), Ok(4));
        assert!(matches!(
            checked_gpu_batch_layout::<GpuSpatialLine>(1, item_size - 1),
            Err(SpatialSceneError::ResourceExhausted { requested: 1 })
        ));
        assert!(matches!(
            checked_gpu_batch_layout::<GpuSpatialLine>(usize::MAX, u64::MAX),
            Err(SpatialSceneError::ResourceExhausted {
                requested: usize::MAX
            })
        ));
    }

    #[test]
    fn refinement_counts_reject_overflow() {
        assert!(matches!(
            checked_triangle_count([usize::MAX, 5]),
            Err(SpatialSceneError::ResourceExhausted {
                requested: usize::MAX
            })
        ));
        assert!(matches!(
            checked_refinement_total(usize::MAX, 1),
            Err(SpatialSceneError::ResourceExhausted {
                requested: usize::MAX
            })
        ));
    }
}
