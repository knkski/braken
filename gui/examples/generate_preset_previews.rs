//! Generate the checked-in vector thumbnails used by the GUI preset catalog.

#[allow(dead_code)]
#[path = "../src/camera.rs"]
mod camera;

use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};
use braken_gui::{
    orientation::OrientationLandmarks,
    presets::{Preset, get_presets},
    theme_palette::theme_default_rgb8_at,
};
use braken_viz::targets::{
    Palette, StrokeWidthEstimator, adaptive_scene_stroke_width,
    adaptive_spatial_scene_stroke_width, spatial_lit_color_bounded, spatial_theme_default_color_at,
    turtle_stroke_rgb,
};
use braken_viz::{
    Point2d, Polygon2d, Primitive2d, Primitive3d, Scene2d, Scene3d, StrokeColor, StyledLine2d,
    Turtle3dConfig, Turtle3dWidthReferenceEstimator, Visualization, VisualizationContext,
    VisualizeRequest, VisualizerBackend, VisualizerConfig, VisualizerKind,
    normalized_turtle_3d_width, visualize,
};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use camera::{
    Camera2d, Orbit3d, ViewBounds, ViewBounds3d, ViewTransform, ViewTransform3d, ViewportSize,
    WorldPoint, WorldPoint3d,
};

const PREVIEW_WIDTH: f64 = 64.0;
const PREVIEW_HEIGHT: f64 = 46.0;
const PREVIEW_PADDING: f64 = 3.0;
const PREVIEW_SEED: u64 = 0;

#[derive(Debug)]
struct GeneratedPreview {
    name: String,
    light_file_name: String,
    dark_file_name: String,
    light_svg: String,
    dark_svg: String,
}

#[derive(Clone, Copy)]
struct Segment {
    start: (i32, i32),
    end: (i32, i32),
}

#[derive(Clone, Copy)]
struct PreviewLine {
    styled: StyledLine2d,
    palette_position: f64,
    spatial_appearance: Option<SpatialAppearance>,
}

struct PreviewPolygon {
    polygon: Polygon2d,
    palette_position: f64,
    spatial_appearance: Option<SpatialAppearance>,
}

#[derive(Clone, Copy)]
struct SpatialAppearance {
    light: f64,
    near_depth: f64,
}

#[derive(Clone, Copy)]
enum PreviewPrimitiveIndex {
    Line(usize),
    Polygon(usize),
}

#[derive(Clone, Copy)]
struct PreviewDepthEntry {
    depth: f64,
    source_position: usize,
    primitive: PreviewPrimitiveIndex,
}

struct PreviewGeometry {
    lines: Vec<PreviewLine>,
    polygons: Vec<PreviewPolygon>,
    bounds: (f64, f64, f64, f64),
    spatial_depth_order: Option<Vec<PreviewDepthEntry>>,
    spatial_background: Option<[u8; 3]>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let check = parse_args()?;
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let asset_dir = manifest_dir.join("assets/preset-previews");
    let generated_manifest = manifest_dir.join("src/generated_preset_previews.rs");

    let previews = generate_all().map_err(|error| format!("preview generation failed: {error}"))?;
    let manifest = encode_manifest(&previews);
    if check {
        check_outputs(&asset_dir, &generated_manifest, &previews, &manifest)?;
        println!("all {} preset previews are current", previews.len());
    } else {
        write_outputs(&asset_dir, &generated_manifest, &previews, &manifest)?;
        println!("generated {} preset previews", previews.len());
    }
    Ok(())
}

fn parse_args() -> Result<bool, String> {
    let mut check = false;
    for argument in env::args().skip(1) {
        match argument.as_str() {
            "--check" if !check => check = true,
            "-h" | "--help" => {
                println!(
                    "Generate GUI preset SVG previews.\n\nUsage: generate_preset_previews [--check]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {argument:?}")),
        }
    }
    Ok(check)
}

fn generate_all() -> Result<Vec<GeneratedPreview>, String> {
    let presets = get_presets();
    let mut file_names = BTreeSet::new();
    let mut previews = Vec::with_capacity(presets.len());
    for preset in presets {
        let light_file_name = asset_file_name(&preset.name, Palette::Light);
        let dark_file_name = asset_file_name(&preset.name, Palette::Dark);
        for file_name in [&light_file_name, &dark_file_name] {
            if !file_names.insert(file_name.clone()) {
                return Err(format!(
                    "{} collides with another generated asset name {file_name}",
                    preset.name
                ));
            }
        }
        let (light_svg, dark_svg) = generate_preview(&preset)?;
        previews.push(GeneratedPreview {
            name: preset.name,
            light_file_name,
            dark_file_name,
            light_svg,
            dark_svg,
        });
    }
    Ok(previews)
}

fn generate_preview(preset: &Preset) -> Result<(String, String), String> {
    if !matches!(
        preset.visualizer,
        VisualizerKind::Turtle2d | VisualizerKind::Turtle3d
    ) {
        return Err(format!(
            "{} uses unsupported preview visualizer {}",
            preset.name, preset.visualizer
        ));
    }
    let grammar = CompiledGrammar::parse(&preset.source)
        .map_err(|error| format!("{} failed to parse: {error}", preset.name))?;
    let calculation = calculate(CalculationRequest {
        grammar,
        iterations: preset.preview_iters as usize,
        backend: BackendChoice::Cpu,
        seed: PREVIEW_SEED,
        semantics: Default::default(),
        limits: CalculationLimits::unbounded_production(),
    })
    .map_err(|error| format!("{} failed to derive: {error}", preset.name))?;
    let config = match preset.visualizer {
        VisualizerKind::Turtle2d => VisualizerConfig::Turtle2d(preset.turtle_config.clone()),
        VisualizerKind::Turtle3d => {
            VisualizerConfig::Turtle3d(Turtle3dConfig::from(preset.turtle_config.clone()))
        }
        _ => unreachable!("unsupported preview kind rejected above"),
    };
    let visualization = visualize(VisualizeRequest {
        generation: &calculation.generation,
        backend: VisualizerBackend::Cpu,
        config,
        context: VisualizationContext {
            iterations: preset.preview_iters as usize,
            seed: PREVIEW_SEED,
            derivation_backend: Some("cpu"),
            elapsed: None,
        },
    })
    .map_err(|error| format!("{} failed to visualize: {error}", preset.name))?;
    let geometry = match visualization {
        Visualization::Scene2d(scene) => prepare_2d_geometry(preset, scene)?,
        Visualization::Scene3d(scene) => prepare_3d_geometry(preset, scene)?,
    };
    Ok((
        encode_preview(preset, &geometry, Palette::Light)?,
        encode_preview(preset, &geometry, Palette::Dark)?,
    ))
}

fn prepare_2d_geometry(preset: &Preset, scene: Scene2d) -> Result<PreviewGeometry, String> {
    let mut lines = scene
        .primitives
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive2d::Line(line) => Some(*line),
            Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
        })
        .collect::<Vec<_>>();
    let polygons = scene
        .primitives
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive2d::Polygon(polygon) => Some(polygon.clone()),
            Primitive2d::Line(_) | Primitive2d::Text(_) => None,
        })
        .collect::<Vec<_>>();
    if preset.visualizer == VisualizerKind::Turtle2d && polygons.is_empty() {
        let mut landmarks = OrientationLandmarks::default();
        for line in &lines {
            landmarks.observe(line.line.0, line.line.1);
        }
        let transform = landmarks.transform(
            preset.orientation_anchor,
            preset.turtle_config.initial_angle,
        );
        if !transform.is_identity() {
            for line in &mut lines {
                line.line.0 = transform.apply(line.line.0);
                line.line.1 = transform.apply(line.line.1);
            }
        }
    }
    finish_2d_geometry(&preset.name, preset.preview_iters, lines, polygons)
}

fn prepare_3d_geometry(preset: &Preset, scene: Scene3d) -> Result<PreviewGeometry, String> {
    let bounds = scene_bounds_3d(&scene, &preset.name)?;
    let transform = ViewTransform3d::new(
        ViewBounds3d::new(bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5),
        ViewportSize::new(PREVIEW_WIDTH, PREVIEW_HEIGHT),
        Orbit3d::canonical(),
    );
    let mut width_reference = Turtle3dWidthReferenceEstimator::default();
    for primitive in &scene.primitives {
        if let Primitive3d::Line(line) = primitive {
            width_reference.observe_line(line);
        }
    }
    let width_reference = width_reference.width_reference();
    let background = scene.background;
    let mut lines = Vec::new();
    let mut polygons = Vec::new();
    let mut depth_order = Vec::new();
    depth_order
        .try_reserve_exact(scene.primitives.len())
        .map_err(|_| format!("{} preview is too large to allocate", preset.name))?;
    for (source_position, primitive) in scene.primitives.into_iter().enumerate() {
        match primitive {
            Primitive3d::Line(line) => {
                let start = WorldPoint3d::new(line.line.0.0, line.line.0.1, line.line.0.2);
                let end = WorldPoint3d::new(line.line.1.0, line.line.1.1, line.line.1.2);
                let view_start = transform.view_position(start);
                let view_end = transform.view_position(end);
                let near_depth = transform.normalized_midpoint_depth(start, end);
                let line_index = lines.len();
                lines.push(PreviewLine {
                    styled: StyledLine2d {
                        line: braken_viz::Line2d(
                            (view_start.x, view_start.y),
                            (view_end.x, view_end.y),
                        ),
                        width: normalized_turtle_3d_width(line.width, width_reference),
                        color: line.color,
                    },
                    palette_position: transform.world_palette_position(start, end),
                    spatial_appearance: Some(SpatialAppearance {
                        light: transform.rod_light(start, end),
                        near_depth,
                    }),
                });
                depth_order.push(PreviewDepthEntry {
                    depth: near_depth,
                    source_position,
                    primitive: PreviewPrimitiveIndex::Line(line_index),
                });
            }
            Primitive3d::Polygon(polygon) => {
                let Some(center) = polygon_center(&polygon.vertices) else {
                    continue;
                };
                let surface_light = transform.surface_light(
                    polygon
                        .vertices
                        .iter()
                        .map(|&(x, y, z)| WorldPoint3d::new(x, y, z)),
                );
                let mut mean_view_depth = 0.0;
                let vertices = polygon
                    .vertices
                    .iter()
                    .enumerate()
                    .map(|(index, &(x, y, z))| {
                        let view = transform.view_position(WorldPoint3d::new(x, y, z));
                        mean_view_depth += (view.z - mean_view_depth) / (index + 1) as f64;
                        (view.x, view.y)
                    })
                    .collect();
                let near_depth = transform.normalized_view_depth(mean_view_depth);
                let polygon_index = polygons.len();
                polygons.push(PreviewPolygon {
                    polygon: Polygon2d {
                        vertices,
                        color: polygon.color,
                    },
                    palette_position: transform.world_palette_position(center, center),
                    spatial_appearance: Some(SpatialAppearance {
                        light: surface_light,
                        near_depth,
                    }),
                });
                depth_order.push(PreviewDepthEntry {
                    depth: near_depth,
                    source_position,
                    primitive: PreviewPrimitiveIndex::Polygon(polygon_index),
                });
            }
        }
    }
    depth_order.sort_by(|left, right| {
        left.depth
            .total_cmp(&right.depth)
            .then_with(|| left.source_position.cmp(&right.source_position))
    });
    finish_geometry(
        preset.name.clone(),
        preset.preview_iters,
        lines,
        polygons,
        Some(depth_order),
        background,
    )
}

fn finish_2d_geometry(
    name: &str,
    preview_iters: u32,
    lines: Vec<StyledLine2d>,
    polygons: Vec<Polygon2d>,
) -> Result<PreviewGeometry, String> {
    let bounds = scene_bounds_2d(&lines, &polygons, name)?;
    let transform = ViewTransform::new(
        ViewBounds::new(bounds.0, bounds.1, bounds.2, bounds.3),
        ViewportSize::new(PREVIEW_WIDTH, PREVIEW_HEIGHT),
        Camera2d::fit(),
    );
    let lines = lines
        .into_iter()
        .map(|styled| PreviewLine {
            palette_position: transform.world_palette_position(
                WorldPoint::new(styled.line.0.0, styled.line.0.1),
                WorldPoint::new(styled.line.1.0, styled.line.1.1),
            ),
            styled,
            spatial_appearance: None,
        })
        .collect();
    let polygons = polygons
        .into_iter()
        .map(|polygon| PreviewPolygon {
            polygon,
            // Planar Canvas rendering uses the center flare color for a
            // theme-default polygon rather than deriving a line direction.
            palette_position: 0.5,
            spatial_appearance: None,
        })
        .collect();
    finish_geometry(name.to_owned(), preview_iters, lines, polygons, None, None)
}

fn finish_geometry(
    name: String,
    preview_iters: u32,
    lines: Vec<PreviewLine>,
    polygons: Vec<PreviewPolygon>,
    spatial_depth_order: Option<Vec<PreviewDepthEntry>>,
    spatial_background: Option<[u8; 3]>,
) -> Result<PreviewGeometry, String> {
    if lines.is_empty() && polygons.is_empty() {
        return Err(format!(
            "{name} produced no preview geometry at iteration {preview_iters}"
        ));
    }
    let bounds = preview_bounds_2d(&lines, &polygons, &name)?;
    Ok(PreviewGeometry {
        lines,
        polygons,
        bounds,
        spatial_depth_order,
        spatial_background,
    })
}

fn scene_bounds_2d(
    lines: &[StyledLine2d],
    polygons: &[Polygon2d],
    name: &str,
) -> Result<(f64, f64, f64, f64), String> {
    bounds_2d(
        lines
            .iter()
            .flat_map(|line| [line.line.0, line.line.1])
            .chain(
                polygons
                    .iter()
                    .flat_map(|polygon| polygon.vertices.iter().copied()),
            ),
        name,
    )
}

fn preview_bounds_2d(
    lines: &[PreviewLine],
    polygons: &[PreviewPolygon],
    name: &str,
) -> Result<(f64, f64, f64, f64), String> {
    bounds_2d(
        lines
            .iter()
            .flat_map(|line| [line.styled.line.0, line.styled.line.1])
            .chain(
                polygons
                    .iter()
                    .flat_map(|polygon| polygon.polygon.vertices.iter().copied()),
            ),
        name,
    )
}

fn bounds_2d(
    mut points: impl Iterator<Item = Point2d>,
    name: &str,
) -> Result<(f64, f64, f64, f64), String> {
    let Some(first) = points.next() else {
        return Err(format!("{name} produced no preview geometry"));
    };
    if !first.0.is_finite() || !first.1.is_finite() {
        return Err(format!("{name} produced non-finite preview geometry"));
    }
    points.try_fold(
        (first.0, first.0, first.1, first.1),
        |(min_x, max_x, min_y, max_y), point| {
            if !point.0.is_finite() || !point.1.is_finite() {
                return Err(format!("{name} produced non-finite preview geometry"));
            }
            Ok((
                min_x.min(point.0),
                max_x.max(point.0),
                min_y.min(point.1),
                max_y.max(point.1),
            ))
        },
    )
}

fn scene_bounds_3d(scene: &Scene3d, name: &str) -> Result<(f64, f64, f64, f64, f64, f64), String> {
    let mut bounds = None;
    let mut include = |point: (f64, f64, f64)| -> Result<(), String> {
        if !point.0.is_finite() || !point.1.is_finite() || !point.2.is_finite() {
            return Err(format!("{name} produced non-finite preview geometry"));
        }
        bounds = Some(bounds.map_or(
            (point.0, point.0, point.1, point.1, point.2, point.2),
            |(min_x, max_x, min_y, max_y, min_z, max_z): (f64, f64, f64, f64, f64, f64)| {
                (
                    min_x.min(point.0),
                    max_x.max(point.0),
                    min_y.min(point.1),
                    max_y.max(point.1),
                    min_z.min(point.2),
                    max_z.max(point.2),
                )
            },
        ));
        Ok(())
    };
    for primitive in &scene.primitives {
        match primitive {
            Primitive3d::Line(line) => {
                include(line.line.0)?;
                include(line.line.1)?;
            }
            Primitive3d::Polygon(polygon) => {
                for &point in &polygon.vertices {
                    include(point)?;
                }
            }
        }
    }
    bounds.ok_or_else(|| format!("{name} produced no preview geometry"))
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

fn encode_preview(
    preset: &Preset,
    geometry: &PreviewGeometry,
    palette: Palette,
) -> Result<String, String> {
    let lines = &geometry.lines;
    let polygons = &geometry.polygons;
    let mut width_estimator = StrokeWidthEstimator::default();
    for line in lines {
        width_estimator.observe(line.styled.line);
    }
    let (min_x, max_x, min_y, max_y) = geometry.bounds;
    let world_width = max_x - min_x;
    let world_height = max_y - min_y;
    if world_width <= f64::EPSILON && world_height <= f64::EPSILON {
        return Err(format!(
            "{} produced degenerate preview geometry",
            preset.name
        ));
    }
    let available_width = PREVIEW_WIDTH - PREVIEW_PADDING * 2.0;
    let available_height = PREVIEW_HEIGHT - PREVIEW_PADDING * 2.0;
    let horizontal_scale = if world_width > f64::EPSILON {
        available_width / world_width
    } else {
        f64::INFINITY
    };
    let vertical_scale = if world_height > f64::EPSILON {
        available_height / world_height
    } else {
        f64::INFINITY
    };
    let scale = horizontal_scale.min(vertical_scale);
    let base_stroke_width = if geometry.spatial_depth_order.is_some() {
        adaptive_spatial_scene_stroke_width(
            width_estimator.total_line_length(),
            width_estimator.line_count(),
            (world_width, world_height),
            scale,
        )
    } else {
        adaptive_scene_stroke_width(
            width_estimator.total_line_length(),
            width_estimator.line_count(),
            (world_width, world_height),
            scale,
        )
    };
    let drawn_width = world_width * scale;
    let drawn_height = world_height * scale;
    let offset_x = PREVIEW_PADDING + (available_width - drawn_width) / 2.0;
    let offset_y = PREVIEW_PADDING + (available_height - drawn_height) / 2.0;

    let map_point = |point: (f64, f64)| {
        (
            quantize(offset_x + (point.0 - min_x) * scale),
            quantize(offset_y + (max_y - point.1) * scale),
        )
    };
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"64\" height=\"46\" viewBox=\"0 0 64 46\">\n<!-- Generated from {} at iteration {} with seed {}. -->\n",
        escape_xml(&preset.name),
        preset.preview_iters,
        PREVIEW_SEED
    );
    if let Some(order) = &geometry.spatial_depth_order {
        let written = encode_spatial_preview(
            &mut svg,
            geometry,
            order,
            palette,
            base_stroke_width,
            &map_point,
        );
        if written == 0 {
            return Err(format!(
                "{} preview collapses completely at thumbnail resolution",
                preset.name
            ));
        }
        svg.push_str("</svg>\n");
        return Ok(svg);
    }

    let mut paths: BTreeMap<(i32, [u8; 3]), Vec<Segment>> = BTreeMap::new();
    for line in lines {
        let styled = line.styled;
        let start = map_point(styled.line.0);
        let end = map_point(styled.line.1);
        if start == end {
            continue;
        }
        let effective_width = styled.width * base_stroke_width;
        if !effective_width.is_finite() || effective_width <= 0.0 {
            continue;
        }
        // Thumbnail SVGs retain fractional coverage but bound extreme source
        // width commands so one imported branch cannot obscure the preview.
        let stroke_width = quantize(effective_width.clamp(0.01, 10.0));
        paths
            .entry((
                stroke_width,
                resolve_preview_rgb(
                    styled.color,
                    line.palette_position,
                    palette,
                    line.spatial_appearance,
                    geometry.spatial_background,
                ),
            ))
            .or_default()
            .push(Segment { start, end });
    }
    if paths.is_empty() && polygons.is_empty() {
        return Err(format!(
            "{} preview collapses completely at thumbnail resolution",
            preset.name
        ));
    }

    for polygon in polygons {
        let points = polygon
            .polygon
            .vertices
            .iter()
            .copied()
            .map(map_point)
            .map(|point| {
                format!(
                    "{},{}",
                    format_quantized(point.0),
                    format_quantized(point.1)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let [red, green, blue] = resolve_preview_rgb(
            polygon.polygon.color,
            polygon.palette_position,
            palette,
            polygon.spatial_appearance,
            geometry.spatial_background,
        );
        writeln!(
            svg,
            "<polygon points=\"{points}\" fill=\"#{red:02x}{green:02x}{blue:02x}\"/>"
        )
        .expect("writing to a String cannot fail");
    }
    for ((stroke_width, [red, green, blue]), segments) in paths {
        let mut data = String::new();
        let mut current = None;
        for segment in segments {
            if current != Some(segment.start) {
                write_point_command(&mut data, 'M', segment.start);
            }
            write_point_command(&mut data, 'L', segment.end);
            current = Some(segment.end);
        }
        writeln!(
            svg,
            "<path d=\"{data}\" fill=\"none\" stroke=\"#{red:02x}{green:02x}{blue:02x}\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
            format_quantized(stroke_width)
        )
        .expect("writing to a String cannot fail");
    }
    svg.push_str("</svg>\n");
    Ok(svg)
}

fn encode_spatial_preview(
    svg: &mut String,
    geometry: &PreviewGeometry,
    order: &[PreviewDepthEntry],
    palette: Palette,
    base_stroke_width: f64,
    map_point: &impl Fn(Point2d) -> (i32, i32),
) -> usize {
    let mut pending_key = None;
    let mut pending_data = String::new();
    let mut current = None;
    let mut written = 0usize;

    for entry in order {
        match entry.primitive {
            PreviewPrimitiveIndex::Line(index) => {
                let Some(line) = geometry.lines.get(index) else {
                    continue;
                };
                let start = map_point(line.styled.line.0);
                let end = map_point(line.styled.line.1);
                if start == end {
                    continue;
                }
                let effective_width = line.styled.width * base_stroke_width;
                if !effective_width.is_finite() || effective_width <= 0.0 {
                    continue;
                }
                let key = (
                    quantize(effective_width.clamp(0.01, 10.0)),
                    resolve_preview_rgb(
                        line.styled.color,
                        line.palette_position,
                        palette,
                        line.spatial_appearance,
                        geometry.spatial_background,
                    ),
                );
                if pending_key != Some(key) {
                    flush_preview_path(svg, pending_key.take(), &mut pending_data);
                    current = None;
                    pending_key = Some(key);
                }
                if current != Some(start) {
                    write_point_command(&mut pending_data, 'M', start);
                }
                write_point_command(&mut pending_data, 'L', end);
                current = Some(end);
                written = written.saturating_add(1);
            }
            PreviewPrimitiveIndex::Polygon(index) => {
                flush_preview_path(svg, pending_key.take(), &mut pending_data);
                current = None;
                let Some(polygon) = geometry.polygons.get(index) else {
                    continue;
                };
                let points = polygon
                    .polygon
                    .vertices
                    .iter()
                    .copied()
                    .map(map_point)
                    .map(|point| {
                        format!(
                            "{},{}",
                            format_quantized(point.0),
                            format_quantized(point.1)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let [red, green, blue] = resolve_preview_rgb(
                    polygon.polygon.color,
                    polygon.palette_position,
                    palette,
                    polygon.spatial_appearance,
                    geometry.spatial_background,
                );
                writeln!(
                    svg,
                    "<polygon points=\"{points}\" fill=\"#{red:02x}{green:02x}{blue:02x}\"/>"
                )
                .expect("writing to a String cannot fail");
                written = written.saturating_add(1);
            }
        }
    }
    flush_preview_path(svg, pending_key, &mut pending_data);
    written
}

fn flush_preview_path(svg: &mut String, key: Option<(i32, [u8; 3])>, data: &mut String) {
    let Some((stroke_width, [red, green, blue])) = key else {
        return;
    };
    if data.is_empty() {
        return;
    }
    writeln!(
        svg,
        "<path d=\"{data}\" fill=\"none\" stroke=\"#{red:02x}{green:02x}{blue:02x}\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
        format_quantized(stroke_width)
    )
    .expect("writing to a String cannot fail");
    data.clear();
}

fn resolve_preview_rgb(
    color: StrokeColor,
    position: f64,
    palette: Palette,
    spatial_appearance: Option<SpatialAppearance>,
    spatial_background: Option<[u8; 3]>,
) -> [u8; 3] {
    let Some(appearance) = spatial_appearance else {
        return turtle_stroke_rgb(color, palette)
            .unwrap_or_else(|| theme_default_rgb8_at(position, palette));
    };
    let base = turtle_stroke_rgb(color, palette).map_or_else(
        || spatial_theme_default_color_at(position, palette),
        |rgb| rgb.map(|channel| f32::from(channel) / 255.0),
    );
    let [red, green, blue] = spatial_background.unwrap_or(match palette {
        Palette::Light => [242, 245, 249],
        Palette::Dark => [15, 20, 32],
    });
    spatial_lit_color_bounded(
        base,
        [
            f32::from(red) / 255.0,
            f32::from(green) / 255.0,
            f32::from(blue) / 255.0,
        ],
        appearance.light,
        appearance.near_depth,
    )
    .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn quantize(value: f64) -> i32 {
    (value * 100.0).round() as i32
}

fn format_quantized(value: i32) -> String {
    if value % 100 == 0 {
        return (value / 100).to_string();
    }
    let mut value = format!("{:.2}", value as f64 / 100.0);
    while value.ends_with('0') {
        value.pop();
    }
    value
}

fn write_point_command(output: &mut String, command: char, point: (i32, i32)) {
    output.push(command);
    output.push_str(&format_quantized(point.0));
    output.push(' ');
    output.push_str(&format_quantized(point.1));
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn asset_file_name(name: &str, palette: Palette) -> String {
    let mut slug = String::new();
    let mut separator = false;
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            if separator && !slug.is_empty() {
                slug.push('-');
            }
            separator = false;
            if slug.len() < 72 {
                slug.push(character.to_ascii_lowercase());
            }
        } else {
            separator = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug.push_str("preset");
    }
    let theme = match palette {
        Palette::Light => "light",
        Palette::Dark => "dark",
    };
    format!("{slug}--{:016x}-{theme}.svg", fnv1a(name.as_bytes()))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn encode_manifest(previews: &[GeneratedPreview]) -> String {
    let mut manifest = String::from(
        "// Generated by cargo run -p braken-gui --example generate_preset_previews.\n\
         // Do not edit by hand.\n\n\
         use braken_viz::targets::Palette;\n\n\
         #[cfg(test)]\n\
         pub const PRESET_PREVIEW_NAMES: &[&str] = &[\n",
    );
    for preview in previews {
        writeln!(manifest, "    {:?},", preview.name).expect("writing to a String cannot fail");
    }
    manifest.push_str(
        "];\n\n#[rustfmt::skip]\npub fn preset_preview_svg(name: &str, palette: Palette) -> Option<&'static [u8]> {\n    match name {\n",
    );
    for preview in previews {
        writeln!(
            manifest,
            "        {:?} => Some(match palette {{\n            Palette::Light => include_bytes!(\"../assets/preset-previews/{}\").as_slice(),\n            Palette::Dark => include_bytes!(\"../assets/preset-previews/{}\").as_slice(),\n        }}),",
            preview.name,
            preview.light_file_name,
            preview.dark_file_name,
        )
        .expect("writing to a String cannot fail");
    }
    manifest.push_str("        _ => None,\n    }\n}\n");
    manifest
}

fn check_outputs(
    asset_dir: &Path,
    manifest_path: &Path,
    previews: &[GeneratedPreview],
    manifest: &str,
) -> Result<(), String> {
    let mut problems = Vec::new();
    let expected_names = previews
        .iter()
        .flat_map(|preview| {
            [
                preview.light_file_name.as_str(),
                preview.dark_file_name.as_str(),
            ]
        })
        .collect::<BTreeSet<_>>();
    for preview in previews {
        for (file_name, expected) in [
            (&preview.light_file_name, &preview.light_svg),
            (&preview.dark_file_name, &preview.dark_svg),
        ] {
            let path = asset_dir.join(file_name);
            match fs::read_to_string(&path) {
                Ok(actual) if actual == *expected => {}
                Ok(_) => problems.push(format!("stale {}", path.display())),
                Err(error) => problems.push(format!("missing {}: {error}", path.display())),
            }
        }
    }
    if asset_dir.is_dir() {
        for path in svg_files(asset_dir)? {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                problems.push(format!("non-UTF-8 preview filename {}", path.display()));
                continue;
            };
            if !expected_names.contains(name) {
                problems.push(format!("extra {}", path.display()));
            }
        }
    }
    match fs::read_to_string(manifest_path) {
        Ok(actual) if actual == manifest => {}
        Ok(_) => problems.push(format!("stale {}", manifest_path.display())),
        Err(error) => problems.push(format!("missing {}: {error}", manifest_path.display())),
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

fn write_outputs(
    asset_dir: &Path,
    manifest_path: &Path,
    previews: &[GeneratedPreview],
    manifest: &str,
) -> Result<(), String> {
    fs::create_dir_all(asset_dir)
        .map_err(|error| format!("could not create {}: {error}", asset_dir.display()))?;
    let expected_names = previews
        .iter()
        .flat_map(|preview| {
            [
                preview.light_file_name.as_str(),
                preview.dark_file_name.as_str(),
            ]
        })
        .collect::<BTreeSet<_>>();
    for preview in previews {
        write_if_changed(
            &asset_dir.join(&preview.light_file_name),
            preview.light_svg.as_bytes(),
        )?;
        write_if_changed(
            &asset_dir.join(&preview.dark_file_name),
            preview.dark_svg.as_bytes(),
        )?;
    }
    write_if_changed(manifest_path, manifest.as_bytes())?;
    for path in svg_files(asset_dir)? {
        let keep = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| expected_names.contains(name));
        if !keep {
            fs::remove_file(&path)
                .map_err(|error| format!("could not remove stale {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn write_if_changed(path: &Path, contents: &[u8]) -> Result<(), String> {
    if fs::read(path).is_ok_and(|current| current == contents) {
        return Ok(());
    }
    fs::write(path, contents)
        .map_err(|error| format!("could not write {}: {error}", path.display()))
}

fn svg_files(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = fs::read_dir(directory)
        .map_err(|error| format!("could not read {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "svg"))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_asset_names_are_stable_and_filesystem_safe() {
        assert_eq!(
            asset_file_name("Hilbert Curve", Palette::Light),
            "hilbert-curve--79607893f5f850fa-light.svg"
        );
        assert_eq!(
            asset_file_name("Hilbert Curve", Palette::Dark),
            "hilbert-curve--79607893f5f850fa-dark.svg"
        );
        let name = asset_file_name("Étoile / star?", Palette::Light);
        assert!(name.starts_with("toile-star--"));
        assert!(!name.contains('/'));
    }

    #[test]
    fn color_resolution_preserves_explicit_rgb_and_themes_other_colors() {
        let exact = StrokeColor::Rgb([12, 34, 56]);
        assert_eq!(
            resolve_preview_rgb(exact, 0.0, Palette::Light, None, None),
            [12, 34, 56]
        );
        assert_eq!(
            resolve_preview_rgb(exact, 1.0, Palette::Dark, None, None),
            [12, 34, 56]
        );
        assert_ne!(
            resolve_preview_rgb(
                StrokeColor::PaletteIndex(0),
                0.5,
                Palette::Light,
                None,
                None,
            ),
            resolve_preview_rgb(StrokeColor::PaletteIndex(0), 0.5, Palette::Dark, None, None,),
        );
        assert_ne!(
            resolve_preview_rgb(StrokeColor::ThemeDefault, 0.0, Palette::Light, None, None,),
            resolve_preview_rgb(StrokeColor::ThemeDefault, 1.0, Palette::Light, None, None,),
        );
        assert_ne!(
            resolve_preview_rgb(StrokeColor::ThemeDefault, 0.5, Palette::Light, None, None,),
            resolve_preview_rgb(StrokeColor::ThemeDefault, 0.5, Palette::Dark, None, None,),
        );
    }

    #[test]
    fn spatial_preview_uses_green_albedo_and_shades_explicit_colors() {
        let appearance = Some(SpatialAppearance {
            light: 0.78,
            near_depth: 0.25,
        });
        let light = resolve_preview_rgb(
            StrokeColor::ThemeDefault,
            0.5,
            Palette::Light,
            appearance,
            None,
        );
        let dark = resolve_preview_rgb(
            StrokeColor::ThemeDefault,
            0.5,
            Palette::Dark,
            appearance,
            None,
        );
        assert!(light[1] > light[0] && light[1] > light[2]);
        assert!(dark[1] > dark[0] && dark[1] > dark[2]);

        let exact = [120, 70, 30];
        let shaded = resolve_preview_rgb(
            StrokeColor::Rgb(exact),
            0.5,
            Palette::Light,
            appearance,
            None,
        );
        assert_ne!(shaded, exact);
        assert!(shaded[0] > shaded[1] && shaded[1] > shaded[2]);
    }

    #[test]
    fn every_catalog_preview_is_deterministic_and_compact_svg() {
        let first = generate_all().expect("the preset catalog should generate");
        let second = generate_all().expect("the preset catalog should generate twice");
        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(second.iter()) {
            assert_eq!(left.name, right.name);
            assert_eq!(left.light_svg, right.light_svg);
            assert_eq!(left.dark_svg, right.dark_svg);
            for svg in [&left.light_svg, &left.dark_svg] {
                assert!(svg.starts_with("<svg "));
                assert!(svg.contains("<path "));
                assert!(svg.contains("stroke-linecap=\"round\""));
                assert!(svg.contains("stroke-linejoin=\"round\""));
                assert!(!svg.contains("<rect "));
                assert!(!svg.contains("<line "));
            }
        }
    }
}
