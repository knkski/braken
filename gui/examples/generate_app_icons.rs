//! Generate the checked-in Braken browser and native application icon assets.

use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};
use braken_viz::{
    Point2d, Primitive2d, Scene2d, StrokeColor, Turtle2dConfig, Visualization,
    VisualizationContext, VisualizeRequest, VisualizerBackend, VisualizerConfig, visualize,
};
use resvg::{tiny_skia, usvg};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const PERIOD_SECONDS: f64 = 3.2;
const STATIC_PHASE: f64 = 0.42;
const PAD: f64 = 0.10;
const SPAN: f64 = 1.0 - 2.0 * PAD;
const LOADER_SCALE: f64 = 0.65;
const LOADER_STROKE: f64 = 4.7;
const LOGO_SPAN: f64 = 82.0;
const LOGO_BASE_STROKE: f64 = 3.0;
const LOGO_SOURCE: &str = include_str!("../branding/app-icon.lsys");
const WEB_SIZES: &[u32] = &[180, 192, 512];
const NATIVE_SIZES: &[u32] = &[16, 32, 48, 64, 128, 256, 512, 1024];
const ICO_SIZES: &[u32] = &[16, 32, 48, 256];
const ICNS_SIZES: &[(u32, [u8; 4])] = &[
    (16, *b"icp4"),
    (32, *b"icp5"),
    (64, *b"icp6"),
    (128, *b"ic07"),
    (256, *b"ic08"),
    (512, *b"ic09"),
    (1024, *b"ic10"),
    (32, *b"ic11"),
    (64, *b"ic12"),
    (256, *b"ic13"),
    (512, *b"ic14"),
];

type Point = [f64; 2];

#[derive(Clone, Debug)]
struct Layer {
    points: Vec<Point>,
    opacity: f64,
}

#[derive(Clone, Copy, Debug)]
struct LogoTransform {
    center: Point2d,
    scale: f64,
}

#[derive(Debug)]
struct Asset {
    relative_path: String,
    bytes: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let check = parse_args()?;
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icons");
    let assets = generate_assets()?;

    if check {
        check_outputs(&output_dir, &assets)?;
        println!("all {} application icon assets are current", assets.len());
    } else {
        write_outputs(&output_dir, &assets)?;
        println!("generated {} application icon assets", assets.len());
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
                    "Generate Braken browser and native application icons.\n\n\
                     Usage: generate_app_icons [--check]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {argument:?}")),
        }
    }
    Ok(check)
}

fn generate_assets() -> Result<Vec<Asset>, String> {
    let animated = animated_loader_svg();
    let static_mark = static_mark_svg(true);
    let logo = generate_logo_scene()?;
    let favicon = logo_svg(&logo, 64, None)?;
    let app_icon = logo_svg(&logo, 1024, logo.background)?;
    let mut raster = BTreeMap::new();

    for size in NATIVE_SIZES
        .iter()
        .chain(WEB_SIZES)
        .copied()
        .collect::<BTreeSet<_>>()
    {
        raster.insert(size, render_svg(&app_icon, size)?);
    }

    let mut assets = vec![
        text_asset("web/recursive-zoom-loader.svg", animated),
        text_asset("web/recursive-zoom-static.svg", static_mark),
        text_asset("web/favicon.svg", favicon),
        text_asset("web/site.webmanifest", web_manifest()),
        text_asset("native/app-icon.svg", app_icon),
    ];

    for &size in NATIVE_SIZES {
        assets.push(binary_asset(
            format!("native/app-icon-{size}.png"),
            raster[&size].png.clone(),
        ));
    }
    assets.push(binary_asset(
        "native/app-icon.ico",
        encode_ico(&raster, ICO_SIZES)?,
    ));
    assets.push(binary_asset("native/app-icon.icns", encode_icns(&raster)?));
    assets.push(binary_asset(
        "native/app-icon-256.rgba",
        raster[&256].rgba.clone(),
    ));

    assets.push(binary_asset(
        "web/apple-touch-icon.png",
        raster[&180].png.clone(),
    ));
    assets.push(binary_asset(
        "web/app-icon-192.png",
        raster[&192].png.clone(),
    ));
    assets.push(binary_asset(
        "web/app-icon-512.png",
        raster[&512].png.clone(),
    ));
    assets.push(binary_asset(
        "web/favicon.ico",
        encode_ico(&raster, ICO_SIZES)?,
    ));

    assets.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(assets)
}

fn text_asset(path: impl Into<String>, contents: String) -> Asset {
    Asset {
        relative_path: path.into(),
        bytes: contents.into_bytes(),
    }
}

fn binary_asset(path: impl Into<String>, bytes: Vec<u8>) -> Asset {
    Asset {
        relative_path: path.into(),
        bytes,
    }
}

#[derive(Debug)]
struct Raster {
    png: Vec<u8>,
    rgba: Vec<u8>,
}

fn render_svg(svg: &str, size: u32) -> Result<Raster, String> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default())
        .map_err(|error| format!("could not parse generated app icon SVG: {error}"))?;
    let mut pixmap = tiny_skia::Pixmap::new(size, size)
        .ok_or_else(|| format!("could not allocate {size}x{size} icon pixmap"))?;
    let scale = size as f32 / tree.size().width();
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    let png = pixmap
        .encode_png()
        .map_err(|error| format!("could not encode {size}x{size} icon PNG: {error}"))?;
    let rgba = pixmap.data().to_vec();
    if rgba.len() != size as usize * size as usize * 4 {
        return Err(format!(
            "generated {size}x{size} icon has an invalid RGBA length"
        ));
    }
    if rgba.chunks_exact(4).any(|pixel| pixel[3] != u8::MAX) {
        return Err(format!(
            "generated {size}x{size} application icon is not opaque"
        ));
    }
    Ok(Raster { png, rgba })
}

fn animated_loader_svg() -> String {
    const GEOMETRY_TIMES: &[f64] = &[0.0, 0.055, 0.43, 0.49, 0.95, 1.0];
    const OPACITY_TIMES: &[f64] = &[0.0, 0.57, 0.92, 1.0];
    let frames = GEOMETRY_TIMES
        .iter()
        .map(|phase| recursive_zoom(*phase))
        .collect::<Vec<_>>();
    let opacity_frames = OPACITY_TIMES
        .iter()
        .map(|phase| recursive_zoom(*phase))
        .collect::<Vec<_>>();
    let mut svg = loader_header("Animated recursive-zoom loading indicator");

    for layer_index in 0..frames[0].len() {
        let initial = &frames[0][layer_index];
        writeln!(
            svg,
            "      <path d=\"{}\" opacity=\"{}\">",
            path_data(&initial.points),
            number(initial.opacity)
        )
        .expect("writing to a String cannot fail");

        let paths = frames
            .iter()
            .map(|frame| path_data(&frame[layer_index].points))
            .collect::<Vec<_>>();
        writeln!(
            svg,
            "        <animate attributeName=\"d\" dur=\"{}s\" repeatCount=\"indefinite\" calcMode=\"spline\" keyTimes=\"{}\" keySplines=\"0 0 1 1;.5 0 .5 1;0 0 1 1;.5 0 .5 1;0 0 1 1\" values=\"{}\"/>",
            number(PERIOD_SECONDS),
            number_list(GEOMETRY_TIMES),
            paths.join(";")
        )
        .expect("writing to a String cannot fail");

        if layer_index != 1 {
            let opacity = opacity_frames
                .iter()
                .map(|frame| number(frame[layer_index].opacity))
                .collect::<Vec<_>>();
            writeln!(
                svg,
                "        <animate attributeName=\"opacity\" dur=\"{}s\" repeatCount=\"indefinite\" calcMode=\"spline\" keyTimes=\"{}\" keySplines=\"0 0 1 1;.5 0 .5 1;0 0 1 1\" values=\"{}\"/>",
                number(PERIOD_SECONDS),
                number_list(OPACITY_TIMES),
                opacity.join(";")
            )
            .expect("writing to a String cannot fail");
        }
        svg.push_str("      </path>\n");
    }
    svg.push_str("    </g>\n  </g>\n</svg>\n");
    svg
}

fn static_mark_svg(theme_aware: bool) -> String {
    let frame = recursive_zoom(STATIC_PHASE);
    let mut svg = String::from(
        "<!-- Generated by the braken-gui generate_app_icons example. -->\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"64\" height=\"64\" viewBox=\"0 0 100 100\" fill=\"none\" role=\"img\" aria-labelledby=\"title desc\">\n\
           <title id=\"title\">Braken recursive-zoom mark</title>\n\
           <desc id=\"desc\">A Hilbert curve rotated forty-five degrees.</desc>\n",
    );
    if theme_aware {
        svg.push_str(
            "  <style>.mark{stroke:#1f4e79}@media (prefers-color-scheme:dark){.mark{stroke:#d2e6ff}}</style>\n",
        );
    }
    write!(
        svg,
        "  <defs><clipPath id=\"frame\"><rect width=\"100\" height=\"100\"/></clipPath></defs>\n\
         <g transform=\"translate(50 50) rotate(45) scale({}) translate(-50 -50)\">\n\
         <g class=\"mark\" clip-path=\"url(#frame)\" fill=\"none\" stroke=\"#1f4e79\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n",
        number(LOADER_SCALE),
        number(LOADER_STROKE)
    )
    .expect("writing to a String cannot fail");
    write_static_paths(&mut svg, &frame);
    svg.push_str("    </g>\n  </g>\n</svg>\n");
    svg
}

fn generate_logo_scene() -> Result<Scene2d, String> {
    generate_logo_scene_at(logo_iterations()?)
}

fn generate_logo_scene_at(iterations: usize) -> Result<Scene2d, String> {
    let grammar = CompiledGrammar::parse(LOGO_SOURCE)
        .map_err(|error| format!("could not parse branding/app-icon.lsys: {error}"))?;
    let turn_angle = logo_numeric_metadata("Angle")?.to_radians();
    let calculation = calculate(CalculationRequest {
        grammar,
        iterations,
        backend: BackendChoice::Cpu,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits::unbounded_production(),
    })
    .map_err(|error| format!("could not derive branding/app-icon.lsys: {error}"))?;
    let visualization = visualize(VisualizeRequest {
        generation: &calculation.generation,
        backend: VisualizerBackend::Cpu,
        config: VisualizerConfig::Turtle2d(
            Turtle2dConfig {
                turn_angle,
                ..Turtle2dConfig::default()
            }
            .with_source_metadata(LOGO_SOURCE),
        ),
        context: VisualizationContext {
            iterations,
            seed: 0,
            derivation_backend: Some("cpu"),
            elapsed: None,
        },
    })
    .map_err(|error| format!("could not visualize branding/app-icon.lsys: {error}"))?;
    match visualization {
        Visualization::Scene2d(scene) => Ok(scene),
        Visualization::Scene3d(_) => {
            Err("branding/app-icon.lsys unexpectedly produced a 3D scene".into())
        }
    }
}

fn logo_iterations() -> Result<usize, String> {
    logo_iteration_metadata("Iterations")
}

fn logo_iteration_metadata(name: &str) -> Result<usize, String> {
    logo_metadata(name)?
        .parse()
        .map_err(|error| format!("branding/app-icon.lsys has invalid {name} metadata: {error}"))
}

fn logo_numeric_metadata(name: &str) -> Result<f64, String> {
    let value = logo_metadata(name)?
        .parse::<f64>()
        .map_err(|error| format!("branding/app-icon.lsys has invalid {name} metadata: {error}"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "branding/app-icon.lsys has non-finite {name} metadata"
        ))
    }
}

fn logo_metadata(name: &str) -> Result<&'static str, String> {
    let prefix = format!("# {name}:");
    LOGO_SOURCE
        .lines()
        .find_map(|line| line.trim().strip_prefix(&prefix))
        .map(str::trim)
        .ok_or_else(|| format!("branding/app-icon.lsys is missing {name} metadata"))
}

fn logo_svg(scene: &Scene2d, size: u32, background: Option<[u8; 3]>) -> Result<String, String> {
    let transform = logo_transform(scene)?;
    let mut svg = format!(
        "<!-- Generated from gui/branding/app-icon.lsys by the braken-gui generate_app_icons example. -->\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{size}\" height=\"{size}\" viewBox=\"0 0 100 100\" fill=\"none\" role=\"img\" aria-labelledby=\"title desc\">\n\
           <title id=\"title\">Braken frond mark</title>\n\
           <desc id=\"desc\">The Braken logo, a green curved bracken frond generated by an L-system.</desc>\n"
    );
    if let Some(color) = background {
        writeln!(
            svg,
            "  <rect width=\"100\" height=\"100\" fill=\"{}\"/>",
            rgb_hex(color)
        )
        .expect("writing to a String cannot fail");
    }

    // Fills are always written before strokes regardless of the source order
    // used by the scene's batch representation.
    for primitive in &scene.primitives {
        let Primitive2d::Polygon(polygon) = primitive else {
            continue;
        };
        let fill = exact_logo_color(polygon.color)?;
        let points = polygon
            .vertices
            .iter()
            .map(|&point| {
                let [x, y] = transform.map(point);
                format!("{},{}", number(x), number(y))
            })
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(
            svg,
            "  <polygon points=\"{points}\" fill=\"{}\"/>",
            rgb_hex(fill)
        )
        .expect("writing to a String cannot fail");
    }
    for primitive in &scene.primitives {
        let Primitive2d::Line(line) = primitive else {
            continue;
        };
        if line.width <= 0.0 {
            continue;
        }
        let [x1, y1] = transform.map(line.line.0);
        let [x2, y2] = transform.map(line.line.1);
        let stroke = exact_logo_color(line.color)?;
        writeln!(
            svg,
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
            number(x1),
            number(y1),
            number(x2),
            number(y2),
            rgb_hex(stroke),
            number(line.width * LOGO_BASE_STROKE)
        )
        .expect("writing to a String cannot fail");
    }
    svg.push_str("</svg>\n");
    Ok(svg)
}

impl LogoTransform {
    fn map(self, point: Point2d) -> Point {
        [
            50.0 + (point.0 - self.center.0) * self.scale,
            50.0 - (point.1 - self.center.1) * self.scale,
        ]
    }
}

fn logo_transform(scene: &Scene2d) -> Result<LogoTransform, String> {
    let mut bounds = None;
    let mut include = |point: Point2d| -> Result<(), String> {
        if !point.0.is_finite() || !point.1.is_finite() {
            return Err("branding/app-icon.lsys produced non-finite geometry".into());
        }
        bounds = Some(bounds.map_or(
            (point.0, point.0, point.1, point.1),
            |(min_x, max_x, min_y, max_y): (f64, f64, f64, f64)| {
                (
                    min_x.min(point.0),
                    max_x.max(point.0),
                    min_y.min(point.1),
                    max_y.max(point.1),
                )
            },
        ));
        Ok(())
    };
    for primitive in &scene.primitives {
        match primitive {
            Primitive2d::Line(line) => {
                include(line.line.0)?;
                include(line.line.1)?;
            }
            Primitive2d::Polygon(polygon) => {
                for &point in &polygon.vertices {
                    include(point)?;
                }
            }
            Primitive2d::Text(_) => {
                return Err("branding/app-icon.lsys unexpectedly produced text".into());
            }
        }
    }
    let (min_x, max_x, min_y, max_y) =
        bounds.ok_or("branding/app-icon.lsys produced no geometry")?;
    let extent = (max_x - min_x).max(max_y - min_y);
    if !extent.is_finite() || extent <= f64::EPSILON {
        return Err("branding/app-icon.lsys produced degenerate geometry".into());
    }
    Ok(LogoTransform {
        center: ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
        scale: LOGO_SPAN / extent,
    })
}

fn exact_logo_color(color: StrokeColor) -> Result<[u8; 3], String> {
    match color {
        StrokeColor::Rgb(rgb) => Ok(rgb),
        StrokeColor::ThemeDefault | StrokeColor::PaletteIndex(_) => Err(
            "branding/app-icon.lsys geometry must use exact colors from its Palette metadata"
                .into(),
        ),
    }
}

fn rgb_hex([red, green, blue]: [u8; 3]) -> String {
    format!("#{red:02x}{green:02x}{blue:02x}")
}

fn loader_header(description: &str) -> String {
    format!(
        "<!-- Generated by the braken-gui generate_app_icons example. -->\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"64\" height=\"64\" viewBox=\"0 0 100 100\" fill=\"none\" role=\"img\" aria-labelledby=\"title desc\">\n\
           <title id=\"title\">Loading Braken</title>\n\
           <desc id=\"desc\">{description}</desc>\n\
           <style>.mark{{stroke:#1f4e79}}@media (prefers-color-scheme:dark){{.mark{{stroke:#d2e6ff}}}}</style>\n\
           <defs><clipPath id=\"frame\"><rect width=\"100\" height=\"100\"/></clipPath></defs>\n\
           <g transform=\"translate(50 50) rotate(45) scale({}) translate(-50 -50)\">\n\
           <g class=\"mark\" clip-path=\"url(#frame)\" fill=\"none\" stroke=\"#1f4e79\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n",
        number(LOADER_SCALE),
        number(LOADER_STROKE)
    )
}

fn write_static_paths(svg: &mut String, frame: &[Layer]) {
    for layer in frame {
        writeln!(
            svg,
            "      <path d=\"{}\" opacity=\"{}\"/>",
            path_data(&layer.points),
            number(layer.opacity)
        )
        .expect("writing to a String cannot fail");
    }
}

fn recursive_zoom(phase: f64) -> Vec<Layer> {
    let h2 = hilbert(2);
    let h3 = hilbert(3);
    let refinement = ramp(phase, 0.055, 0.43);
    let zoom = ramp(phase, 0.49, 0.95);
    let scale = lerp(1.0, 7.0 / 3.0, zoom);
    let offset_y = lerp(0.0, -4.0 / 3.0, zoom);
    let points = h3
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let mixed = mix_point(h2[index / 4], *point, refinement);
            fit([mixed[0] * scale, mixed[1] * scale + offset_y])
        })
        .collect::<Vec<_>>();
    let outside_opacity = 1.0 - ramp(phase, 0.57, 0.92);
    let mut layers = (0..4)
        .map(|index| Layer {
            points: points[index * 16..(index + 1) * 16].to_vec(),
            opacity: if index == 1 { 1.0 } else { outside_opacity },
        })
        .collect::<Vec<_>>();
    layers.extend((0..3).map(|index| Layer {
        points: vec![points[index * 16 + 15], points[(index + 1) * 16]],
        opacity: outside_opacity,
    }));
    layers
}

fn hilbert(order: u32) -> Vec<Point> {
    let n = 1_u32 << order;
    (0..n * n)
        .map(|index| {
            let (mut x, mut y, mut t) = (0_u32, 0_u32, index);
            let mut scale = 1;
            while scale < n {
                let rx = 1 & (t >> 1);
                let ry = 1 & (t ^ rx);
                if ry == 0 {
                    if rx == 1 {
                        x = scale - 1 - x;
                        y = scale - 1 - y;
                    }
                    std::mem::swap(&mut x, &mut y);
                }
                x += scale * rx;
                y += scale * ry;
                t >>= 2;
                scale *= 2;
            }
            [
                f64::from(x) / f64::from(n - 1),
                f64::from(y) / f64::from(n - 1),
            ]
        })
        .collect()
}

fn ramp(phase: f64, start: f64, end: f64) -> f64 {
    let value = ((phase - start) / (end - start)).clamp(0.0, 1.0);
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn lerp(start: f64, end: f64, amount: f64) -> f64 {
    start + (end - start) * amount
}

fn mix_point(start: Point, end: Point, amount: f64) -> Point {
    [
        lerp(start[0], end[0], amount),
        lerp(start[1], end[1], amount),
    ]
}

fn fit(point: Point) -> Point {
    [PAD + point[0] * SPAN, PAD + point[1] * SPAN]
}

fn path_data(points: &[Point]) -> String {
    let mut path = String::new();
    for (index, point) in points.iter().enumerate() {
        write!(
            path,
            "{}{} {}",
            if index == 0 { 'M' } else { 'L' },
            number(point[0] * 100.0),
            number(point[1] * 100.0)
        )
        .expect("writing to a String cannot fail");
    }
    path
}

fn number_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| number(*value))
        .collect::<Vec<_>>()
        .join(";")
}

fn number(value: f64) -> String {
    let mut output = format!("{value:.4}");
    while output.ends_with('0') {
        output.pop();
    }
    if output.ends_with('.') {
        output.pop();
    }
    if output == "-0" {
        output = "0".to_owned();
    }
    output
}

fn web_manifest() -> String {
    "{\n  \"name\": \"Braken\",\n  \"short_name\": \"Braken\",\n  \"icons\": [\n    { \"src\": \"app-icon-192.png\", \"sizes\": \"192x192\", \"type\": \"image/png\", \"purpose\": \"any\" },\n    { \"src\": \"app-icon-512.png\", \"sizes\": \"512x512\", \"type\": \"image/png\", \"purpose\": \"any\" }\n  ],\n  \"theme_color\": \"#0f1420\",\n  \"background_color\": \"#f2f5f9\",\n  \"display\": \"standalone\"\n}\n".to_owned()
}

fn encode_ico(raster: &BTreeMap<u32, Raster>, sizes: &[u32]) -> Result<Vec<u8>, String> {
    let count = u16::try_from(sizes.len()).map_err(|_| "too many ICO images")?;
    let directory_len = 6_usize + sizes.len() * 16;
    let mut offset = u32::try_from(directory_len).map_err(|_| "ICO directory is too large")?;
    let mut output = Vec::new();
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&1_u16.to_le_bytes());
    output.extend_from_slice(&count.to_le_bytes());

    for &size in sizes {
        let png = &raster
            .get(&size)
            .ok_or_else(|| format!("missing {size}x{size} raster for ICO"))?
            .png;
        output.push(if size == 256 { 0 } else { size as u8 });
        output.push(if size == 256 { 0 } else { size as u8 });
        output.extend_from_slice(&[0, 0]);
        output.extend_from_slice(&1_u16.to_le_bytes());
        output.extend_from_slice(&32_u16.to_le_bytes());
        output.extend_from_slice(
            &u32::try_from(png.len())
                .map_err(|_| "ICO image is too large")?
                .to_le_bytes(),
        );
        output.extend_from_slice(&offset.to_le_bytes());
        offset = offset
            .checked_add(u32::try_from(png.len()).map_err(|_| "ICO image is too large")?)
            .ok_or("ICO size overflow")?;
    }
    for &size in sizes {
        output.extend_from_slice(&raster[&size].png);
    }
    Ok(output)
}

fn encode_icns(raster: &BTreeMap<u32, Raster>) -> Result<Vec<u8>, String> {
    let content_len = ICNS_SIZES.iter().try_fold(0_usize, |total, (size, _)| {
        total
            .checked_add(8)
            .and_then(|value| value.checked_add(raster.get(size)?.png.len()))
    });
    let total_len = content_len
        .and_then(|length| length.checked_add(8))
        .ok_or("missing ICNS raster or ICNS size overflow")?;
    let mut output = Vec::with_capacity(total_len);
    output.extend_from_slice(b"icns");
    output.extend_from_slice(
        &u32::try_from(total_len)
            .map_err(|_| "ICNS file is too large")?
            .to_be_bytes(),
    );
    for (size, kind) in ICNS_SIZES {
        let png = &raster[size].png;
        output.extend_from_slice(kind);
        output.extend_from_slice(
            &u32::try_from(png.len() + 8)
                .map_err(|_| "ICNS chunk is too large")?
                .to_be_bytes(),
        );
        output.extend_from_slice(png);
    }
    Ok(output)
}

fn check_outputs(output_dir: &Path, assets: &[Asset]) -> Result<(), String> {
    let expected = assets
        .iter()
        .map(|asset| output_dir.join(&asset.relative_path))
        .collect::<BTreeSet<_>>();
    let mut problems = Vec::new();
    for asset in assets {
        let path = output_dir.join(&asset.relative_path);
        match fs::read(&path) {
            Ok(actual) if actual == asset.bytes => {}
            Ok(_) => problems.push(format!("stale {}", path.display())),
            Err(error) => problems.push(format!("missing {}: {error}", path.display())),
        }
    }
    if output_dir.is_dir() {
        for path in files_recursive(output_dir)? {
            if !expected.contains(&path) {
                problems.push(format!("extra {}", path.display()));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

fn write_outputs(output_dir: &Path, assets: &[Asset]) -> Result<(), String> {
    let expected = assets
        .iter()
        .map(|asset| output_dir.join(&asset.relative_path))
        .collect::<BTreeSet<_>>();
    for asset in assets {
        let path = output_dir.join(&asset.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
        }
        if !fs::read(&path).is_ok_and(|contents| contents == asset.bytes) {
            fs::write(&path, &asset.bytes)
                .map_err(|error| format!("could not write {}: {error}", path.display()))?;
        }
    }
    if output_dir.is_dir() {
        for path in files_recursive(output_dir)? {
            if !expected.contains(&path) {
                fs::remove_file(&path)
                    .map_err(|error| format!("could not remove {}: {error}", path.display()))?;
            }
        }
    }
    Ok(())
}

fn files_recursive(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .map_err(|error| format!("could not read {}: {error}", current.display()))?
        {
            let path = entry
                .map_err(|error| format!("could not read {}: {error}", current.display()))?
                .path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursive_zoom_has_a_visual_loop_seam() {
        let start = recursive_zoom(0.0);
        let end = recursive_zoom(1.0);
        let expected = hilbert(2).into_iter().map(fit).collect::<Vec<_>>();
        assert_eq!(end[1].opacity, 1.0);
        assert!(
            end.iter()
                .enumerate()
                .all(|(index, layer)| { index == 1 || layer.opacity.abs() < f64::EPSILON })
        );
        assert_points_close(&end[1].points, &expected);

        let visible_start = start
            .iter()
            .take(4)
            .flat_map(|layer| layer.points.iter().copied())
            .collect::<Vec<_>>();
        let deduplicated = visible_start
            .into_iter()
            .fold(Vec::new(), |mut points, point| {
                if points.last() != Some(&point) {
                    points.push(point);
                }
                points
            });
        assert_points_close(&deduplicated, &expected);
    }

    #[test]
    fn generated_assets_have_expected_containers_and_dimensions() {
        let assets = generate_assets().expect("icons should generate");
        let by_name = assets
            .iter()
            .map(|asset| (asset.relative_path.as_str(), asset.bytes.as_slice()))
            .collect::<BTreeMap<_, _>>();
        assert!(by_name["web/recursive-zoom-loader.svg"].starts_with(b"<!-- Generated"));
        assert!(by_name["web/favicon.ico"].starts_with(&[0, 0, 1, 0]));
        assert!(by_name["native/app-icon.icns"].starts_with(b"icns"));
        assert_eq!(by_name["native/app-icon-256.rgba"].len(), 256 * 256 * 4);
        assert!(
            std::str::from_utf8(by_name["native/app-icon.svg"])
                .unwrap()
                .contains("<title id=\"title\">Braken frond mark</title>")
        );
        assert!(
            std::str::from_utf8(by_name["web/site.webmanifest"])
                .unwrap()
                .contains("\"name\": \"Braken\"")
        );
        for size in NATIVE_SIZES {
            let name = format!("native/app-icon-{size}.png");
            let png = by_name[name.as_str()];
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
            assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), *size);
            assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), *size);
        }
    }

    #[test]
    fn loader_marks_keep_their_strokes_inside_the_canvas() {
        let loader_extent =
            50.0 * LOADER_SCALE * std::f64::consts::SQRT_2 + LOADER_STROKE * LOADER_SCALE / 2.0;
        assert!(loader_extent < 50.0);
    }

    #[test]
    fn logo_grammar_is_an_upright_planar_spatial_fern() {
        let scene = generate_logo_scene().expect("logo grammar should visualize");
        let iterations = logo_iterations().unwrap();
        assert_eq!(iterations, 7);
        assert_eq!(logo_iteration_metadata("Max Iterations").unwrap(), 7);
        assert_eq!(logo_numeric_metadata("Angle").unwrap(), 4.0);
        assert_eq!(logo_numeric_metadata("Heading").unwrap(), 90.0);
        assert_eq!(scene.background, Some([0x0f, 0x14, 0x20]));

        let lines = scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive2d::Line(line) => Some(*line),
                Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scene.primitives.len(), lines.len());
        assert_eq!(
            lines.len(),
            5 * iterations * iterations + 3 * iterations + 4
        );
        assert!(lines.iter().all(|line| {
            line.color == StrokeColor::Rgb([0x57, 0xb8, 0x6a])
                && (line.width - 1.0).abs() < f64::EPSILON
                && (segment_length(line.line.0, line.line.1) - 1.0).abs() < 1.0e-10
        }));

        let (min_x, max_x, min_y, max_y) = lines
            .iter()
            .flat_map(|line| [line.line.0, line.line.1])
            .fold(
                (
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ),
                |(min_x, max_x, min_y, max_y), point| {
                    (
                        min_x.min(point.0),
                        max_x.max(point.0),
                        min_y.min(point.1),
                        max_y.max(point.1),
                    )
                },
            );
        assert!(max_y - min_y > max_x - min_x);
    }

    #[test]
    fn every_iteration_matches_the_spatial_fern_recurrence() {
        const SPATIAL_FERN_SOURCE: &str =
            include_str!("../presets/web/web-fern-3-d--8e92d92487.lsys");

        let logo_grammar = CompiledGrammar::parse(LOGO_SOURCE).expect("logo grammar should parse");
        let spatial_grammar =
            CompiledGrammar::parse(SPATIAL_FERN_SOURCE).expect("Spatial Fern grammar should parse");
        let max_iterations = logo_iteration_metadata("Max Iterations").unwrap();

        for iterations in 0..=max_iterations {
            let calculate_generation = |grammar: CompiledGrammar| {
                calculate(CalculationRequest {
                    grammar,
                    iterations,
                    backend: BackendChoice::Cpu,
                    seed: 0,
                    semantics: Default::default(),
                    limits: CalculationLimits::unbounded_production(),
                })
                .unwrap_or_else(|error| {
                    panic!("iteration {iterations} should derive on the CPU: {error}")
                })
                .generation
            };
            let logo_generation = calculate_generation(logo_grammar.clone());
            let spatial_generation = calculate_generation(spatial_grammar.clone());
            assert_eq!(
                logo_generation, spatial_generation,
                "iteration {iterations} changed the Spatial Fern recurrence"
            );

            let scene = generate_logo_scene_at(iterations)
                .unwrap_or_else(|error| panic!("iteration {iterations} should visualize: {error}"));
            let lines = scene
                .primitives
                .iter()
                .filter_map(|primitive| match primitive {
                    Primitive2d::Line(line) => Some(*line),
                    Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(scene.primitives.len(), lines.len());
            assert_eq!(
                lines.len(),
                5 * iterations * iterations + 3 * iterations + 4,
                "iteration {iterations}"
            );
            assert!(lines.iter().all(|line| {
                line.width.is_finite()
                    && line.width > 0.0
                    && line.color == StrokeColor::Rgb([0x57, 0xb8, 0x6a])
                    && line.line.0.0.is_finite()
                    && line.line.0.1.is_finite()
                    && line.line.1.0.is_finite()
                    && line.line.1.1.is_finite()
            }));
        }
    }

    #[test]
    fn sixteen_pixel_logo_retains_a_legible_branching_silhouette() {
        let scene = generate_logo_scene().expect("logo grammar should visualize");
        let svg = logo_svg(&scene, 1024, scene.background).expect("logo SVG should encode");
        let raster = render_svg(&svg, 16).expect("16-pixel logo should rasterize");
        let background = [0x0f, 0x14, 0x20];
        let frond = [0x57, 0xb8, 0x6a];

        let foreground = raster
            .rgba
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(index, pixel)| {
                (color_distance(pixel, frond) < color_distance(pixel, background))
                    .then_some((index % 16, index / 16))
            })
            .collect::<Vec<_>>();
        let frond_pixels = foreground.len();
        assert!(
            frond_pixels >= 16,
            "expected at least sixteen frond pixels, found {frond_pixels}"
        );
        assert!(
            frond_pixels <= 192,
            "the frond collapsed into a solid block of {frond_pixels} pixels"
        );

        let (min_x, max_x, min_y, max_y) = foreground.iter().copied().fold(
            (usize::MAX, 0, usize::MAX, 0),
            |(min_x, max_x, min_y, max_y), (x, y)| {
                (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
            },
        );
        assert!(
            max_x - min_x >= 7,
            "expected a branching silhouette at least eight pixels wide"
        );
        assert!(
            max_y - min_y >= 9,
            "expected an upright silhouette at least ten pixels tall"
        );
    }

    #[test]
    fn logo_safe_area_includes_its_strokes() {
        let scene = generate_logo_scene().expect("logo grammar should visualize");
        let transform = logo_transform(&scene).expect("logo should have finite bounds");
        for primitive in &scene.primitives {
            let Primitive2d::Line(line) = primitive else {
                continue;
            };
            let half_width = line.width * LOGO_BASE_STROKE * 0.5;
            for point in [line.line.0, line.line.1] {
                let [x, y] = transform.map(point);
                assert!(x - half_width >= 0.0);
                assert!(x + half_width <= 100.0);
                assert!(y - half_width >= 0.0);
                assert!(y + half_width <= 100.0);
            }
        }
    }

    fn segment_length(start: Point2d, end: Point2d) -> f64 {
        (end.0 - start.0).hypot(end.1 - start.1)
    }

    fn color_distance(pixel: &[u8], color: [u8; 3]) -> u32 {
        pixel[..3]
            .iter()
            .zip(color)
            .map(|(&actual, expected)| u32::from(actual.abs_diff(expected)).pow(2))
            .sum()
    }

    fn assert_points_close(actual: &[Point], expected: &[Point]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual[0] - expected[0]).abs() < 1e-10);
            assert!((actual[1] - expected[1]).abs() < 1e-10);
        }
    }
}
