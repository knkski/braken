use crate::{
    Point3d, Primitive2d, Primitive3d, Scene2d, Scene3d, StrokeColor, TextRole,
    Turtle3dWidthReferenceEstimator, normalized_turtle_3d_width,
    targets::{
        Palette, StrokeWidthEstimator, adaptive_scene_svg_stroke_width,
        adaptive_spatial_scene_svg_stroke_width, spatial_theme_default_rgb8_at, turtle_stroke_rgb,
    },
};

pub fn encode(scene: &Scene2d, palette: Palette) -> String {
    encode_with_stroke_scale(scene, palette, 1.0)
}

/// Encode retained three-dimensional geometry using the canonical static view.
pub fn encode_3d(scene: &Scene3d, palette: Palette) -> String {
    encode_prepared(
        &prepare_3d_scene(scene, palette),
        palette,
        1.0,
        StrokePolicy::Spatial,
    )
}

/// Encode retained three-dimensional geometry using the canonical static view
/// and a caller-selected multiplier for its automatic stroke width.
pub fn encode_3d_with_stroke_scale(scene: &Scene3d, palette: Palette, stroke_scale: f64) -> String {
    encode_prepared(
        &prepare_3d_scene(scene, palette),
        palette,
        stroke_scale,
        StrokePolicy::Spatial,
    )
}

/// Prepare retained 3D geometry for the static SVG target without changing the
/// generic 2D encoder's styling contract.
fn prepare_3d_scene(scene: &Scene3d, palette: Palette) -> Scene2d {
    let bounds = SourceBounds3d::for_scene(scene);
    let mut width_estimator = Turtle3dWidthReferenceEstimator::default();
    for primitive in &scene.primitives {
        if let Primitive3d::Line(line) = primitive {
            width_estimator.observe_line(line);
        }
    }
    let width_reference = width_estimator.width_reference();
    let mut projected = scene.canonical_projection();

    for (source, target) in scene.primitives.iter().zip(&mut projected.primitives) {
        match (source, target) {
            (Primitive3d::Line(source), Primitive2d::Line(target)) => {
                target.width = normalized_turtle_3d_width(source.width, width_reference);
                target.color = spatial_svg_color(
                    source.color,
                    bounds.palette_position(line_midpoint(source.line.0, source.line.1)),
                    palette,
                );
            }
            (Primitive3d::Polygon(source), Primitive2d::Polygon(target)) => {
                let position = polygon_centroid(&source.vertices)
                    .map_or(0.5, |centroid| bounds.palette_position(centroid));
                target.color = spatial_svg_color(source.color, position, palette);
            }
            _ => unreachable!("canonical projection changed primitive kind or order"),
        }
    }

    projected
}

fn spatial_svg_color(color: StrokeColor, position: f64, palette: Palette) -> StrokeColor {
    match color {
        StrokeColor::ThemeDefault => {
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(position, palette))
        }
        StrokeColor::PaletteIndex(_) | StrokeColor::Rgb(_) => color,
    }
}

fn line_midpoint(start: Point3d, end: Point3d) -> Point3d {
    (
        finite_midpoint(start.0, end.0),
        finite_midpoint(start.1, end.1),
        finite_midpoint(start.2, end.2),
    )
}

fn polygon_centroid(vertices: &[Point3d]) -> Option<Point3d> {
    let mut centroid = None;
    for (index, &point) in vertices.iter().enumerate() {
        if !point_is_finite(point) {
            continue;
        }
        let Some((x, y, z)) = centroid.as_mut() else {
            centroid = Some(point);
            continue;
        };
        let weight = 1.0 / (index + 1) as f64;
        *x += (point.0 - *x) * weight;
        *y += (point.1 - *y) * weight;
        *z += (point.2 - *z) * weight;
    }
    centroid
}

#[derive(Debug, Clone, Copy)]
struct SourceBounds3d {
    min: Point3d,
    max: Point3d,
}

impl SourceBounds3d {
    fn for_scene(scene: &Scene3d) -> Self {
        let mut bounds = None;
        for primitive in &scene.primitives {
            match primitive {
                Primitive3d::Line(line) => {
                    include_point(&mut bounds, line.line.0);
                    include_point(&mut bounds, line.line.1);
                }
                Primitive3d::Polygon(polygon) => {
                    for &point in &polygon.vertices {
                        include_point(&mut bounds, point);
                    }
                }
            }
        }
        bounds.unwrap_or(Self {
            min: (0.0, 0.0, 0.0),
            max: (0.0, 0.0, 0.0),
        })
    }

    fn palette_position(self, point: Point3d) -> f64 {
        let x = normalized_axis(point.0, self.min.0, self.max.0);
        let y = normalized_axis(point.1, self.min.1, self.max.1);
        let z = normalized_axis(point.2, self.min.2, self.max.2);
        (0.45 * x + 0.35 * (1.0 - y) + 0.20 * z).clamp(0.0, 1.0)
    }
}

fn include_point(bounds: &mut Option<SourceBounds3d>, point: Point3d) {
    if !point_is_finite(point) {
        return;
    }
    let Some(bounds) = bounds else {
        *bounds = Some(SourceBounds3d {
            min: point,
            max: point,
        });
        return;
    };
    bounds.min.0 = bounds.min.0.min(point.0);
    bounds.min.1 = bounds.min.1.min(point.1);
    bounds.min.2 = bounds.min.2.min(point.2);
    bounds.max.0 = bounds.max.0.max(point.0);
    bounds.max.1 = bounds.max.1.max(point.1);
    bounds.max.2 = bounds.max.2.max(point.2);
}

fn point_is_finite(point: Point3d) -> bool {
    point.0.is_finite() && point.1.is_finite() && point.2.is_finite()
}

fn normalized_axis(value: f64, minimum: f64, maximum: f64) -> f64 {
    const MIN_DRAWING_EXTENT: f64 = 0.1;

    let extent = maximum - minimum;
    let extent = if extent.is_finite() {
        extent.max(MIN_DRAWING_EXTENT)
    } else {
        MIN_DRAWING_EXTENT
    };
    finite_or((value - minimum) / extent, 0.5).clamp(0.0, 1.0)
}

fn finite_midpoint(first: f64, second: f64) -> f64 {
    finite_or(first * 0.5 + second * 0.5, 0.0)
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

/// Encodes `scene` while multiplying its automatic stroke width by
/// `stroke_scale`. Non-positive or non-finite scales use the automatic width.
pub fn encode_with_stroke_scale(scene: &Scene2d, palette: Palette, stroke_scale: f64) -> String {
    encode_prepared(scene, palette, stroke_scale, StrokePolicy::Planar)
}

#[derive(Debug, Clone, Copy)]
enum StrokePolicy {
    Planar,
    Spatial,
}

fn encode_prepared(
    scene: &Scene2d,
    palette: Palette,
    stroke_scale: f64,
    stroke_policy: StrokePolicy,
) -> String {
    let (themed_background, foreground, muted) = match palette {
        Palette::Light => ("#f2f5f9", "#263247", "#687386"),
        Palette::Dark => ("#0f1420", "#eef2f8", "#aab4c3"),
    };
    let background = scene
        .background
        .map(rgb_hex)
        .unwrap_or_else(|| themed_background.to_owned());
    let mut width_estimator = StrokeWidthEstimator::default();
    let mut lines = scene
        .primitives
        .iter()
        .filter_map(|item| match item {
            Primitive2d::Line(line) => {
                width_estimator.observe(line.line);
                Some(line.line)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for primitive in &scene.primitives {
        if let Primitive2d::Polygon(polygon) = primitive {
            for edge in polygon.vertices.windows(2) {
                lines.push(crate::Line2d(edge[0], edge[1]));
            }
            if let (Some(&first), Some(&last)) = (polygon.vertices.first(), polygon.vertices.last())
            {
                lines.push(crate::Line2d(last, first));
            }
        }
    }
    let (view, drawing_extent) = if lines.is_empty() {
        ((0.0, 0.0, 1000.0, 700.0), (0.0, 0.0))
    } else {
        let (min_x, max_x, min_y, max_y) = lines.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(a, b, c, d), line| {
                (
                    a.min(line.0.0).min(line.1.0),
                    b.max(line.0.0).max(line.1.0),
                    c.min(-line.0.1).min(-line.1.1),
                    d.max(-line.0.1).max(-line.1.1),
                )
            },
        );
        let drawing_width = (max_x - min_x).max(0.0);
        let drawing_height = (max_y - min_y).max(0.0);
        let width = drawing_width.max(0.1);
        let height = drawing_height.max(0.1);
        let margin = width.max(height) * 0.04;
        (
            (
                min_x - margin,
                min_y - margin,
                width + margin * 2.0,
                height + margin * 2.0,
            ),
            (drawing_width, drawing_height),
        )
    };
    let view_extent = view.2.max(view.3);
    let stroke_scale = if stroke_scale.is_finite() && stroke_scale > 0.0 {
        stroke_scale
    } else {
        1.0
    };
    let automatic_stroke_width = match stroke_policy {
        StrokePolicy::Planar => adaptive_scene_svg_stroke_width(
            width_estimator.total_line_length(),
            width_estimator.line_count(),
            drawing_extent,
            view_extent,
        ),
        StrokePolicy::Spatial => adaptive_spatial_scene_svg_stroke_width(
            width_estimator.total_line_length(),
            width_estimator.line_count(),
            drawing_extent,
            view_extent,
        ),
    };
    let stroke_width = automatic_stroke_width * stroke_scale;
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{background}\"/>\n",
        view.0, view.1, view.2, view.3, view.0, view.1, view.2, view.3
    );
    // Fills form the backdrop; drawing them first keeps later turtle strokes
    // visible regardless of batching or polygon-close order.
    for primitive in &scene.primitives {
        if let Primitive2d::Polygon(polygon) = primitive {
            let fill = turtle_stroke_rgb(polygon.color, palette)
                .map(rgb_hex)
                .unwrap_or_else(|| foreground.to_owned());
            let points = polygon
                .vertices
                .iter()
                .map(|point| format!("{},{}", point.0, -point.1))
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!(
                "<polygon points=\"{points}\" fill=\"{fill}\" stroke=\"none\"/>\n"
            ));
        }
    }
    for primitive in &scene.primitives {
        match primitive {
            Primitive2d::Line(styled) => {
                let line = styled.line;
                let line_width = stroke_width * styled.width;
                let stroke = turtle_stroke_rgb(styled.color, palette)
                    .map(rgb_hex)
                    .unwrap_or_else(|| foreground.to_owned());
                out.push_str(&format!("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{line_width}\" stroke-linecap=\"round\"/>\n", line.0.0, -line.0.1, line.1.0, -line.1.1));
            }
            Primitive2d::Polygon(_) => {}
            Primitive2d::Text(text) => {
                let color = if text.role == TextRole::Muted {
                    muted
                } else {
                    foreground
                };
                out.push_str(&format!("<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"{}\" fill=\"{color}\">{}</text>\n", text.position.0, text.position.1, text.size, escape(&text.content)));
            }
        }
    }
    out.push_str("</svg>\n");
    out
}

fn rgb_hex([red, green, blue]: [u8; 3]) -> String {
    format!("#{red:02x}{green:02x}{blue:02x}")
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Line2d, Line3d, Lines2d, Polygon3d, StyledLine3d};

    fn projected_line(scene: &Scene2d, index: usize) -> &crate::StyledLine2d {
        let Primitive2d::Line(line) = &scene.primitives[index] else {
            panic!("primitive {index} was not a line");
        };
        line
    }

    fn first_stroke_width(svg: &str) -> f64 {
        svg.split("stroke-width=\"")
            .nth(1)
            .and_then(|value| value.split('"').next())
            .expect("SVG line stroke width")
            .parse::<f64>()
            .expect("numeric SVG line stroke width")
    }

    #[test]
    fn requested_stroke_scale_multiplies_the_automatic_width() {
        let scene = Scene2d::lines(Lines2d(vec![Line2d((0.0, 0.0), (10.0, 0.0))]));
        let automatic = encode(&scene, Palette::Light);
        let thin = encode_with_stroke_scale(&scene, Palette::Light, 0.05);

        assert!(
            (first_stroke_width(&thin) / first_stroke_width(&automatic) - 0.05).abs() < 1.0e-12
        );
    }

    #[test]
    fn spatial_svg_uses_the_narrower_rod_base_without_changing_planar_svg() {
        let scene = Scene3d {
            primitives: vec![Primitive3d::Line(StyledLine3d {
                line: Line3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };
        let planar = encode(&scene.canonical_projection(), Palette::Light);
        let spatial = encode_3d(&scene, Palette::Light);

        assert!(
            (first_stroke_width(&spatial) / first_stroke_width(&planar) - 0.25).abs() < 1.0e-12
        );
    }

    #[test]
    fn spatial_defaults_use_source_aabb_positions_and_theme_palette() {
        let scene = Scene3d {
            primitives: vec![
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
                    width: 1.0,
                    color: StrokeColor::ThemeDefault,
                }),
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
                    width: 1.0,
                    color: StrokeColor::ThemeDefault,
                }),
            ],
            background: None,
        };

        let light = prepare_3d_scene(&scene, Palette::Light);
        let dark = prepare_3d_scene(&scene, Palette::Dark);
        assert_eq!(
            projected_line(&light, 0).color,
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(0.0, Palette::Light))
        );
        assert_eq!(
            projected_line(&light, 1).color,
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(1.0, Palette::Light))
        );
        assert_eq!(
            projected_line(&dark, 0).color,
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(0.0, Palette::Dark))
        );
        assert_eq!(
            projected_line(&dark, 1).color,
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(1.0, Palette::Dark))
        );
    }

    #[test]
    fn spatial_palette_coordinate_matches_camera_degenerate_axis_behavior() {
        let scene = Scene3d {
            primitives: vec![Primitive3d::Line(StyledLine3d {
                line: Line3d((2.0, 3.0, 4.0), (2.0, 3.0, 4.0)),
                width: 1.0,
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };

        let projected = prepare_3d_scene(&scene, Palette::Light);
        assert_eq!(
            projected_line(&projected, 0).color,
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(0.35, Palette::Light))
        );
    }

    #[test]
    fn spatial_polygons_use_their_source_centroid_for_default_color() {
        let scene = Scene3d {
            primitives: vec![Primitive3d::Polygon(Polygon3d {
                vertices: vec![
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0),
                    (0.0, 1.0, 1.0),
                ],
                color: StrokeColor::ThemeDefault,
            })],
            background: None,
        };

        let projected = prepare_3d_scene(&scene, Palette::Light);
        let Primitive2d::Polygon(polygon) = &projected.primitives[0] else {
            panic!("primitive was not a polygon");
        };
        assert_eq!(
            polygon.color,
            StrokeColor::Rgb(spatial_theme_default_rgb8_at(0.5, Palette::Light))
        );
    }

    #[test]
    fn spatial_projection_preserves_non_default_colors() {
        let scene = Scene3d {
            primitives: vec![
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                    width: 1.0,
                    color: StrokeColor::Rgb([1, 2, 3]),
                }),
                Primitive3d::Polygon(Polygon3d {
                    vertices: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    color: StrokeColor::PaletteIndex(7),
                }),
            ],
            background: None,
        };

        let projected = prepare_3d_scene(&scene, Palette::Dark);
        assert_eq!(
            projected_line(&projected, 0).color,
            StrokeColor::Rgb([1, 2, 3])
        );
        let Primitive2d::Polygon(polygon) = &projected.primitives[1] else {
            panic!("second primitive was not a polygon");
        };
        assert_eq!(polygon.color, StrokeColor::PaletteIndex(7));
    }

    #[test]
    fn spatial_widths_use_a_source_length_weighted_reference() {
        let scene = Scene3d {
            primitives: vec![
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                    width: 80.0,
                    color: StrokeColor::ThemeDefault,
                }),
                Primitive3d::Line(StyledLine3d {
                    line: Line3d((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                    width: 100.0,
                    color: StrokeColor::ThemeDefault,
                }),
            ],
            background: None,
        };
        let expected_reference = (80.0 + 2.0_f64.sqrt() * 100.0) / (1.0 + 2.0_f64.sqrt());

        let projected = prepare_3d_scene(&scene, Palette::Light);
        assert!((projected_line(&projected, 0).width - 80.0 / expected_reference).abs() < 1.0e-12);
        assert!((projected_line(&projected, 1).width - 100.0 / expected_reference).abs() < 1.0e-12);
        let canonical = scene.canonical_projection();
        assert_eq!(
            projected_line(&projected, 0).line,
            projected_line(&canonical, 0).line
        );
    }
}
