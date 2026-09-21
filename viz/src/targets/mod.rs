pub mod svg;
pub mod tui;

const TARGET_INK_COVERAGE: f64 = 0.65;
const MAXIMUM_FITTED_WIDTH: f64 = 10.0;
const SPARSE_PATH_DIAGONALS: f64 = 2.0;

/// Largest automatic base width, in fitted logical pixels, for spatial rods.
///
/// Source width commands are applied after this cap through
/// [`crate::normalized_turtle_3d_width`]. Keeping the base narrower than the
/// planar sparse-path maximum prevents open plant skeletons from becoming
/// solid ribbons while leaving already-detailed curves unchanged.
pub const MAX_SPATIAL_FITTED_STROKE_WIDTH: f64 = 2.5;

/// Streaming summary of the unstyled length of rendered line segments.
///
/// Stroke styles are deliberately excluded: the summary describes geometry,
/// while [`crate::StyledLine2d::width`] remains a multiplier applied by the
/// output target. Zero-length and non-finite segments do not contribute.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StrokeWidthEstimator {
    total_line_length: f64,
    line_count: usize,
}

impl StrokeWidthEstimator {
    /// Include one segment in the total unstyled path length.
    pub fn observe(&mut self, line: crate::Line2d) {
        let length = (line.1.0 - line.0.0).hypot(line.1.1 - line.0.1);
        if !length.is_finite() || length <= 0.0 {
            return;
        }

        let total = self.total_line_length + length;
        self.total_line_length = if total.is_finite() { total } else { f64::MAX };
        self.line_count = self.line_count.saturating_add(1);
    }

    /// Sum of the observed positive, finite segment lengths.
    pub fn total_line_length(self) -> Option<f64> {
        (self.total_line_length > 0.0).then_some(self.total_line_length)
    }

    /// Number of observed positive, finite segments.
    pub fn line_count(self) -> usize {
        self.line_count
    }
}

/// Select an automatic, unstyled stroke width in fitted target units.
///
/// For a two-dimensional drawing, `area / total_line_length` estimates the
/// average distance between neighboring tracks. The default stroke consumes
/// 65% of that distance, leaving a visible gap where the output resolution can
/// represent one. Sparse and degenerate paths use the chunky fitted maximum.
/// There is deliberately no positive minimum: dense scenes need fractional
/// pixel coverage instead of overlapping opaque one-pixel strokes.
///
/// Callers apply camera zoom and per-line style after this fitted width so
/// strokes scale geometrically during navigation and explicit turtle width
/// commands retain their relative meaning.
pub fn adaptive_stroke_width(
    total_line_length: Option<f64>,
    drawing_extent: (f64, f64),
    fitted_scale: f64,
) -> f64 {
    if !fitted_scale.is_finite() || fitted_scale <= 0.0 {
        return MAXIMUM_FITTED_WIDTH;
    }
    let Some(total_line_length) =
        total_line_length.filter(|length| length.is_finite() && *length > 0.0)
    else {
        return MAXIMUM_FITTED_WIDTH;
    };
    let (drawing_width, drawing_height) = drawing_extent;
    if !drawing_width.is_finite()
        || drawing_width < 0.0
        || !drawing_height.is_finite()
        || drawing_height < 0.0
    {
        return MAXIMUM_FITTED_WIDTH;
    }

    let diagonal = drawing_width.hypot(drawing_height);
    if diagonal <= 0.0 || total_line_length <= diagonal * SPARSE_PATH_DIAGONALS {
        return MAXIMUM_FITTED_WIDTH;
    }
    let drawing_area = drawing_width * drawing_height;
    if !drawing_area.is_finite() || drawing_area <= 0.0 {
        return MAXIMUM_FITTED_WIDTH;
    }

    let fitted_width = drawing_area / total_line_length * fitted_scale * TARGET_INK_COVERAGE;
    if fitted_width.is_finite() {
        fitted_width.clamp(0.0, MAXIMUM_FITTED_WIDTH)
    } else {
        MAXIMUM_FITTED_WIDTH
    }
}

/// Select an automatic stroke width using both path density and scene detail.
///
/// [`adaptive_stroke_width`] estimates the separation between neighboring
/// tracks, but that separation alone can stay large when a recursive drawing
/// expands along with its path. This scene-level variant adds two caps: 65% of
/// `fitted_long_edge / sqrt(line_count)`, and a cap derived from the fitted
/// mean segment length. The first models two-dimensional spacing without
/// assuming an axis-aligned grammar. The second follows the actual scale of the
/// finest rendered steps, preventing lower-dimensional curves such as Koch
/// outlines from receiving strokes much wider than their recursive detail.
/// Once the mean step is below one logical pixel, its square root balances
/// aggregate coverage against detail that the target cannot resolve exactly;
/// the width still approaches zero as the projected step shrinks.
///
/// Taking the narrowest estimate preserves the density behavior for
/// overlapping tilings while preventing high-detail scenes from being
/// dominated by round caps. Empty and one-dimensional scenes retain the
/// density-only result.
pub fn adaptive_scene_stroke_width(
    total_line_length: Option<f64>,
    line_count: usize,
    drawing_extent: (f64, f64),
    fitted_scale: f64,
) -> f64 {
    let density_width = adaptive_stroke_width(total_line_length, drawing_extent, fitted_scale);
    let (drawing_width, drawing_height) = drawing_extent;
    if line_count == 0
        || !fitted_scale.is_finite()
        || fitted_scale <= 0.0
        || !drawing_width.is_finite()
        || drawing_width <= 0.0
        || !drawing_height.is_finite()
        || drawing_height <= 0.0
    {
        return density_width;
    }

    let fitted_long_edge = drawing_width.max(drawing_height) * fitted_scale;
    let detail_width = fitted_long_edge * TARGET_INK_COVERAGE / (line_count as f64).sqrt();
    let mut fitted_width = if detail_width.is_finite() && detail_width > 0.0 {
        density_width.min(detail_width)
    } else {
        density_width
    };
    if let Some(total_line_length) =
        total_line_length.filter(|length| length.is_finite() && *length > 0.0)
    {
        let fitted_mean_segment_length = total_line_length / line_count as f64 * fitted_scale;
        let resolvable_step = if fitted_mean_segment_length < 1.0 {
            fitted_mean_segment_length.sqrt()
        } else {
            fitted_mean_segment_length
        };
        let segment_width = resolvable_step * TARGET_INK_COVERAGE;
        if segment_width.is_finite() && segment_width > 0.0 {
            fitted_width = fitted_width.min(segment_width);
        }
    }
    fitted_width
}

/// Select the automatic base width for a fitted three-dimensional line scene.
///
/// This retains the same density/detail response as
/// [`adaptive_scene_stroke_width`] but caps sparse spatial skeletons at a
/// rod-like width. Scene-relative turtle width multipliers and the user's
/// display scale are applied separately by the renderer.
pub fn adaptive_spatial_scene_stroke_width(
    total_line_length: Option<f64>,
    line_count: usize,
    drawing_extent: (f64, f64),
    fitted_scale: f64,
) -> f64 {
    adaptive_scene_stroke_width(total_line_length, line_count, drawing_extent, fitted_scale)
        .min(MAX_SPATIAL_FITTED_STROKE_WIDTH)
}

/// Select an automatic stroke width expressed in SVG view-box units.
///
/// SVG has no fixed display resolution, so targets use a nominal 500-unit long
/// edge to apply the same fitted-unit policy as raster displays.
pub fn adaptive_svg_stroke_width(
    total_line_length: Option<f64>,
    drawing_extent: (f64, f64),
    view_extent: f64,
) -> f64 {
    const NOMINAL_LONG_EDGE: f64 = 500.0;

    if !view_extent.is_finite() || view_extent <= 0.0 {
        return 10.0;
    }
    let nominal_scale = NOMINAL_LONG_EDGE / view_extent;
    adaptive_stroke_width(total_line_length, drawing_extent, nominal_scale) / nominal_scale
}

/// Select a scene-detail-aware automatic width in SVG view-box units.
pub fn adaptive_scene_svg_stroke_width(
    total_line_length: Option<f64>,
    line_count: usize,
    drawing_extent: (f64, f64),
    view_extent: f64,
) -> f64 {
    const NOMINAL_LONG_EDGE: f64 = 500.0;

    if !view_extent.is_finite() || view_extent <= 0.0 {
        return MAXIMUM_FITTED_WIDTH;
    }
    let nominal_scale = NOMINAL_LONG_EDGE / view_extent;
    adaptive_scene_stroke_width(total_line_length, line_count, drawing_extent, nominal_scale)
        / nominal_scale
}

/// Select a spatial automatic base width in SVG view-box units.
///
/// The spatial cap is converted through the same nominal 500-unit fitted edge
/// used by [`adaptive_scene_svg_stroke_width`].
pub fn adaptive_spatial_scene_svg_stroke_width(
    total_line_length: Option<f64>,
    line_count: usize,
    drawing_extent: (f64, f64),
    view_extent: f64,
) -> f64 {
    const NOMINAL_LONG_EDGE: f64 = 500.0;

    let width =
        adaptive_scene_svg_stroke_width(total_line_length, line_count, drawing_extent, view_extent);
    if !view_extent.is_finite() || view_extent <= 0.0 {
        return width.min(MAX_SPATIAL_FITTED_STROKE_WIDTH);
    }
    width.min(MAX_SPATIAL_FITTED_STROKE_WIDTH * view_extent / NOMINAL_LONG_EDGE)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TargetKind {
    #[default]
    Auto,
    Tui,
    Svg,
    Png,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    Light,
    Dark,
}

/// Number of discrete colors in the theme-default spatial stroke ramp.
///
/// Keeping this bounded lets software renderers batch nearby colors while the
/// endpoints remain identical across live displays, SVG exports, and preset
/// previews.
pub const SPATIAL_THEME_DEFAULT_COLOR_BUCKETS: usize = 48;

/// Maximum opaque background fog applied to the far side of a spatial scene.
pub const SPATIAL_FAR_FOG: f64 = 0.18;
/// Minimum opaque background fog applied to the near side of a spatial scene.
pub const SPATIAL_NEAR_FOG: f64 = 0.02;
/// Directional-light levels used by software renderers and vector exports.
pub const SPATIAL_SOFTWARE_LIGHT_BUCKETS: usize = 8;
/// Near-to-far levels used by software renderers and vector exports.
pub const SPATIAL_SOFTWARE_DEPTH_BUCKETS: usize = 16;

const LIGHT_SPATIAL_START: [u8; 3] = [0x06, 0x4e, 0x3b];
const LIGHT_SPATIAL_END: [u8; 3] = [0x20, 0x76, 0x42];
const DARK_SPATIAL_START: [u8; 3] = [0x4a, 0xde, 0x80];
const DARK_SPATIAL_END: [u8; 3] = [0xa7, 0xf3, 0xd0];
const SPATIAL_SOFTWARE_MIN_LIGHT: f64 = 0.68;

/// Convert a normalized spatial palette position into its bounded bucket.
///
/// Non-finite positions select the middle of the ramp. The final bucket is
/// reachable at exactly `1.0`, so both declared emerald endpoints are retained.
pub fn spatial_theme_default_bucket(position: f64) -> usize {
    let position = if position.is_finite() {
        position.clamp(0.0, 1.0)
    } else {
        0.5
    };
    ((position * SPATIAL_THEME_DEFAULT_COLOR_BUCKETS as f64).floor() as usize)
        .min(SPATIAL_THEME_DEFAULT_COLOR_BUCKETS - 1)
}

/// Return one theme-default spatial color as normalized sRGB components.
///
/// The emerald endpoints are interpolated in linear-light RGB rather than
/// encoded sRGB, avoiding the muddy midpoint produced by component-wise byte
/// interpolation. This palette applies only when a 3D renderer resolves
/// [`crate::StrokeColor::ThemeDefault`]; explicit turtle colors are unaffected.
pub fn spatial_theme_default_color(bucket: usize, palette: Palette) -> [f32; 3] {
    let bucket = bucket.min(SPATIAL_THEME_DEFAULT_COLOR_BUCKETS - 1);
    spatial_theme_default_colors(palette)[bucket]
}

/// Resolve a normalized position on the theme-default spatial color ramp.
pub fn spatial_theme_default_color_at(position: f64, palette: Palette) -> [f32; 3] {
    spatial_theme_default_color(spatial_theme_default_bucket(position), palette)
}

/// Resolve a normalized spatial palette position to SVG/CSS byte components.
pub fn spatial_theme_default_rgb8_at(position: f64, palette: Palette) -> [u8; 3] {
    spatial_theme_default_color_at(position, palette)
        .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Apply shared opaque rod lighting and depth fog to a spatial stroke color.
///
/// `light` is the continuous ambient-plus-directional response returned by the
/// shared camera, and `near_depth` runs from `0.0` at the back of the fitted
/// sphere to `1.0` at its front. Shading and fog composition happen in linear
/// light before the result is converted back to normalized sRGB. Invalid input
/// is replaced with finite, bounded defaults; alpha is deliberately absent so
/// depth-buffered and painter-sorted targets agree.
pub fn spatial_lit_color(
    base: [f32; 3],
    background: [f32; 3],
    light: f64,
    near_depth: f64,
) -> [f32; 3] {
    let base = base.map(finite_unit_or_zero);
    let background = background.map(finite_unit_or_zero);
    let light = finite_unit_or(light, 1.0) as f32;
    let near_depth = finite_unit_or(near_depth, 0.5);
    let fog = SPATIAL_FAR_FOG + (SPATIAL_NEAR_FOG - SPATIAL_FAR_FOG) * near_depth;
    let fog = fog as f32;
    std::array::from_fn(|channel| {
        let lit = (srgb_to_linear(base[channel]) * light).clamp(0.0, 1.0);
        let background = srgb_to_linear(background[channel]);
        linear_to_srgb(lit + (background - lit) * fog)
    })
}

/// Apply spatial lighting after bounding it to the shared software/export grid.
///
/// Quantizing the camera's continuous samples to eight light levels over
/// `[0.68, 1.0]` and sixteen near-depth levels over `[0.0, 1.0]` bounds the
/// number of stroke colors. Canvas renderers can consequently retain path
/// batching, and separately implemented SVG encoders produce identical colors.
/// WGPU shaders should continue to mirror [`spatial_lit_color`] continuously.
pub fn spatial_lit_color_bounded(
    base: [f32; 3],
    background: [f32; 3],
    light: f64,
    near_depth: f64,
) -> [f32; 3] {
    let light = quantize_to_endpoints(
        finite_bounded_or(light, SPATIAL_SOFTWARE_MIN_LIGHT, 1.0, 1.0),
        SPATIAL_SOFTWARE_MIN_LIGHT,
        1.0,
        SPATIAL_SOFTWARE_LIGHT_BUCKETS,
    );
    let near_depth = quantize_to_endpoints(
        finite_bounded_or(near_depth, 0.0, 1.0, 0.5),
        0.0,
        1.0,
        SPATIAL_SOFTWARE_DEPTH_BUCKETS,
    );
    spatial_lit_color(base, background, light, near_depth)
}

fn finite_unit_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

fn finite_bounded_or(value: f64, minimum: f64, maximum: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(minimum, maximum)
    } else {
        fallback
    }
}

fn quantize_to_endpoints(value: f64, minimum: f64, maximum: f64, buckets: usize) -> f64 {
    debug_assert!(buckets >= 2);
    let intervals = (buckets - 1) as f64;
    let normalized = ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0);
    minimum + (normalized * intervals).round() / intervals * (maximum - minimum)
}

fn finite_unit_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn spatial_theme_default_colors(
    palette: Palette,
) -> &'static [[f32; 3]; SPATIAL_THEME_DEFAULT_COLOR_BUCKETS] {
    static LIGHT: std::sync::OnceLock<[[f32; 3]; SPATIAL_THEME_DEFAULT_COLOR_BUCKETS]> =
        std::sync::OnceLock::new();
    static DARK: std::sync::OnceLock<[[f32; 3]; SPATIAL_THEME_DEFAULT_COLOR_BUCKETS]> =
        std::sync::OnceLock::new();
    match palette {
        Palette::Light => {
            LIGHT.get_or_init(|| spatial_color_ramp(LIGHT_SPATIAL_START, LIGHT_SPATIAL_END))
        }
        Palette::Dark => {
            DARK.get_or_init(|| spatial_color_ramp(DARK_SPATIAL_START, DARK_SPATIAL_END))
        }
    }
}

fn spatial_color_ramp(
    start: [u8; 3],
    end: [u8; 3],
) -> [[f32; 3]; SPATIAL_THEME_DEFAULT_COLOR_BUCKETS] {
    std::array::from_fn(|bucket| {
        let position = bucket as f32 / (SPATIAL_THEME_DEFAULT_COLOR_BUCKETS - 1) as f32;
        std::array::from_fn(|channel| {
            let start = srgb_to_linear(f32::from(start[channel]) / 255.0);
            let end = srgb_to_linear(f32::from(end[channel]) / 255.0);
            linear_to_srgb(start + (end - start) * position)
        })
    })
}

fn srgb_to_linear(channel: f32) -> f32 {
    if channel <= 0.040_45 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(channel: f32) -> f32 {
    let channel = finite_unit_or_zero(channel);
    if channel <= 0.003_130_8 {
        channel * 12.92
    } else {
        1.055 * channel.powf(1.0 / 2.4) - 0.055
    }
    .clamp(0.0, 1.0)
}

const LIGHT_TURTLE_COLORS: [[u8; 3]; 12] = [
    [37, 99, 235],
    [124, 58, 237],
    [219, 39, 119],
    [220, 38, 38],
    [234, 88, 12],
    [202, 138, 4],
    [22, 163, 74],
    [13, 148, 136],
    [8, 145, 178],
    [79, 70, 229],
    [147, 51, 234],
    [71, 85, 105],
];

const DARK_TURTLE_COLORS: [[u8; 3]; 12] = [
    [96, 165, 250],
    [167, 139, 250],
    [244, 114, 182],
    [251, 113, 133],
    [251, 146, 60],
    [250, 204, 21],
    [74, 222, 128],
    [45, 212, 191],
    [34, 211, 238],
    [129, 140, 248],
    [192, 132, 252],
    [203, 213, 225],
];

/// Resolve an explicitly selected turtle color. `ThemeDefault` returns `None`
/// so each target can retain its established default treatment.
pub fn turtle_stroke_rgb(color: crate::StrokeColor, palette: Palette) -> Option<[u8; 3]> {
    match color {
        crate::StrokeColor::ThemeDefault => None,
        crate::StrokeColor::Rgb(rgb) => Some(rgb),
        crate::StrokeColor::PaletteIndex(index) => {
            let colors = match palette {
                Palette::Light => &LIGHT_TURTLE_COLORS,
                Palette::Dark => &DARK_TURTLE_COLORS,
            };
            Some(colors[usize::from(index) % colors.len()])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative_luminance(color: [f32; 3]) -> f32 {
        0.2126 * srgb_to_linear(color[0])
            + 0.7152 * srgb_to_linear(color[1])
            + 0.0722 * srgb_to_linear(color[2])
    }

    fn contrast(left: [f32; 3], right: [f32; 3]) -> f32 {
        let left = relative_luminance(left);
        let right = relative_luminance(right);
        let (lighter, darker) = if left >= right {
            (left, right)
        } else {
            (right, left)
        };
        (lighter + 0.05) / (darker + 0.05)
    }

    #[test]
    fn spatial_palette_retains_emerald_endpoints_and_bounded_buckets() {
        assert_eq!(spatial_theme_default_bucket(-1.0), 0);
        assert_eq!(spatial_theme_default_bucket(0.0), 0);
        assert_eq!(spatial_theme_default_bucket(1.0), 47);
        assert_eq!(spatial_theme_default_bucket(f64::NAN), 24);
        assert_eq!(spatial_theme_default_bucket(f64::INFINITY), 24);
        assert_eq!(
            spatial_theme_default_rgb8_at(0.0, Palette::Light),
            LIGHT_SPATIAL_START
        );
        assert_eq!(
            spatial_theme_default_rgb8_at(1.0, Palette::Light),
            LIGHT_SPATIAL_END
        );
        assert_eq!(
            spatial_theme_default_rgb8_at(0.0, Palette::Dark),
            DARK_SPATIAL_START
        );
        assert_eq!(
            spatial_theme_default_rgb8_at(1.0, Palette::Dark),
            DARK_SPATIAL_END
        );
    }

    #[test]
    fn spatial_palette_interpolates_in_linear_light() {
        let bucket = SPATIAL_THEME_DEFAULT_COLOR_BUCKETS / 2;
        let position = bucket as f32 / (SPATIAL_THEME_DEFAULT_COLOR_BUCKETS - 1) as f32;
        let actual = spatial_theme_default_color(bucket, Palette::Light);
        for channel in 0..3 {
            let start = srgb_to_linear(f32::from(LIGHT_SPATIAL_START[channel]) / 255.0);
            let end = srgb_to_linear(f32::from(LIGHT_SPATIAL_END[channel]) / 255.0);
            let expected = start + (end - start) * position;
            assert!((srgb_to_linear(actual[channel]) - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn spatial_lighting_is_finite_and_near_geometry_has_more_contrast() {
        let cases = [
            (
                Palette::Light,
                [242.0 / 255.0, 245.0 / 255.0, 249.0 / 255.0],
            ),
            (Palette::Dark, [15.0 / 255.0, 20.0 / 255.0, 32.0 / 255.0]),
        ];
        for (palette, background) in cases {
            let base = spatial_theme_default_color_at(0.5, palette);
            let far = spatial_lit_color(base, background, 1.0, 0.0);
            let near = spatial_lit_color(base, background, 1.0, 1.0);
            assert!(far.into_iter().all(|channel| channel.is_finite()));
            assert!(near.into_iter().all(|channel| channel.is_finite()));
            assert!(contrast(near, background) > contrast(far, background));
            assert!(contrast(far, background) >= 3.0);
        }

        let invalid = spatial_lit_color(
            [f32::NAN, f32::INFINITY, -1.0],
            [f32::NAN, 2.0, f32::NEG_INFINITY],
            f64::NAN,
            f64::INFINITY,
        );
        assert!(
            invalid
                .into_iter()
                .all(|channel| channel.is_finite() && (0.0..=1.0).contains(&channel))
        );
    }

    #[test]
    fn bounded_spatial_lighting_retains_endpoints_and_sanitizes_samples() {
        let base = spatial_theme_default_color_at(0.5, Palette::Light);
        let background = [242.0 / 255.0, 245.0 / 255.0, 249.0 / 255.0];
        assert_eq!(
            spatial_lit_color_bounded(base, background, -1.0, -1.0),
            spatial_lit_color(base, background, SPATIAL_SOFTWARE_MIN_LIGHT, 0.0),
        );
        assert_eq!(
            spatial_lit_color_bounded(base, background, 2.0, 2.0),
            spatial_lit_color(base, background, 1.0, 1.0),
        );

        let invalid = spatial_lit_color_bounded(
            [f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            [f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            f64::NAN,
            f64::INFINITY,
        );
        assert!(
            invalid
                .into_iter()
                .all(|channel| channel.is_finite() && (0.0..=1.0).contains(&channel))
        );

        let light = quantize_to_endpoints(
            0.78,
            SPATIAL_SOFTWARE_MIN_LIGHT,
            1.0,
            SPATIAL_SOFTWARE_LIGHT_BUCKETS,
        );
        let normalized = (light - SPATIAL_SOFTWARE_MIN_LIGHT) / (1.0 - SPATIAL_SOFTWARE_MIN_LIGHT);
        assert!((normalized * (SPATIAL_SOFTWARE_LIGHT_BUCKETS - 1) as f64).fract() < 1.0e-12);
    }

    #[test]
    fn stroke_estimator_accumulates_total_length() {
        let mut estimator = StrokeWidthEstimator::default();
        estimator.observe(crate::Line2d((0.0, 0.0), (1.0, 0.0)));
        estimator.observe(crate::Line2d((0.0, 0.0), (0.0, 4.0)));
        estimator.observe(crate::Line2d((3.0, 3.0), (3.0, 3.0)));
        estimator.observe(crate::Line2d((0.0, 0.0), (f64::INFINITY, 0.0)));

        assert_eq!(estimator.total_line_length(), Some(5.0));
        assert_eq!(estimator.line_count(), 2);
    }

    #[test]
    fn total_length_is_invariant_under_collinear_subdivision() {
        let mut whole = StrokeWidthEstimator::default();
        whole.observe(crate::Line2d((0.0, 0.0), (10.0, 0.0)));
        let mut subdivided = StrokeWidthEstimator::default();
        for x in 0..10 {
            subdivided.observe(crate::Line2d((f64::from(x), 0.0), (f64::from(x + 1), 0.0)));
        }

        assert_eq!(whole.total_line_length(), subdivided.total_line_length());
    }

    #[test]
    fn adaptive_width_targets_density_without_a_positive_floor() {
        let low_iteration = adaptive_stroke_width(Some(3.0), (1.0, 1.0), 500.0 / 1.08);
        let high_iteration = adaptive_stroke_width(Some(262_143.0), (511.0, 511.0), 500.0 / 551.88);

        assert_eq!(low_iteration, 10.0);
        assert!((0.586..0.587).contains(&high_iteration));
        assert!(high_iteration < 1.0);
    }

    #[test]
    fn adaptive_width_uses_the_chunky_default_for_sparse_or_invalid_geometry() {
        assert_eq!(adaptive_stroke_width(Some(10.0), (10.0, 0.0), 8.0), 10.0);
        assert_eq!(adaptive_stroke_width(None, (10.0, 10.0), 8.0), 10.0);
        assert_eq!(
            adaptive_stroke_width(Some(10.0), (10.0, 10.0), f64::NAN),
            10.0
        );
    }

    #[test]
    fn svg_width_converts_the_nominal_fitted_width_back_to_view_units() {
        let width = adaptive_svg_stroke_width(Some(1_000.0), (10.0, 10.0), 100.0);
        assert!((width - 0.065).abs() < 1.0e-12);
    }

    #[test]
    fn svg_fitted_width_thins_as_path_density_increases() {
        let fitted_width = |total_line_length| {
            adaptive_svg_stroke_width(total_line_length, (31.0, 31.0), 33.48) * 500.0 / 33.48
        };

        assert!(fitted_width(Some(1_023.0)) > fitted_width(Some(4_095.0)));
        assert!(fitted_width(Some(4_095.0)) > fitted_width(Some(16_383.0)));
    }

    #[test]
    fn scene_detail_cap_preserves_hilbert_and_penrose_density_widths() {
        let hilbert =
            adaptive_scene_stroke_width(Some(262_143.0), 262_143, (511.0, 511.0), 500.0 / 551.88);
        let penrose =
            adaptive_scene_stroke_width(Some(7_920.0), 7_920, (24.172, 23.489), 500.0 / 26.106);

        assert!((0.586..0.587).contains(&hilbert));
        assert!((0.89..0.90).contains(&penrose));
    }

    #[test]
    fn scene_segment_cap_tracks_recursive_step_length() {
        let fitted_width = |line_count, long_edge| {
            adaptive_scene_stroke_width(
                Some(line_count as f64),
                line_count,
                (long_edge * 0.57, long_edge),
                500.0 / (long_edge * 1.08),
            )
        };

        assert!((4.88..4.90).contains(&fitted_width(728, 61.5)));
        assert!((2.77..2.79).contains(&fitted_width(2_186, 108.3)));
        assert!((1.58..1.60).contains(&fitted_width(6_560, 189.3)));
    }

    #[test]
    fn scene_segment_cap_preserves_gaps_in_koch_and_dragon_curves() {
        let fitted_width = |line_count, drawing_extent, view_extent| {
            adaptive_scene_stroke_width(
                Some(line_count as f64),
                line_count,
                drawing_extent,
                500.0 / view_extent,
            )
        };

        let koch_4 = fitted_width(768, (93.53, 81.0), 101.013);
        let koch_9 = fitted_width(786_432, (22_727.97, 19_683.0), 24_546.208);
        let dragon_12 = fitted_width(4_096, (63.0, 95.0), 102.6);
        let dragon_17 = fitted_width(131_072, (511.0, 426.0), 551.88);

        assert!((3.21..3.23).contains(&koch_4));
        assert!((0.092..0.094).contains(&koch_9));
        assert!((3.16..3.18).contains(&dragon_12));
        assert!((0.617..0.619).contains(&dragon_17));
    }

    #[test]
    fn spatial_base_width_caps_sparse_rods_but_keeps_fine_detail() {
        let sparse = adaptive_spatial_scene_stroke_width(None, 3, (10.0, 4.0), 20.0);
        assert_eq!(sparse, MAX_SPATIAL_FITTED_STROKE_WIDTH);

        let detailed =
            adaptive_scene_stroke_width(Some(262_143.0), 262_143, (511.0, 511.0), 500.0 / 551.88);
        assert_eq!(
            adaptive_spatial_scene_stroke_width(
                Some(262_143.0),
                262_143,
                (511.0, 511.0),
                500.0 / 551.88,
            ),
            detailed
        );

        let svg = adaptive_spatial_scene_svg_stroke_width(None, 3, (10.0, 4.0), 100.0);
        assert!((svg - 0.5).abs() < 1.0e-12);
    }
}
