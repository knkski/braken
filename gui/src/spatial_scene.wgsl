struct SpatialUniform {
    viewport_and_scale: vec4<f32>,
    center_and_radius: vec4<f32>,
    quaternion: vec4<f32>,
    background: vec4<f32>,
    base_width: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> spatial: SpatialUniform;

const SPATIAL_LIGHT_DIRECTION: vec3<f32> = vec3<f32>(-0.45112924, 0.55138018, 0.70175659);
const SPATIAL_LINE_AMBIENT: f32 = 0.78;
const SPATIAL_POLYGON_AMBIENT: f32 = 0.68;
const SPATIAL_FAR_FOG: f32 = 0.18;
const SPATIAL_NEAR_FOG: f32 = 0.02;

fn srgb_to_linear_component(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb_component(value: f32) -> f32 {
    let bounded = clamp(value, 0.0, 1.0);
    if bounded <= 0.0031308 {
        return bounded * 12.92;
    }
    return 1.055 * pow(bounded, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_component(color.r),
        srgb_to_linear_component(color.g),
        srgb_to_linear_component(color.b),
    );
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_component(color.r),
        linear_to_srgb_component(color.g),
        linear_to_srgb_component(color.b),
    );
}

fn normalized_near_depth(view_depth: f32) -> f32 {
    let radius = max(spatial.center_and_radius.w, 0.0001);
    return clamp(0.5 + view_depth / (2.0 * radius), 0.0, 1.0);
}

fn spatial_lit_color(base: vec3<f32>, light: f32, view_depth: f32) -> vec3<f32> {
    let near_depth = normalized_near_depth(view_depth);
    let fog = mix(SPATIAL_FAR_FOG, SPATIAL_NEAR_FOG, near_depth);
    let shaded = srgb_to_linear(clamp(base, vec3<f32>(0.0), vec3<f32>(1.0)))
        * clamp(light, 0.0, 1.0);
    let background = srgb_to_linear(clamp(spatial.background.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    return linear_to_srgb(mix(shaded, background, fog));
}

fn rod_light(start_view: vec3<f32>, end_view: vec3<f32>) -> f32 {
    let delta = end_view - start_view;
    let delta_length = length(delta);
    if delta_length <= 0.000001 {
        return 1.0;
    }
    let tangent = delta / delta_length;
    let axial_light = clamp(dot(tangent, SPATIAL_LIGHT_DIRECTION), -1.0, 1.0);
    let cylindrical_response = sqrt(max(0.0, 1.0 - axial_light * axial_light));
    return mix(SPATIAL_LINE_AMBIENT, 1.0, cylindrical_response);
}

fn rotate_by_quaternion(vector: vec3<f32>) -> vec3<f32> {
    // `spatial.quaternion` is stored as (w, x, y, z).
    let q = spatial.quaternion.yzw;
    let first = 2.0 * cross(q, vector);
    return vector + spatial.quaternion.x * first + cross(q, first);
}

fn view_position(point: vec3<f32>) -> vec3<f32> {
    return rotate_by_quaternion(point - spatial.center_and_radius.xyz);
}

fn pixel_position(view: vec3<f32>) -> vec2<f32> {
    let viewport = spatial.viewport_and_scale.xy;
    let scale = spatial.viewport_and_scale.z;
    return vec2<f32>(
        spatial.base_width.y + view.x * scale,
        spatial.base_width.z - view.y * scale,
    );
}

fn clip_position(pixel: vec2<f32>, view_depth: f32) -> vec4<f32> {
    let viewport = spatial.viewport_and_scale.xy;
    let radius = max(spatial.center_and_radius.w, 0.0001);
    // Positive view Z is nearer. WGPU's less-equal comparison therefore uses
    // the inverted, sphere-normalized value below.
    let depth = clamp(0.5 - view_depth / (2.0 * radius), 0.0, 1.0);
    return vec4<f32>(
        pixel.x / viewport.x * 2.0 - 1.0,
        1.0 - pixel.y / viewport.y * 2.0,
        depth,
        1.0,
    );
}

struct BackgroundOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) vertex_index: u32) -> BackgroundOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: BackgroundOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

@fragment
fn fs_background(_input: BackgroundOutput) -> @location(0) vec4<f32> {
    return spatial.background;
}

struct PolygonInput {
    @location(0) position: vec3<f32>,
    @location(1) _padding: f32,
    @location(2) color: vec4<f32>,
};

struct PolygonOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) view_position: vec3<f32>,
};

@vertex
fn vs_polygon(input: PolygonInput) -> PolygonOutput {
    let view = view_position(input.position);
    var output: PolygonOutput;
    output.position = clip_position(pixel_position(view), view.z);
    output.color = input.color;
    output.view_position = view;
    return output;
}

@fragment
fn fs_polygon(input: PolygonOutput) -> @location(0) vec4<f32> {
    let x_derivative = dpdx(input.view_position);
    let y_derivative = dpdy(input.view_position);
    let raw_normal = cross(x_derivative, y_derivative);
    let normal_length = length(raw_normal);
    var diffuse = 1.0;
    if normal_length > 0.000001 {
        diffuse = abs(dot(raw_normal / normal_length, SPATIAL_LIGHT_DIRECTION));
    }
    let light = mix(SPATIAL_POLYGON_AMBIENT, 1.0, clamp(diffuse, 0.0, 1.0));
    return vec4<f32>(
        spatial_lit_color(input.color.rgb, light, input.view_position.z),
        input.color.a,
    );
}

struct LineInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) width: f32,
    @location(3) _padding: f32,
    @location(4) color: vec4<f32>,
};

struct LineOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) line_space: vec2<f32>,
    @location(1) @interpolate(flat) length: f32,
    @location(2) @interpolate(flat) half_width: f32,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) coverage_scale: f32,
    @location(5) @interpolate(flat) light: f32,
    @location(6) @interpolate(flat) midpoint_depth: f32,
};

@vertex
fn vs_line(input: LineInput) -> LineOutput {
    let start_view = view_position(input.start);
    let end_view = view_position(input.end);
    let start = pixel_position(start_view);
    let end = pixel_position(end_view);
    let delta = end - start;
    let raw_length = length(delta);
    let segment_length = max(raw_length, 0.0001);
    let direction = select(delta / segment_length, vec2<f32>(1.0, 0.0), raw_length < 0.0001);
    let normal = vec2<f32>(-direction.y, direction.x);
    let requested_width = max(input.width * spatial.base_width.x, 0.0);
    let raster_width = max(requested_width, 1.0);
    let half_width = raster_width * 0.5;
    let raster_extent = half_width + 0.5;
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-raster_extent, -raster_extent),
        vec2<f32>(segment_length + raster_extent, -raster_extent),
        vec2<f32>(segment_length + raster_extent, raster_extent),
        vec2<f32>(-raster_extent, -raster_extent),
        vec2<f32>(segment_length + raster_extent, raster_extent),
        vec2<f32>(-raster_extent, raster_extent),
    );
    let line_space = corners[input.vertex_index];
    let pixel = start + direction * line_space.x + normal * line_space.y;
    let interpolation = clamp(line_space.x / segment_length, 0.0, 1.0);
    let view_depth = mix(start_view.z, end_view.z, interpolation);

    var output: LineOutput;
    output.position = clip_position(pixel, view_depth);
    output.line_space = line_space;
    output.length = segment_length;
    output.half_width = half_width;
    output.color = input.color;
    output.coverage_scale = min(requested_width, 1.0);
    output.light = rod_light(start_view, end_view);
    output.midpoint_depth = (start_view.z + end_view.z) * 0.5;
    return output;
}

@fragment
fn fs_line(input: LineOutput) -> @location(0) vec4<f32> {
    let nearest = vec2<f32>(clamp(input.line_space.x, 0.0, input.length), 0.0);
    let distance_to_segment = distance(input.line_space, nearest);
    let coverage = clamp(input.half_width + 0.5 - distance_to_segment, 0.0, 1.0)
        * input.coverage_scale;
    if coverage <= 0.0 {
        discard;
    }
    var cross_section = 1.0;
    if input.half_width >= 0.75 {
        let radial = clamp(distance_to_segment / max(input.half_width, 0.0001), 0.0, 1.0);
        cross_section = 0.86 + 0.14 * sqrt(max(0.0, 1.0 - radial * radial));
    }
    return vec4<f32>(
        spatial_lit_color(input.color.rgb, input.light * cross_section, input.midpoint_depth),
        input.color.a * coverage,
    );
}
