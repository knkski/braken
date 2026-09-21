struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) start: vec2<f32>,
    @location(1) end: vec2<f32>,
    @location(2) width: f32,
    @location(3) _padding: f32,
    @location(4) color: vec4<f32>,
    @location(5) viewport: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) line_space: vec2<f32>,
    @location(1) @interpolate(flat) length: f32,
    @location(2) @interpolate(flat) half_width: f32,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) coverage_scale: f32,
};

@vertex
fn vs_line(input: VertexInput) -> VertexOutput {
    let delta = input.end - input.start;
    let raw_length = length(delta);
    let segment_length = max(raw_length, 0.0001);
    // A zero-length draw is a round point, not six coincident vertices.
    let direction = select(delta / segment_length, vec2<f32>(1.0, 0.0), raw_length < 0.0001);
    let normal = vec2<f32>(-direction.y, direction.x);
    // A one-pixel raster footprint keeps subpixel lines sampleable, while
    // coverage_scale preserves their requested ink instead of widening them.
    let requested_width = max(input.width, 0.0);
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
    let pixel = input.start + direction * line_space.x + normal * line_space.y;

    let ndc = vec2<f32>(
        pixel.x / input.viewport.x * 2.0 - 1.0,
        1.0 - pixel.y / input.viewport.y * 2.0,
    );
    var output: VertexOutput;
    output.position = vec4<f32>(ndc, 0.0, 1.0);
    output.line_space = line_space;
    output.length = segment_length;
    output.half_width = half_width;
    output.color = input.color;
    output.coverage_scale = min(requested_width, 1.0);
    return output;
}

@fragment
fn fs_line(input: VertexOutput) -> @location(0) vec4<f32> {
    let nearest = vec2<f32>(clamp(input.line_space.x, 0.0, input.length), 0.0);
    let distance_to_segment = distance(input.line_space, nearest);
    let coverage = clamp(input.half_width + 0.5 - distance_to_segment, 0.0, 1.0)
        * input.coverage_scale;
    if coverage <= 0.0 {
        discard;
    }
    return vec4<f32>(input.color.rgb, input.color.a * coverage);
}

struct CompositeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_composite(@builtin(vertex_index) vertex_index: u32) -> CompositeOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    var output: CompositeOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return output;
}

@group(0) @binding(0)
var composite_texture: texture_2d<f32>;

@group(0) @binding(1)
var composite_sampler: sampler;

struct CompositeUniform {
    sample_scale: vec2<f32>,
    sample_offset: vec2<f32>,
    background: vec4<f32>,
};

@group(0) @binding(2)
var<uniform> composite: CompositeUniform;

@fragment
fn fs_composite(input: CompositeOutput) -> @location(0) vec4<f32> {
    let sample_uv = input.uv * composite.sample_scale + composite.sample_offset;
    if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
        return composite.background;
    }
    return textureSample(composite_texture, composite_sampler, sample_uv);
}
