// JFA Dilation Shader
// Creates a mask of pixels within max_width of the silhouette
// Uses separable passes (horizontal then vertical) for O(width) complexity

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct DilateParams {
    // x = max_width, y = direction (0 = horizontal, 1 = vertical)
    data: vec4<f32>,
};

@group(0) @binding(2) var<uniform> params: DilateParams;

// Sample the appropriate channel based on pass direction
// Horizontal pass reads from silhouette.a, vertical reads from mask.r
fn sample_input(uv: vec2<f32>, is_vertical: bool) -> f32 {
    if is_vertical {
        return textureSample(input_texture, input_sampler, uv).r;
    } else {
        // Horizontal pass samples alpha from RGBA silhouette texture
        return textureSample(input_texture, input_sampler, uv).a;
    }
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) f32 {
    let tex_size = vec2<f32>(textureDimensions(input_texture));
    let texel_size = 1.0 / tex_size;
    let max_width = i32(params.data.x);
    let is_vertical = params.data.y > 0.5;

    // Check if any pixel within max_width distance has silhouette/mask
    let step = select(vec2<f32>(texel_size.x, 0.0), vec2<f32>(0.0, texel_size.y), is_vertical);

    // Sample center
    var max_val = sample_input(in.uv, is_vertical);

    // Sample in both directions along the axis
    for (var i: i32 = 1; i <= max_width; i++) {
        let offset = step * f32(i);
        let sample_pos = sample_input(in.uv + offset, is_vertical);
        let sample_neg = sample_input(in.uv - offset, is_vertical);
        max_val = max(max_val, max(sample_pos, sample_neg));

        // Early out if we found a hit
        if max_val > 0.5 {
            return 1.0;
        }
    }

    return max_val;
}
