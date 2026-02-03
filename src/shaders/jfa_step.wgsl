// JFA Step Shader
// One pass of the Jump Flood Algorithm
// Samples 9 neighbors at step_size offset and propagates closest seed

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var jfa_texture: texture_2d<f32>;
@group(0) @binding(1) var jfa_sampler: sampler;

struct JfaParams {
    // vec4 to match Rust struct layout: [step_size, pad, pad, pad]
    data: vec4<f32>,
};

@group(0) @binding(2) var<uniform> params: JfaParams;

fn get_step_size() -> f32 {
    return params.data.x;
}

const INVALID_SEED: vec2<f32> = vec2<f32>(-1.0, -1.0);

// Check if a seed coordinate is valid
fn is_valid_seed(seed: vec2<f32>) -> bool {
    return seed.x >= 0.0;
}

// Calculate squared distance from pixel UV to seed UV (in pixel space)
fn seed_distance_sq(pixel_uv: vec2<f32>, seed_uv: vec2<f32>, tex_size: vec2<f32>) -> f32 {
    let diff = (pixel_uv - seed_uv) * tex_size;
    return dot(diff, diff);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec2<f32> {
    let tex_size = vec2<f32>(textureDimensions(jfa_texture));
    let texel_size = 1.0 / tex_size;
    let step_offset = get_step_size() * texel_size;

    var best_seed = INVALID_SEED;
    var best_dist_sq = 1e20;

    // Sample 9 neighbors: self + 8 directions at step_size offset
    // This is the core of JFA - exponentially decreasing step sizes
    // converge to the nearest seed in O(log n) passes
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * step_offset;
            let sample_uv = in.uv + offset;

            // Skip out-of-bounds samples
            if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
                continue;
            }

            let seed = textureSample(jfa_texture, jfa_sampler, sample_uv).xy;

            if is_valid_seed(seed) {
                let dist_sq = seed_distance_sq(in.uv, seed, tex_size);
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_seed = seed;
                }
            }
        }
    }

    return best_seed;
}
