// JFA Initialization Shader
// Converts silhouette texture to seed coordinates for Jump Flood Algorithm
// Pixels inside objects store their own UV, pixels outside store invalid (-1, -1)

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var silhouette_texture: texture_2d<f32>;
@group(0) @binding(1) var silhouette_sampler: sampler;

// Invalid seed marker - any negative value works since valid UVs are [0,1]
const INVALID_SEED: vec2<f32> = vec2<f32>(-1.0, -1.0);

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec2<f32> {
    let alpha = textureSample(silhouette_texture, silhouette_sampler, in.uv).a;

    // If inside object (silhouette), this pixel is a seed - store its UV
    // Otherwise, store invalid marker
    if alpha > 0.5 {
        return in.uv;
    } else {
        return INVALID_SEED;
    }
}
