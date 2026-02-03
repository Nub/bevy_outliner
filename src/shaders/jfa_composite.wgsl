// JFA Composite Shader
// Reads distance field from JFA result and composites outline over scene

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var jfa_texture: texture_2d<f32>;
@group(0) @binding(3) var jfa_sampler: sampler;
@group(0) @binding(4) var silhouette_texture: texture_2d<f32>;
@group(0) @binding(5) var silhouette_sampler: sampler;

struct OutlineSettings {
    color: vec4<f32>,
    width: f32,
    enabled: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(6) var<uniform> settings: OutlineSettings;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene_color = textureSample(scene_texture, scene_sampler, in.uv);

    if settings.enabled < 0.5 {
        return scene_color;
    }

    // Check if we're inside the object (no outline on object surface)
    let silhouette = textureSample(silhouette_texture, silhouette_sampler, in.uv).a;
    if silhouette > 0.5 {
        return scene_color;
    }

    // Read nearest seed UV from JFA result
    let seed_uv = textureSample(jfa_texture, jfa_sampler, in.uv).xy;

    // Check if we found a valid seed
    if seed_uv.x < 0.0 {
        return scene_color;
    }

    // Calculate distance in pixels
    let tex_size = vec2<f32>(textureDimensions(jfa_texture));
    let diff = (in.uv - seed_uv) * tex_size;
    let dist = length(diff);

    // Create smooth outline based on distance
    // 1-pixel smooth falloff at the outer edge for anti-aliasing
    let width = settings.width;
    let outline_strength = 1.0 - smoothstep(width - 1.0, width, dist);

    // Composite outline over scene
    return mix(scene_color, settings.color, outline_strength * settings.color.a);
}
