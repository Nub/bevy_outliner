// Outline Composite Shader
// Detects edges on silhouette texture and composites over scene
// Uses multi-pass sampling for smoother corners

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Bind group layout:
// - binding 0: scene texture
// - binding 1: scene sampler
// - binding 2: silhouette texture (white objects on transparent)
// - binding 3: silhouette sampler
// - binding 4: settings uniform

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var silhouette_texture: texture_2d<f32>;
@group(0) @binding(3) var silhouette_sampler: sampler;

struct OutlineSettings {
    color: vec4<f32>,
    width: f32,
    enabled: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(4) var<uniform> settings: OutlineSettings;

// Sample silhouette alpha (object presence)
fn sample_silhouette(uv: vec2<f32>) -> f32 {
    return textureSample(silhouette_texture, silhouette_sampler, uv).a;
}

// JFA-style distance field approximation
// Samples at multiple distances to find the nearest edge
fn compute_distance_field(uv: vec2<f32>, texel_size: vec2<f32>) -> f32 {
    let center = sample_silhouette(uv);

    // If we're inside the object, return 0 (no outline on object)
    if center > 0.5 {
        return 0.0;
    }

    let width = settings.width;
    var min_dist: f32 = width + 1.0; // Start with max distance

    // JFA-style: sample at multiple step sizes (like jump flood passes)
    // Start with large jumps and progressively get smaller
    let step_sizes = array<f32, 5>(1.0, 0.5, 0.25, 0.125, 0.0625);

    for (var step_idx: i32 = 0; step_idx < 5; step_idx++) {
        let step = step_sizes[step_idx] * width;
        if step < 0.5 {
            continue;
        }

        // Sample in 16 directions for smooth corners
        let num_samples = 16;
        for (var i: i32 = 0; i < num_samples; i++) {
            let angle = f32(i) * 6.28318530718 / f32(num_samples);
            let dir = vec2<f32>(cos(angle), sin(angle));

            // Check at this step distance
            let sample_uv = uv + dir * step * texel_size;
            let sample_val = sample_silhouette(sample_uv);

            if sample_val > 0.5 {
                // Found the object - compute distance
                let dist = step;
                min_dist = min(min_dist, dist);
            }
        }
    }

    // Additional fine samples near the edge for precision
    if min_dist < width {
        let fine_samples = 32;
        for (var i: i32 = 0; i < fine_samples; i++) {
            let angle = f32(i) * 6.28318530718 / f32(fine_samples);
            let dir = vec2<f32>(cos(angle), sin(angle));

            // Binary search style - sample between 0 and current min_dist
            for (var d: f32 = 0.5; d <= min_dist; d += 0.5) {
                let sample_uv = uv + dir * d * texel_size;
                let sample_val = sample_silhouette(sample_uv);

                if sample_val > 0.5 {
                    min_dist = min(min_dist, d);
                    break;
                }
            }
        }
    }

    return min_dist;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene_color = textureSample(scene_texture, scene_sampler, in.uv);

    if settings.enabled < 0.5 {
        return scene_color;
    }

    let tex_size = vec2<f32>(textureDimensions(silhouette_texture));
    let texel_size = 1.0 / tex_size;

    // Compute distance to nearest silhouette edge
    let dist = compute_distance_field(in.uv, texel_size);

    // Create smooth outline based on distance
    let width = settings.width;

    // Outline is drawn where distance is within the width
    // 1-pixel smooth falloff at the outer edge for anti-aliasing
    let outline_strength = 1.0 - smoothstep(width - 1.0, width, dist);

    // Only show outline outside the object
    let center = sample_silhouette(in.uv);
    let is_outside = 1.0 - step(0.5, center);

    let final_strength = outline_strength * is_outside;

    // Composite outline over scene
    return mix(scene_color, settings.color, final_strength * settings.color.a);
}
