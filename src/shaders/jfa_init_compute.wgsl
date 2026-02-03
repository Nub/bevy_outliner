// JFA Initialization Compute Shader
// Converts silhouette texture to seed coordinates for Jump Flood Algorithm
// Each thread processes one pixel

@group(0) @binding(0) var silhouette_texture: texture_2d<f32>;
@group(0) @binding(1) var mask_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rg16float, write>;

// Invalid seed marker - any negative value works since valid UVs are [0,1]
const INVALID_SEED: vec2<f32> = vec2<f32>(-1.0, -1.0);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tex_size = textureDimensions(silhouette_texture);

    // Bounds check
    if global_id.x >= tex_size.x || global_id.y >= tex_size.y {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(tex_size);

    // Early discard if pixel is outside the region of interest
    let mask = textureLoad(mask_texture, coord, 0).r;
    if mask < 0.5 {
        textureStore(output_texture, coord, vec4<f32>(INVALID_SEED, 0.0, 0.0));
        return;
    }

    let alpha = textureLoad(silhouette_texture, coord, 0).a;

    // If inside object (silhouette), this pixel is a seed - store its UV
    // Otherwise, store invalid marker
    if alpha > 0.5 {
        textureStore(output_texture, coord, vec4<f32>(uv, 0.0, 0.0));
    } else {
        textureStore(output_texture, coord, vec4<f32>(INVALID_SEED, 0.0, 0.0));
    }
}
