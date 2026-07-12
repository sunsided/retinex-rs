// Second half of the separable Gaussian blur, consuming the output of
// `blur_horizontal.wgsl` and finishing the illumination estimate for one
// sigma. Same clamped-edge behavior as the horizontal pass.

struct Params {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let radius = i32(params.radius);
    let height = i32(params.height);

    var accum = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = -radius; k <= radius; k = k + 1) {
        let sample_y = clamp(i32(gid.y) + k, 0, height - 1);
        let w = kernel[u32(k + radius)];
        accum = accum + src[u32(sample_y) * params.width + gid.x].xyz * w;
    }

    dst[gid.y * params.width + gid.x] = vec4<f32>(accum, 1.0);
}
