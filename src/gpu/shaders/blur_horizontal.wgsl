// First half of a separable Gaussian blur. Runs along the x axis and hands
// the partial result to `blur_vertical.wgsl`. Splitting the 2D convolution
// into two 1D passes turns an O(r^2) per-pixel cost into O(r), which is
// where basically all of the win over the CPU path comes from.

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
    let width = i32(params.width);
    let row_base = gid.y * params.width;

    var accum = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = -radius; k <= radius; k = k + 1) {
        // Edge pixels get clamped rather than wrapped or zero-padded, matching
        // the boundary behavior of imageproc's gaussian_blur_f32.
        let sample_x = clamp(i32(gid.x) + k, 0, width - 1);
        let w = kernel[u32(k + radius)];
        accum = accum + src[row_base + u32(sample_x)].xyz * w;
    }

    dst[row_base + gid.x] = vec4<f32>(accum, 1.0);
}
