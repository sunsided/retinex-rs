// Turns a blurred (illumination) buffer and the original image into a
// log-domain reflectance contribution for one sigma, then folds it straight
// into the running accumulator with its MSR weight.
//
// For single-scale Retinex this is just called once with weight = 1.0, so
// SSR and MSR share the exact same dispatch instead of needing two code
// paths like the CPU implementation does.

struct Params {
    width: u32,
    height: u32,
    weight: f32,
    epsilon: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> original: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> illumination: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> accum_reflectance: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> accum_illumination: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let idx = gid.y * params.width + gid.x;
    let i = original[idx].xyz;
    let l = illumination[idx].xyz;
    let eps = vec3<f32>(params.epsilon);

    let reflectance = log(i + eps) - log(l + eps);

    // Every pixel index is only ever touched by one invocation per dispatch,
    // and scales run as separate dispatches in submission order, so this
    // read-modify-write never races even without atomics.
    accum_reflectance[idx] = vec4<f32>(accum_reflectance[idx].xyz + reflectance * params.weight, 1.0);
    accum_illumination[idx] = vec4<f32>(accum_illumination[idx].xyz + l * params.weight, 1.0);
}
