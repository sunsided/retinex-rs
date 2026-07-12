//! wgpu-backed Retinex processing (feature = "gpu")
//!
//! Mirrors the CPU entry points in the crate root — [`single_scale_retinex_full_gpu`],
//! [`multi_scale_retinex_full_gpu`] and their color-restored counterparts return the
//! same [`RetinexOutput`] / [`RgbImage`] types the CPU path does, so callers can swap
//! implementations without touching anything downstream.
//!
//! # Why this is faster than the CPU path
//!
//! The CPU implementation calls `imageproc::filter::gaussian_blur_f32` once per
//! channel per sigma, and for multi-scale Retinex those scales are independent so
//! they parallelize well with rayon — but they're still bound by system memory
//! bandwidth and scalar (or at best SIMD-lane-width) throughput per core.
//!
//! On the GPU, one dispatch blurs all three channels of every pixel at once, using
//! a separable two-pass convolution (horizontal then vertical) so an NxN blur costs
//! O(2N) samples per pixel instead of O(N^2). The log-domain combine step and the
//! multi-scale weighted accumulation are fused into a single compute pass per sigma,
//! so a full MSR call over three scales is nine dispatches total with no readback
//! in between — only the final reflectance/illumination buffers come back to the CPU.
//!
//! # Numerical differences from the CPU path
//!
//! Gaussian kernel truncation here uses a fixed 3-sigma radius rather than
//! whatever heuristic `imageproc` uses internally, so pixel values will be very
//! close to but not bit-identical with [`crate::single_scale_retinex_full`].
//! For Retinex's use case (illumination estimation) this makes no visible
//! difference; if you need exact parity for a golden-image test, use the CPU path.

use crate::{RetinexError, RetinexOutput, RetinexResult};
use image::{DynamicImage, Rgb32FImage};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 8;
const KERNEL_SIGMA_CUTOFF: f32 = 3.0;

/// Single-scale Retinex on the GPU, returning both reflectance and illumination.
///
/// Numerically equivalent in intent to [`crate::single_scale_retinex_full`] — see
/// the [module docs](self) for the (small) differences in Gaussian kernel truncation.
pub fn single_scale_retinex_full_gpu(image: &DynamicImage, sigma: f32) -> RetinexResult<RetinexOutput> {
    if sigma <= 0.0 {
        return Err(RetinexError::InvalidSigma(sigma));
    }
    let rgb = image.to_rgb32f();
    retinex_full_gpu(&rgb, &[sigma])
}

/// Multi-scale Retinex on the GPU, returning both reflectance and illumination.
///
/// Numerically equivalent in intent to [`crate::multi_scale_retinex_full`] — see
/// the [module docs](self) for the (small) differences in Gaussian kernel truncation.
pub fn multi_scale_retinex_full_gpu(
    image: &DynamicImage,
    sigmas: &[f32],
) -> RetinexResult<RetinexOutput> {
    if sigmas.is_empty() {
        return Err(RetinexError::EmptySigmaSet);
    }
    if let Some(&invalid) = sigmas.iter().find(|&&s| s <= 0.0) {
        return Err(RetinexError::InvalidSigma(invalid));
    }
    let rgb = image.to_rgb32f();
    retinex_full_gpu(&rgb, sigmas)
}

/// GPU counterpart of [`crate::single_scale_retinex_color_restored`].
pub fn single_scale_retinex_color_restored_gpu(
    image: &DynamicImage,
    sigma: f32,
) -> RetinexResult<image::RgbImage> {
    if sigma <= 0.0 {
        return Err(RetinexError::InvalidSigma(sigma));
    }
    let rgb = image.to_rgb32f();
    let output = retinex_full_gpu(&rgb, &[sigma])?;
    let restored = crate::apply_color_restoration(&output.reflectance, &rgb);
    Ok(crate::float_to_rgb8(&restored))
}

/// GPU counterpart of [`crate::multi_scale_retinex_color_restored`].
pub fn multi_scale_retinex_color_restored_gpu(
    image: &DynamicImage,
    sigmas: &[f32],
) -> RetinexResult<image::RgbImage> {
    if sigmas.is_empty() {
        return Err(RetinexError::EmptySigmaSet);
    }
    if let Some(&invalid) = sigmas.iter().find(|&&s| s <= 0.0) {
        return Err(RetinexError::InvalidSigma(invalid));
    }
    let rgb = image.to_rgb32f();
    let output = retinex_full_gpu(&rgb, sigmas)?;
    let restored = crate::apply_color_restoration(&output.reflectance, &rgb);
    Ok(crate::float_to_rgb8(&restored))
}

// Takes an already-converted Rgb32FImage rather than a DynamicImage so
// callers that also need the RGB buffer for color restoration (see the
// _color_restored_gpu variants above) only pay for to_rgb32f() once.
fn retinex_full_gpu(rgb: &Rgb32FImage, sigmas: &[f32]) -> RetinexResult<RetinexOutput> {
    let ctx = context()?;
    let (reflectance, illumination) = ctx
        .run(rgb, sigmas)
        .map_err(|e| RetinexError::Gpu(e.to_string()))?;
    Ok(RetinexOutput {
        reflectance,
        illumination,
    })
}

// One adapter/device/pipeline set for the whole process. Spinning up a wgpu
// device is not cheap (driver init, shader compilation), and nothing about it
// depends on the image being processed, so we pay that cost exactly once.
static CONTEXT: OnceLock<Result<GpuContext, String>> = OnceLock::new();

fn context() -> RetinexResult<&'static GpuContext> {
    CONTEXT
        .get_or_init(|| GpuContext::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| RetinexError::Gpu(e.clone()))
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    blur_layout: wgpu::BindGroupLayout,
    combine_pipeline: wgpu::ComputePipeline,
    combine_layout: wgpu::BindGroupLayout,
    max_buffer_bytes: u64,
}

#[derive(Debug)]
struct GpuInitError(String);

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for GpuInitError {}

impl GpuContext {
    fn new() -> Result<Self, GpuInitError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                GpuInitError(
                    "no compatible GPU adapter found (Vulkan/Metal/DX12) — install a GPU driver \
                     or fall back to the CPU path"
                        .to_string(),
                )
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("retinex-gpu-device"),
                    required_features: wgpu::Features::empty(),
                    // Ask for what the adapter actually supports rather than the
                    // conservative downlevel defaults (~128MB buffers) — Retinex
                    // buffers scale with image resolution and the default cap
                    // bites well before anything the CPU path would balk at.
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .map_err(|e| GpuInitError(format!("failed to acquire device: {e}")))?;

        let max_buffer_bytes = device.limits().max_storage_buffer_binding_size as u64;

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur-bind-group-layout"),
            entries: &[
                storage_entry(0, wgpu::BufferBindingType::Uniform),
                storage_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                storage_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                storage_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let combine_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("combine-bind-group-layout"),
            entries: &[
                storage_entry(0, wgpu::BufferBindingType::Uniform),
                storage_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                storage_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                storage_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                storage_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        // Horizontal and vertical blur share a bind group layout (same bindings,
        // same params struct) but are different shader modules, so each gets its
        // own pipeline built off that shared layout.
        let blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blur-pipeline-layout"),
                bind_group_layouts: &[&blur_layout],
                push_constant_ranges: &[],
            });

        let blur_h_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur-horizontal"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur_horizontal.wgsl").into()),
        });
        let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blur-horizontal-pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_h_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let blur_v_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur-vertical"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur_vertical.wgsl").into()),
        });
        let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("blur-vertical-pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_v_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let combine_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retinex-combine"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/retinex_combine.wgsl").into()),
        });
        let combine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("combine-pipeline-layout"),
                bind_group_layouts: &[&combine_layout],
                push_constant_ranges: &[],
            });
        let combine_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("combine-pipeline"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            device,
            queue,
            blur_h_pipeline,
            blur_v_pipeline,
            blur_layout,
            combine_pipeline,
            combine_layout,
            max_buffer_bytes,
        })
    }

    fn run(
        &self,
        image: &Rgb32FImage,
        sigmas: &[f32],
    ) -> Result<(Rgb32FImage, Rgb32FImage), GpuInitError> {
        let (width, height) = image.dimensions();
        let pixel_count = (width as u64) * (height as u64);
        let buffer_bytes = pixel_count * std::mem::size_of::<[f32; 4]>() as u64;

        if buffer_bytes > self.max_buffer_bytes {
            return Err(GpuInitError(format!(
                "image is {width}x{height} ({buffer_bytes} bytes per channel buffer), which \
                 exceeds this device's max_storage_buffer_binding_size of {}; tile the image \
                 or use the CPU path for images this large",
                self.max_buffer_bytes
            )));
        }

        let device = &self.device;
        let queue = &self.queue;

        let original_data = to_padded_rgba(image);
        let original_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("original"),
            contents: bytemuck::cast_slice(&original_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let storage_rw = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buffer_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let scratch_h = storage_rw("blur-scratch-horizontal");
        let scratch_v = storage_rw("blur-scratch-vertical");
        let accum_reflectance = storage_rw("accum-reflectance");
        let accum_illumination = storage_rw("accum-illumination");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("retinex-encoder"),
        });
        encoder.clear_buffer(&accum_reflectance, 0, None);
        encoder.clear_buffer(&accum_illumination, 0, None);

        let weight = 1.0 / sigmas.len() as f32;

        for &sigma in sigmas {
            let (kernel_weights, radius) = gaussian_kernel(sigma);
            let kernel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gaussian-kernel"),
                contents: bytemuck::cast_slice(&kernel_weights),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let blur_params = BlurParams {
                width,
                height,
                radius,
                _pad: 0,
            };
            let blur_params_h = uniform_buf(device, "blur-params-h", &blur_params);
            let blur_params_v = uniform_buf(device, "blur-params-v", &blur_params);

            let bg_h = self.blur_bind_group(&blur_params_h, &original_buf, &kernel_buf, &scratch_h);
            let bg_v = self.blur_bind_group(&blur_params_v, &scratch_h, &kernel_buf, &scratch_v);

            let (dispatch_x, dispatch_y) = dispatch_size(width, height);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("blur-horizontal-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.blur_h_pipeline);
                pass.set_bind_group(0, &bg_h, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("blur-vertical-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.blur_v_pipeline);
                pass.set_bind_group(0, &bg_v, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            let combine_params = CombineParams {
                width,
                height,
                weight,
                epsilon: crate::EPSILON,
            };
            let combine_params_buf = uniform_buf(device, "combine-params", &combine_params);
            let bg_combine = self.combine_bind_group(
                &combine_params_buf,
                &original_buf,
                &scratch_v,
                &accum_reflectance,
                &accum_illumination,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("combine-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.combine_pipeline);
                pass.set_bind_group(0, &bg_combine, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }
        }

        let reflectance_staging = readback_buffer(device, &mut encoder, &accum_reflectance, buffer_bytes);
        let illumination_staging = readback_buffer(device, &mut encoder, &accum_illumination, buffer_bytes);

        queue.submit(Some(encoder.finish()));

        let reflectance = map_and_read(device, &reflectance_staging, width, height)?;
        let illumination = map_and_read(device, &illumination_staging, width, height)?;

        Ok((reflectance, illumination))
    }

    fn blur_bind_group(
        &self,
        params: &wgpu::Buffer,
        src: &wgpu::Buffer,
        kernel: &wgpu::Buffer,
        dst: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur-bind-group"),
            layout: &self.blur_layout,
            entries: &[
                bind(0, params),
                bind(1, src),
                bind(2, kernel),
                bind(3, dst),
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn combine_bind_group(
        &self,
        params: &wgpu::Buffer,
        original: &wgpu::Buffer,
        illumination: &wgpu::Buffer,
        accum_reflectance: &wgpu::Buffer,
        accum_illumination: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("combine-bind-group"),
            layout: &self.combine_layout,
            entries: &[
                bind(0, params),
                bind(1, original),
                bind(2, illumination),
                bind(3, accum_reflectance),
                bind(4, accum_illumination),
            ],
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    width: u32,
    height: u32,
    weight: f32,
    epsilon: f32,
}

fn storage_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_buf<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, data: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn dispatch_size(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(WORKGROUP_SIZE),
        height.div_ceil(WORKGROUP_SIZE),
    )
}

fn readback_buffer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    size: u64,
) -> wgpu::Buffer {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    staging
}

fn map_and_read(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> Result<Rgb32FImage, GpuInitError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| GpuInitError(format!("readback channel closed unexpectedly: {e}")))?
        .map_err(|e| GpuInitError(format!("failed to map staging buffer: {e}")))?;

    let raw = slice.get_mapped_range();
    let padded: &[[f32; 4]] = bytemuck::cast_slice(&raw);
    let image = from_padded_rgba(padded, width, height);
    drop(raw);
    staging.unmap();
    Ok(image)
}

fn to_padded_rgba(image: &Rgb32FImage) -> Vec<[f32; 4]> {
    image
        .pixels()
        .map(|p| [p.0[0], p.0[1], p.0[2], 0.0])
        .collect()
}

fn from_padded_rgba(data: &[[f32; 4]], width: u32, height: u32) -> Rgb32FImage {
    let mut out = Rgb32FImage::new(width, height);
    for (pixel, chunk) in out.pixels_mut().zip(data.iter()) {
        pixel.0 = [chunk[0], chunk[1], chunk[2]];
    }
    out
}

/// Normalized 1D Gaussian kernel weights and the radius they were truncated at.
///
/// Uses a fixed 3-sigma cutoff, which captures >99.7% of the kernel's mass —
/// plenty for illumination estimation, where we're deliberately throwing away
/// high-frequency detail anyway.
fn gaussian_kernel(sigma: f32) -> (Vec<f32>, u32) {
    let radius = (sigma * KERNEL_SIGMA_CUTOFF).ceil().max(1.0) as u32;
    let two_sigma_sq = 2.0 * sigma * sigma;

    let mut weights: Vec<f32> = (-(radius as i32)..=(radius as i32))
        .map(|k| (-(k * k) as f32 / two_sigma_sq).exp())
        .collect();

    let sum: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }

    (weights, radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    fn load_test_image() -> DynamicImage {
        image::open("images/house.jpg").expect("failed to load test image")
    }

    // The CPU path (imageproc) and the GPU path use different Gaussian
    // implementations -- imageproc appears to use a fast recursive/IIR
    // approximation with its own edge handling, while the GPU kernel is a
    // truncated FIR convolution with clamped edges. Both are legitimate
    // Gaussian blurs, but they will not agree pixel-for-pixel, and log-domain
    // reflectance amplifies even small illumination differences near dark
    // pixels. So instead of asserting numerical parity with the CPU path
    // (fragile, and not actually what we care about), these tests check the
    // invariants that must hold regardless of which blur implementation
    // produced the illumination estimate.

    fn mean_abs_diff(a: &Rgb32FImage, b: &Rgb32FImage) -> f32 {
        assert_eq!(a.dimensions(), b.dimensions());
        let mut total = 0.0f64;
        let mut count = 0u64;
        for (pa, pb) in a.pixels().zip(b.pixels()) {
            for c in 0..3 {
                total += (pa.0[c] - pb.0[c]).abs() as f64;
                count += 1;
            }
        }
        (total / count as f64) as f32
    }

    // Retinex identity: reflectance is defined as log(I) - log(L), so
    // exp(reflectance) * illumination must reconstruct the original pixel
    // (up to the EPSILON fudge both paths add before taking the log). This
    // catches real bugs in the combine shader -- wrong buffer, wrong sign,
    // wrong epsilon -- without depending on matching another library's
    // specific blur kernel.
    // Single-scale reconstruction is algebraically exact (up to f32 rounding),
    // so every pixel gets a tight per-pixel bound.
    fn assert_retinex_identity_exact(original: &Rgb32FImage, output: &RetinexOutput) {
        for ((orig, refl), illum) in original
            .pixels()
            .zip(output.reflectance.pixels())
            .zip(output.illumination.pixels())
        {
            for c in 0..3 {
                // reflectance = ln(I + eps) - ln(L + eps), so exactly:
                // I = exp(reflectance) * (L + eps) - eps
                let reconstructed = refl.0[c].exp() * (illum.0[c] + crate::EPSILON) - crate::EPSILON;
                let diff = (reconstructed - orig.0[c]).abs();
                assert!(
                    diff < 1e-4,
                    "retinex identity broken: orig={}, reconstructed={}, diff={diff}",
                    orig.0[c],
                    reconstructed
                );
            }
        }
    }

    // MSR averages log-reflectance and illumination separately across scales,
    // and log() is nonlinear, so mean(log(I) - log(L_i)) != log(I) -
    // log(mean(L_i)) in general (Jensen's inequality) -- the CPU implementation
    // has the exact same property, it isn't a GPU-specific approximation. The
    // gap is a function of local contrast between the sigma scales and is
    // occasionally a percent or two at individual high-contrast pixels, so we
    // check the mean reconstruction error over the whole image rather than a
    // per-pixel bound -- that's what would actually indicate a broken shader
    // versus normal MSR behavior.
    fn assert_retinex_identity_approx(original: &Rgb32FImage, output: &RetinexOutput) {
        let mut total = 0.0f64;
        let mut count = 0u64;
        for ((orig, refl), illum) in original
            .pixels()
            .zip(output.reflectance.pixels())
            .zip(output.illumination.pixels())
        {
            for c in 0..3 {
                let reconstructed = refl.0[c].exp() * (illum.0[c] + crate::EPSILON) - crate::EPSILON;
                total += (reconstructed - orig.0[c]).abs() as f64;
                count += 1;
            }
        }
        let mean_diff = (total / count as f64) as f32;
        assert!(
            mean_diff < 2e-3,
            "mean MSR reconstruction error too high: {mean_diff}"
        );
    }

    #[test]
    fn test_single_scale_gpu_satisfies_retinex_identity() {
        let img = load_test_image();
        let gpu = single_scale_retinex_full_gpu(&img, 15.0).unwrap();
        assert_retinex_identity_exact(&img.to_rgb32f(), &gpu);
    }

    #[test]
    fn test_multi_scale_gpu_satisfies_retinex_identity() {
        let img = load_test_image();
        let gpu = multi_scale_retinex_full_gpu(&img, &[10.0, 40.0]).unwrap();
        assert_retinex_identity_approx(&img.to_rgb32f(), &gpu);
    }

    // Illumination is a low-pass version of the image, so it varies far less
    // pixel-to-pixel than the original -- regardless of which Gaussian
    // implementation produced it. A GPU bug that leaked the unblurred image
    // straight through (e.g. a swapped buffer binding) would fail this.
    #[test]
    fn test_gpu_illumination_is_smoother_than_original() {
        let img = load_test_image();
        let rgb = img.to_rgb32f();
        let gpu = single_scale_retinex_full_gpu(&img, 15.0).unwrap();

        let variation = |image: &Rgb32FImage| -> f64 {
            let (w, h) = image.dimensions();
            let mut total = 0.0f64;
            for y in 0..h {
                for x in 1..w {
                    let a = image.get_pixel(x - 1, y);
                    let b = image.get_pixel(x, y);
                    for c in 0..3 {
                        total += (a.0[c] - b.0[c]).abs() as f64;
                    }
                }
            }
            total
        };

        let original_variation = variation(&rgb);
        let illumination_variation = variation(&gpu.illumination);
        assert!(
            illumination_variation < original_variation,
            "blurred illumination should vary less than the original image"
        );
    }

    // Sanity cross-check against the CPU path: same ballpark, not exact
    // parity. A generous tolerance because of the kernel/edge differences
    // noted above -- this is here to catch gross errors (wrong channel order,
    // orientation flipped, scale way off), not to enforce numerical parity.
    #[test]
    fn test_single_scale_gpu_roughly_agrees_with_cpu() {
        let img = load_test_image();
        let cpu = crate::single_scale_retinex_full(&img, 15.0).unwrap();
        let gpu = single_scale_retinex_full_gpu(&img, 15.0).unwrap();

        assert_eq!(cpu.reflectance.dimensions(), gpu.reflectance.dimensions());
        assert!(
            mean_abs_diff(&cpu.illumination, &gpu.illumination) < 0.15,
            "illumination is wildly different between CPU and GPU paths -- likely a real bug, \
             not just a kernel implementation difference"
        );
    }

    #[test]
    fn test_gpu_color_restored_dimensions() {
        let img = load_test_image();
        let result = multi_scale_retinex_color_restored_gpu(&img, &[10.0, 40.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimensions(), img.dimensions());
    }

    #[test]
    fn test_gpu_invalid_sigma_rejected() {
        let img = load_test_image();
        let result = single_scale_retinex_full_gpu(&img, -1.0);
        assert!(matches!(result, Err(RetinexError::InvalidSigma(_))));
    }

    #[test]
    fn test_gpu_empty_sigma_set_rejected() {
        let img = load_test_image();
        let result = multi_scale_retinex_full_gpu(&img, &[]);
        assert!(matches!(result, Err(RetinexError::EmptySigmaSet)));
    }

    #[test]
    fn test_gaussian_kernel_is_normalized_and_symmetric() {
        let (weights, radius) = gaussian_kernel(5.0);
        assert_eq!(weights.len(), (2 * radius + 1) as usize);

        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel should sum to 1, got {sum}");

        for i in 0..weights.len() / 2 {
            let a = weights[i];
            let b = weights[weights.len() - 1 - i];
            assert!((a - b).abs() < 1e-6, "kernel should be symmetric");
        }
    }
}

