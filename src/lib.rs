use image::{DynamicImage, ImageBuffer, Luma, Rgb, Rgb32FImage, RgbImage};
use imageproc::filter::gaussian_blur_f32;

const EPSILON: f32 = 1e-6;

#[derive(Debug)]
pub enum RetinexError {
    EmptySigmaSet,
    InvalidSigma(f32),
}

impl std::fmt::Display for RetinexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetinexError::EmptySigmaSet => write!(f, "expected at least one sigma value"),
            RetinexError::InvalidSigma(value) => write!(f, "sigma must be positive, got {value}"),
        }
    }
}

impl std::error::Error for RetinexError {}

pub type RetinexResult<T> = Result<T, RetinexError>;

/// Output from Retinex processing containing both reflectance and illumination
#[derive(Debug, Clone)]
pub struct RetinexOutput {
    /// The estimated reflectance (log-domain, unnormalized)
    pub reflectance: Rgb32FImage,
    /// The estimated illumination (average of blurred versions across scales)
    pub illumination: Rgb32FImage,
}

/// Single-scale Retinex returning both reflectance and illumination
pub fn single_scale_retinex_full(image: &DynamicImage, sigma: f32) -> RetinexResult<RetinexOutput> {
    if sigma <= 0.0 {
        return Err(RetinexError::InvalidSigma(sigma));
    }

    let rgb = image.to_rgb32f();
    let (reflectance, illumination) = ssr_rgb32f_with_illumination(&rgb, sigma);

    Ok(RetinexOutput {
        reflectance,
        illumination,
    })
}

/// Multi-scale Retinex returning both reflectance and illumination
pub fn multi_scale_retinex_full(
    image: &DynamicImage,
    sigmas: &[f32],
) -> RetinexResult<RetinexOutput> {
    if sigmas.is_empty() {
        return Err(RetinexError::EmptySigmaSet);
    }

    if let Some(&invalid) = sigmas.iter().find(|&&sigma| sigma <= 0.0) {
        return Err(RetinexError::InvalidSigma(invalid));
    }

    let rgb = image.to_rgb32f();
    let (reflectance, illumination) = msr_rgb32f_with_illumination(&rgb, sigmas);

    Ok(RetinexOutput {
        reflectance,
        illumination,
    })
}

/// Single-scale Retinex with color restoration (MSRCR)
pub fn single_scale_retinex_color_restored(
    image: &DynamicImage,
    sigma: f32,
    gain: f32,
    offset: f32,
) -> RetinexResult<RgbImage> {
    if sigma <= 0.0 {
        return Err(RetinexError::InvalidSigma(sigma));
    }

    let rgb = image.to_rgb32f();
    let (reflectance, _) = ssr_rgb32f_with_illumination(&rgb, sigma);
    let color_restored = apply_color_restoration(&reflectance, &rgb, gain, offset);
    Ok(float_to_rgb8_scaled(&color_restored))
}

/// Multi-scale Retinex with color restoration (MSRCR)
pub fn multi_scale_retinex_color_restored(
    image: &DynamicImage,
    sigmas: &[f32],
    gain: f32,
    offset: f32,
) -> RetinexResult<RgbImage> {
    if sigmas.is_empty() {
        return Err(RetinexError::EmptySigmaSet);
    }

    if let Some(&invalid) = sigmas.iter().find(|&&sigma| sigma <= 0.0) {
        return Err(RetinexError::InvalidSigma(invalid));
    }

    let rgb = image.to_rgb32f();
    let (reflectance, _) = msr_rgb32f_with_illumination(&rgb, sigmas);
    let color_restored = apply_color_restoration(&reflectance, &rgb, gain, offset);
    Ok(float_to_rgb8_scaled(&color_restored))
}

/// Legacy single-scale Retinex (returns normalized 8-bit image)
pub fn single_scale_retinex(image: &DynamicImage, sigma: f32) -> RetinexResult<RgbImage> {
    let output = single_scale_retinex_full(image, sigma)?;
    Ok(normalize_reflectance_for_display(&output.reflectance))
}

/// Legacy multi-scale Retinex (returns normalized 8-bit image)
pub fn multi_scale_retinex(image: &DynamicImage, sigmas: &[f32]) -> RetinexResult<RgbImage> {
    let output = multi_scale_retinex_full(image, sigmas)?;
    Ok(normalize_reflectance_for_display(&output.reflectance))
}

/// Extract illumination at a specific scale - returns visible image
pub fn extract_illumination(image: &DynamicImage, sigma: f32) -> RetinexResult<RgbImage> {
    if sigma <= 0.0 {
        return Err(RetinexError::InvalidSigma(sigma));
    }

    let rgb = image.to_rgb32f();
    let (width, height) = rgb.dimensions();
    let mut illumination = Rgb32FImage::new(width, height);

    for channel in 0..3 {
        let channel_image = extract_channel(&rgb, channel);
        let blurred = gaussian_blur_f32(&channel_image, sigma);

        for y in 0..height {
            for x in 0..width {
                let l = blurred.get_pixel(x, y)[0];
                illumination.get_pixel_mut(x, y).0[channel] = l;
            }
        }
    }

    // Illumination is in [0, 1] range, scale to [0, 255]
    Ok(float_to_rgb8_scaled(&illumination))
}

fn ssr_rgb32f_with_illumination(image: &Rgb32FImage, sigma: f32) -> (Rgb32FImage, Rgb32FImage) {
    let (width, height) = image.dimensions();
    let mut reflectance = Rgb32FImage::new(width, height);
    let mut illumination = Rgb32FImage::new(width, height);

    for channel in 0..3 {
        let channel_image = extract_channel(image, channel);
        let blurred = gaussian_blur_f32(&channel_image, sigma);

        for y in 0..height {
            for x in 0..width {
                let i = channel_image.get_pixel(x, y)[0];
                let l = blurred.get_pixel(x, y)[0];

                // Store illumination (in [0, 1] range)
                illumination.get_pixel_mut(x, y).0[channel] = l;

                // Compute reflectance in log domain
                // log(I) - log(L) = log(I/L)
                // When I ≈ L (smooth areas), result ≈ 0
                // When I > L (details), result > 0
                // When I < L (shadows), result < 0
                let value = (i + EPSILON).ln() - (l + EPSILON).ln();
                reflectance.get_pixel_mut(x, y).0[channel] = value;
            }
        }
    }

    (reflectance, illumination)
}

fn msr_rgb32f_with_illumination(image: &Rgb32FImage, sigmas: &[f32]) -> (Rgb32FImage, Rgb32FImage) {
    let (width, height) = image.dimensions();
    let mut reflectance_acc = Rgb32FImage::new(width, height);
    let mut illumination_acc = Rgb32FImage::new(width, height);

    for &sigma in sigmas {
        let (reflectance, illumination) = ssr_rgb32f_with_illumination(image, sigma);

        for (acc_pixel, resp_pixel) in reflectance_acc.pixels_mut().zip(reflectance.pixels()) {
            for channel in 0..3 {
                acc_pixel.0[channel] += resp_pixel.0[channel];
            }
        }

        for (acc_pixel, illum_pixel) in illumination_acc.pixels_mut().zip(illumination.pixels()) {
            for channel in 0..3 {
                acc_pixel.0[channel] += illum_pixel.0[channel];
            }
        }
    }

    let weight = 1.0 / sigmas.len() as f32;
    for pixel in reflectance_acc.pixels_mut() {
        for channel in 0..3 {
            pixel.0[channel] *= weight;
        }
    }

    for pixel in illumination_acc.pixels_mut() {
        for channel in 0..3 {
            pixel.0[channel] *= weight;
        }
    }

    (reflectance_acc, illumination_acc)
}

/// Apply color restoration to prevent grayscale output (MSRCR)
/// Formula: R_final = R_msr * (I_c / sum(I)) * 3
/// The factor of 3 normalizes the color weights (since sum of ratios = 1)
fn apply_color_restoration(
    reflectance: &Rgb32FImage,
    original: &Rgb32FImage,
    _gain: f32,
    _offset: f32,
) -> Rgb32FImage {
    let (width, height) = reflectance.dimensions();
    let mut result = Rgb32FImage::new(width, height);

    // First, normalize reflectance to [0, 1] using percentile clipping
    let mut all_refl: Vec<f32> = reflectance
        .pixels()
        .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
        .collect();
    all_refl.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = all_refl.len();
    let low_idx = (n as f32 * 0.02) as usize;
    let high_idx = (n as f32 * 0.98) as usize;
    let low_val = all_refl[low_idx.min(n - 1)];
    let high_val = all_refl[high_idx.min(n - 1)];
    let refl_range = (high_val - low_val).max(EPSILON);

    for y in 0..height {
        for x in 0..width {
            let r = reflectance.get_pixel(x, y);
            let orig = original.get_pixel(x, y);

            // Sum of original channels for color restoration factor
            let sum_orig: f32 = orig.0.iter().sum::<f32>().max(EPSILON);

            for channel in 0..3 {
                // Clip and normalize reflectance to [0, 1]
                let clipped = r.0[channel].clamp(low_val, high_val);
                let norm_refl = (clipped - low_val) / refl_range;

                // Color restoration factor: channel_value / sum * 3
                // The *3 compensates for dividing by sum of 3 channels
                let color_factor = (orig.0[channel] / sum_orig) * 3.0;

                // Apply color restoration and scale to [0, 255]
                let restored = norm_refl * color_factor * 255.0;
                result.get_pixel_mut(x, y).0[channel] = restored.clamp(0.0, 255.0);
            }
        }
    }

    result
}

fn extract_channel(image: &Rgb32FImage, channel: usize) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (width, height) = image.dimensions();
    let mut buffer: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let value = image.get_pixel(x, y).0[channel];
            buffer.put_pixel(x, y, Luma([value]));
        }
    }

    buffer
}

/// Normalize reflectance for display
/// Reflectance is in log domain (typically -5 to +5 range)
/// We need to map this to [0, 255] for visualization
/// Uses percentile clipping to handle outliers
pub fn normalize_reflectance(image: &Rgb32FImage) -> RgbImage {
    let (width, height) = image.dimensions();

    // Find global min/max across all channels
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for pixel in image.pixels() {
        for channel in 0..3 {
            let v = pixel.0[channel];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
    }

    // Use percentile-based clipping to handle outliers
    // Collect all values to compute percentiles
    let mut all_values: Vec<f32> = image
        .pixels()
        .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
        .collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = all_values.len();
    let low_idx = (n as f32 * 0.02) as usize; // 2nd percentile
    let high_idx = (n as f32 * 0.98) as usize; // 98th percentile
    let low_val = all_values[low_idx.min(n - 1)];
    let high_val = all_values[high_idx.min(n - 1)];

    let range = (high_val - low_val).max(EPSILON);

    let mut output = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let source = image.get_pixel(x, y);
            let mut normalized = [0u8; 3];

            for channel in 0..3 {
                // Clip to percentile range then map to [0, 255]
                let clipped = source.0[channel].clamp(low_val, high_val);
                let scaled = (clipped - low_val) / range;
                normalized[channel] = (scaled.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }

            output.put_pixel(x, y, Rgb(normalized));
        }
    }

    output
}

fn normalize_reflectance_for_display(image: &Rgb32FImage) -> RgbImage {
    normalize_reflectance(image)
}

/// Per-channel normalization (for comparison - causes grayscale effect)
pub fn normalize_per_channel(image: &Rgb32FImage) -> RgbImage {
    let (width, height) = image.dimensions();
    let mut min_vals = [f32::INFINITY; 3];
    let mut max_vals = [f32::NEG_INFINITY; 3];

    for pixel in image.pixels() {
        for channel in 0..3 {
            let value = pixel.0[channel];
            if value < min_vals[channel] {
                min_vals[channel] = value;
            }
            if value > max_vals[channel] {
                max_vals[channel] = value;
            }
        }
    }

    let mut output = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let source = image.get_pixel(x, y);
            let mut normalized = [0u8; 3];

            for channel in 0..3 {
                let min = min_vals[channel];
                let max = max_vals[channel];
                let value = source.0[channel];

                let scaled = if (max - min).abs() < EPSILON {
                    0.0
                } else {
                    (value - min) / (max - min)
                };

                normalized[channel] = (scaled.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }

            output.put_pixel(x, y, Rgb(normalized));
        }
    }

    output
}

/// Convert float image [0, 1] or [0, 255] to 8-bit RGB, scaling appropriately
fn float_to_rgb8_scaled(image: &Rgb32FImage) -> RgbImage {
    let (width, height) = image.dimensions();

    // Detect if image is in [0, 1] or [0, 255] range by checking max value
    let max_val = image
        .pixels()
        .flat_map(|p| p.0.iter().copied())
        .fold(0.0f32, |a, b| a.max(b));

    let scale = if max_val <= 1.5 {
        // Input is in [0, 1] range, scale to [0, 255]
        255.0
    } else {
        // Input is already in [0, 255] range or higher
        1.0
    };

    let mut output = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let r = (pixel.0[0] * scale).clamp(0.0, 255.0) as u8;
            let g = (pixel.0[1] * scale).clamp(0.0, 255.0) as u8;
            let b = (pixel.0[2] * scale).clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    output
}
