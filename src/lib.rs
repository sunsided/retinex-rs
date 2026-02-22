//! Retinex image enhancement library
//!
//! This library implements single-scale and multi-scale Retinex algorithms
//! for image enhancement, with optional color restoration (MSRCR).
//!
//! # Overview
//!
//! Retinex separates an image into illumination (slowly varying lighting)
//! and reflectance (object colors). The basic formula is:
//!
//! ```text
//! I(x,y) = R(x,y) × L(x,y)
//! ```
//!
//! Where `I` is the observed image, `R` is reflectance, and `L` is illumination.
//!
//! # Algorithms
//!
//! ## Single-Scale Retinex (SSR)
//! Uses one Gaussian blur scale:
//! ```text
//! R = log(I) - log(G_σ * I)
//! ```
//!
//! ## Multi-Scale Retinex (MSR)
//! Averages multiple scales for better results:
//! ```text
//! R = Σ w_i [ log(I) - log(G_σi * I) ]
//! ```
//!
//! ## MSRCR (Multi-Scale Retinex with Color Restoration)
//! Applies color restoration to prevent grayscale output:
//! ```text
//! R_final = R_msr × (I_c / sum(I)) × 3
//! ```
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use retinex::{multi_scale_retinex_color_restored, single_scale_retinex};
//!
//! // Single-scale (grayscale output)
//! let image = image::open("images/house.jpg").unwrap();
//! let result = single_scale_retinex(&image, 15.0).unwrap();
//! result.save("output.jpg").unwrap();
//!
//! // Multi-scale with color restoration (recommended)
//! let result = multi_scale_retinex_color_restored(
//!     &image,
//!     &[15.0, 80.0, 250.0]
//! ).unwrap();
//! result.save("color_output.jpg").unwrap();
//! ```
//!
//! # Value Ranges
//!
//! All processing is done in the [0, 1] range internally:
//! - Input images are converted to `Rgb32FImage` with values in [0, 1]
//! - Illumination is stored in [0, 1]
//! - Reflectance is in log-domain (can be negative)
//! - Output is converted to 8-bit RGB [0, 255] only at export time

use image::{DynamicImage, ImageBuffer, Luma, Rgb, Rgb32FImage, RgbImage};
use imageproc::filter::gaussian_blur_f32;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

const EPSILON: f32 = 1e-6;

/// Errors that can occur during Retinex processing
#[derive(Debug)]
pub enum RetinexError {
    /// No sigma values were provided for multi-scale processing
    EmptySigmaSet,
    /// A sigma value was not positive
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

/// Result type for Retinex operations
pub type RetinexResult<T> = Result<T, RetinexError>;

/// Output from Retinex processing containing both reflectance and illumination
///
/// The `reflectance` field contains log-domain values (can be negative).
/// The `illumination` field contains values in [0, 1] range.
///
/// Use [`normalize_reflectance`] to convert reflectance to a displayable image,
/// or access the raw data for custom processing.
#[derive(Debug, Clone)]
pub struct RetinexOutput {
    /// The estimated reflectance in log-domain (unnormalized)
    ///
    /// Values are in log space: log(I) - log(L)
    /// Use [`normalize_reflectance`] to convert to displayable image
    pub reflectance: Rgb32FImage,
    /// The estimated illumination (average of blurred versions across scales)
    ///
    /// Values are in [0, 1] range
    pub illumination: Rgb32FImage,
}

/// Single-scale Retinex returning both reflectance and illumination components
///
/// # Arguments
///
/// * `image` - Input image (will be converted to RGB32F)
/// * `sigma` - Gaussian blur radius (must be positive)
///
/// # Returns
///
/// Returns a [`RetinexOutput`] containing both reflectance (log-domain) and
/// illumination ([0, 1] range).
///
/// # Errors
///
/// Returns [`RetinexError::InvalidSigma`] if sigma is not positive.
///
/// # Example
///
/// ```rust,no_run
/// use retinex::single_scale_retinex_full;
///
/// let image = image::open("images/house.jpg").unwrap();
/// let output = single_scale_retinex_full(&image, 15.0).unwrap();
///
/// // Access raw components
/// let refl = &output.reflectance;
/// let illum = &output.illumination;
/// ```
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

/// Multi-scale Retinex returning both reflectance and illumination components
///
/// Computes Retinex at multiple Gaussian scales and averages the results.
/// This balances local detail enhancement (small sigma) with global
/// illumination correction (large sigma).
///
/// # Arguments
///
/// * `image` - Input image (will be converted to RGB32F)
/// * `sigmas` - Slice of Gaussian blur radii (must be non-empty, all positive)
///
/// # Returns
///
/// Returns a [`RetinexOutput`] containing averaged reflectance and illumination.
///
/// # Errors
///
/// - Returns [`RetinexError::EmptySigmaSet`] if sigmas is empty
/// - Returns [`RetinexError::InvalidSigma`] if any sigma is not positive
///
/// # Example
///
/// ```rust,no_run
/// use retinex::multi_scale_retinex_full;
///
/// let image = image::open("images/house.jpg").unwrap();
/// let output = multi_scale_retinex_full(&image, &[15.0, 80.0, 250.0]).unwrap();
/// ```
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
///
/// Applies the MSRCR algorithm which prevents the grayscale output that
/// basic Retinex produces by restoring color ratios from the original image.
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigma` - Gaussian blur radius
///
/// # Returns
///
/// Returns an 8-bit RGB image with color restoration applied.
///
/// # Errors
///
/// Returns [`RetinexError::InvalidSigma`] if sigma is not positive.
pub fn single_scale_retinex_color_restored(
    image: &DynamicImage,
    sigma: f32,
) -> RetinexResult<RgbImage> {
    if sigma <= 0.0 {
        return Err(RetinexError::InvalidSigma(sigma));
    }

    let rgb = image.to_rgb32f();
    let (reflectance, _) = ssr_rgb32f_with_illumination(&rgb, sigma);
    let color_restored = apply_color_restoration(&reflectance, &rgb);
    Ok(float_to_rgb8(&color_restored))
}

/// Multi-scale Retinex with color restoration (MSRCR)
///
/// The recommended approach for best results. Combines multi-scale processing
/// with color restoration to produce natural-looking enhanced images.
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigmas` - Slice of Gaussian blur radii (e.g., `[15.0, 80.0, 250.0]`)
///
/// # Returns
///
/// Returns an 8-bit RGB image with color restoration applied.
///
/// # Errors
///
/// - Returns [`RetinexError::EmptySigmaSet`] if sigmas is empty
/// - Returns [`RetinexError::InvalidSigma`] if any sigma is not positive
///
/// # Example
///
/// ```rust,no_run
/// use retinex::multi_scale_retinex_color_restored;
///
/// let image = image::open("images/house.jpg").unwrap();
/// let result = multi_scale_retinex_color_restored(
///     &image,
///     &[15.0, 80.0, 250.0]
/// ).unwrap();
/// result.save("output.jpg").unwrap();
/// ```
pub fn multi_scale_retinex_color_restored(
    image: &DynamicImage,
    sigmas: &[f32],
) -> RetinexResult<RgbImage> {
    if sigmas.is_empty() {
        return Err(RetinexError::EmptySigmaSet);
    }

    if let Some(&invalid) = sigmas.iter().find(|&&sigma| sigma <= 0.0) {
        return Err(RetinexError::InvalidSigma(invalid));
    }

    let rgb = image.to_rgb32f();
    let (reflectance, _) = msr_rgb32f_with_illumination(&rgb, sigmas);
    let color_restored = apply_color_restoration(&reflectance, &rgb);
    Ok(float_to_rgb8(&color_restored))
}

/// Single-scale Retinex (returns normalized 8-bit grayscale image)
///
/// Produces a high-contrast grayscale image by normalizing the reflectance
/// using percentile-based clipping.
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigma` - Gaussian blur radius
///
/// # Returns
///
/// Returns an 8-bit RGB image (grayscale appearance).
///
/// # Errors
///
/// Returns [`RetinexError::InvalidSigma`] if sigma is not positive.
pub fn single_scale_retinex(image: &DynamicImage, sigma: f32) -> RetinexResult<RgbImage> {
    let output = single_scale_retinex_full(image, sigma)?;
    Ok(normalize_reflectance(&output.reflectance))
}

/// Multi-scale Retinex (returns normalized 8-bit grayscale image)
///
/// Produces a high-contrast grayscale image using multiple scales.
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigmas` - Slice of Gaussian blur radii
///
/// # Returns
///
/// Returns an 8-bit RGB image (grayscale appearance).
///
/// # Errors
///
/// - Returns [`RetinexError::EmptySigmaSet`] if sigmas is empty
/// - Returns [`RetinexError::InvalidSigma`] if any sigma is not positive
pub fn multi_scale_retinex(image: &DynamicImage, sigmas: &[f32]) -> RetinexResult<RgbImage> {
    let output = multi_scale_retinex_full(image, sigmas)?;
    Ok(normalize_reflectance(&output.reflectance))
}

/// Extract illumination at a specific scale
///
/// Returns the estimated illumination component as a displayable image.
/// This is the blurred version of the input that represents the
/// slowly-varying lighting component.
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigma` - Gaussian blur radius
///
/// # Returns
///
/// Returns an 8-bit RGB image showing the illumination.
///
/// # Errors
///
/// Returns [`RetinexError::InvalidSigma`] if sigma is not positive.
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

    Ok(float_to_rgb8(&illumination))
}

/// Normalize log-domain reflectance for display using percentile clipping
///
/// Converts log-domain reflectance values to a displayable 8-bit image.
/// Uses 2nd-98th percentile clipping to handle outliers.
///
/// # Arguments
///
/// * `image` - Log-domain reflectance image (any range)
///
/// # Returns
///
/// Returns an 8-bit RGB image.
///
/// # Example
///
/// ```rust,no_run
/// use retinex::{single_scale_retinex_full, normalize_reflectance};
///
/// let image = image::open("images/house.jpg").unwrap();
/// let output = single_scale_retinex_full(&image, 15.0).unwrap();
/// let display = normalize_reflectance(&output.reflectance);
/// ```
pub fn normalize_reflectance(image: &Rgb32FImage) -> RgbImage {
    let (width, height) = image.dimensions();

    let mut all_values: Vec<f32> = image
        .pixels()
        .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
        .collect();

    #[cfg(feature = "rayon")]
    {
        all_values.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }
    #[cfg(not(feature = "rayon"))]
    {
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    let n = all_values.len();
    let low_idx = (n as f32 * 0.02) as usize;
    let high_idx = (n as f32 * 0.98) as usize;
    let low_val = all_values[low_idx.min(n - 1)];
    let high_val = all_values[high_idx.min(n - 1)];
    let range = (high_val - low_val).max(EPSILON);

    let mut output = RgbImage::new(width, height);

    #[cfg(feature = "rayon")]
    {
        let rows: Vec<_> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row_pixels = vec![[0u8; 3]; width as usize];
                for x in 0..width {
                    let source = image.get_pixel(x, y);
                    for channel in 0..3 {
                        let clipped = source.0[channel].clamp(low_val, high_val);
                        let scaled = (clipped - low_val) / range;
                        row_pixels[x as usize][channel] = (scaled * 255.0 + 0.5) as u8;
                    }
                }
                (y, row_pixels)
            })
            .collect();

        for (y, row_data) in rows {
            for x in 0..width {
                output.put_pixel(x, y, Rgb(row_data[x as usize]));
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for y in 0..height {
            for x in 0..width {
                let source = image.get_pixel(x, y);
                let mut normalized = [0u8; 3];

                for channel in 0..3 {
                    let clipped = source.0[channel].clamp(low_val, high_val);
                    let scaled = (clipped - low_val) / range;
                    normalized[channel] = (scaled * 255.0 + 0.5) as u8;
                }

                output.put_pixel(x, y, Rgb(normalized));
            }
        }
    }

    output
}

/// Per-channel normalization (demonstrates the grayscale problem)
///
/// Normalizes each channel independently, which destroys color ratios
/// and produces a grayscale image. This is provided for educational
/// purposes to demonstrate why color restoration is needed.
///
/// # Arguments
///
/// * `image` - Input image (any range)
///
/// # Returns
///
/// Returns an 8-bit RGB image with grayscale appearance.
pub fn normalize_per_channel(image: &Rgb32FImage) -> RgbImage {
    let (width, height) = image.dimensions();

    #[cfg(feature = "rayon")]
    let (min_vals, max_vals): ([f32; 3], [f32; 3]) = {
        let channel_stats: Vec<_> = (0..3)
            .into_par_iter()
            .map(|channel| {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;
                for pixel in image.pixels() {
                    let value = pixel.0[channel];
                    if value < min_val {
                        min_val = value;
                    }
                    if value > max_val {
                        max_val = value;
                    }
                }
                (channel, min_val, max_val)
            })
            .collect();

        let mut min_vals = [f32::INFINITY; 3];
        let mut max_vals = [f32::NEG_INFINITY; 3];
        for (channel, min_val, max_val) in channel_stats {
            min_vals[channel] = min_val;
            max_vals[channel] = max_val;
        }
        (min_vals, max_vals)
    };

    #[cfg(not(feature = "rayon"))]
    let (min_vals, max_vals): ([f32; 3], [f32; 3]) = {
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
        (min_vals, max_vals)
    };

    let mut output = RgbImage::new(width, height);

    #[cfg(feature = "rayon")]
    {
        let rows: Vec<_> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row_pixels = vec![[0u8; 3]; width as usize];
                for x in 0..width {
                    let source = image.get_pixel(x, y);

                    for channel in 0..3 {
                        let min = min_vals[channel];
                        let max = max_vals[channel];
                        let value = source.0[channel];

                        let scaled = if (max - min).abs() < EPSILON {
                            0.0
                        } else {
                            (value - min) / (max - min)
                        };

                        row_pixels[x as usize][channel] = (scaled * 255.0 + 0.5) as u8;
                    }
                }
                (y, row_pixels)
            })
            .collect();

        for (y, row_data) in rows {
            for x in 0..width {
                output.put_pixel(x, y, Rgb(row_data[x as usize]));
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
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

                    normalized[channel] = (scaled * 255.0 + 0.5) as u8;
                }

                output.put_pixel(x, y, Rgb(normalized));
            }
        }
    }

    output
}

/// Clamp reflectance to physical constraint (reflectance ≤ 1.0 in linear space)
///
/// In log space, this means log-reflectance ≤ 0. If the maximum reflectance
/// is greater than 0, shifts all values down and adds the offset to illumination.
///
/// This enforces the physical constraint that objects cannot reflect more
/// light than they receive.
///
/// # Arguments
///
/// * `reflectance` - Log-domain reflectance (modified in place)
/// * `illumination` - Illumination component (modified in place)
pub fn clamp_reflectance(reflectance: &mut Rgb32FImage, illumination: &mut Rgb32FImage) {
    #[cfg(feature = "rayon")]
    let max_refl: f32 = {
        reflectance
            .pixels()
            .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
            .par_bridge()
            .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b))
    };

    #[cfg(not(feature = "rayon"))]
    let max_refl: f32 = {
        let mut max_refl = f32::NEG_INFINITY;
        for pixel in reflectance.pixels() {
            for channel in 0..3 {
                if pixel.0[channel] > max_refl {
                    max_refl = pixel.0[channel];
                }
            }
        }
        max_refl
    };

    if max_refl > 0.0 {
        #[cfg(feature = "rayon")]
        {
            let refl_data: Vec<_> = reflectance.pixels_mut().collect();
            let illum_data: Vec<_> = illumination.pixels_mut().collect();

            refl_data
                .into_par_iter()
                .zip(illum_data.into_par_iter())
                .for_each(|(refl_pixel, illum_pixel)| {
                    for channel in 0..3 {
                        let r = refl_pixel.0[channel];
                        let l = illum_pixel.0[channel];
                        refl_pixel.0[channel] = r - max_refl;
                        illum_pixel.0[channel] = (l + max_refl).clamp(0.0, 1.0);
                    }
                });
        }

        #[cfg(not(feature = "rayon"))]
        {
            for y in 0..height {
                for x in 0..width {
                    for channel in 0..3 {
                        let r = reflectance.get_pixel(x, y).0[channel];
                        let l = illumination.get_pixel(x, y).0[channel];
                        reflectance.get_pixel_mut(x, y).0[channel] = r - max_refl;
                        illumination.get_pixel_mut(x, y).0[channel] =
                            (l + max_refl).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
}

// Internal helper functions

fn ssr_rgb32f_with_illumination(image: &Rgb32FImage, sigma: f32) -> (Rgb32FImage, Rgb32FImage) {
    let (width, height) = image.dimensions();
    let mut reflectance = Rgb32FImage::new(width, height);
    let mut illumination = Rgb32FImage::new(width, height);

    #[cfg(feature = "rayon")]
    {
        let channels: Vec<_> = (0..3)
            .into_par_iter()
            .map(|channel| {
                let channel_image = extract_channel(image, channel);
                let blurred = gaussian_blur_f32(&channel_image, sigma);

                let mut refl_channel = vec![0.0f32; (width * height) as usize];
                let mut illum_channel = vec![0.0f32; (width * height) as usize];

                for y in 0..height {
                    for x in 0..width {
                        let i = channel_image.get_pixel(x, y)[0];
                        let l = blurred.get_pixel(x, y)[0];

                        let idx = (y * width + x) as usize;
                        illum_channel[idx] = l;
                        refl_channel[idx] = (i + EPSILON).ln() - (l + EPSILON).ln();
                    }
                }

                (channel, refl_channel, illum_channel)
            })
            .collect();

        for (channel, refl_data, illum_data) in channels {
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    reflectance.get_pixel_mut(x, y).0[channel] = refl_data[idx];
                    illumination.get_pixel_mut(x, y).0[channel] = illum_data[idx];
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for channel in 0..3 {
            let channel_image = extract_channel(image, channel);
            let blurred = gaussian_blur_f32(&channel_image, sigma);

            for y in 0..height {
                for x in 0..width {
                    let i = channel_image.get_pixel(x, y)[0];
                    let l = blurred.get_pixel(x, y)[0];

                    illumination.get_pixel_mut(x, y).0[channel] = l;
                    let value = (i + EPSILON).ln() - (l + EPSILON).ln();
                    reflectance.get_pixel_mut(x, y).0[channel] = value;
                }
            }
        }
    }

    (reflectance, illumination)
}

fn msr_rgb32f_with_illumination(image: &Rgb32FImage, sigmas: &[f32]) -> (Rgb32FImage, Rgb32FImage) {
    let (width, height) = image.dimensions();

    #[cfg(feature = "rayon")]
    {
        let results: Vec<_> = sigmas
            .par_iter()
            .map(|&sigma| ssr_rgb32f_with_illumination(image, sigma))
            .collect();

        let mut reflectance_acc = Rgb32FImage::new(width, height);
        let mut illumination_acc = Rgb32FImage::new(width, height);

        for (reflectance, illumination) in results {
            for (acc_pixel, resp_pixel) in reflectance_acc.pixels_mut().zip(reflectance.pixels()) {
                for channel in 0..3 {
                    acc_pixel.0[channel] += resp_pixel.0[channel];
                }
            }

            for (acc_pixel, illum_pixel) in illumination_acc.pixels_mut().zip(illumination.pixels())
            {
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

    #[cfg(not(feature = "rayon"))]
    {
        let mut reflectance_acc = Rgb32FImage::new(width, height);
        let mut illumination_acc = Rgb32FImage::new(width, height);

        for &sigma in sigmas {
            let (reflectance, illumination) = ssr_rgb32f_with_illumination(image, sigma);

            for (acc_pixel, resp_pixel) in reflectance_acc.pixels_mut().zip(reflectance.pixels()) {
                for channel in 0..3 {
                    acc_pixel.0[channel] += resp_pixel.0[channel];
                }
            }

            for (acc_pixel, illum_pixel) in illumination_acc.pixels_mut().zip(illumination.pixels())
            {
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
}

fn apply_color_restoration(reflectance: &Rgb32FImage, original: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = reflectance.dimensions();
    let mut result = Rgb32FImage::new(width, height);

    let mut all_refl: Vec<f32> = reflectance
        .pixels()
        .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
        .collect();

    #[cfg(feature = "rayon")]
    {
        all_refl.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }
    #[cfg(not(feature = "rayon"))]
    {
        all_refl.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    let n = all_refl.len();
    let low_idx = (n as f32 * 0.02) as usize;
    let high_idx = (n as f32 * 0.98) as usize;
    let low_val = all_refl[low_idx.min(n - 1)];
    let high_val = all_refl[high_idx.min(n - 1)];
    let refl_range = (high_val - low_val).max(EPSILON);

    #[cfg(feature = "rayon")]
    {
        let rows: Vec<_> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row_result = vec![[0.0f32; 3]; width as usize];
                for x in 0..width {
                    let r = reflectance.get_pixel(x, y);
                    let orig = original.get_pixel(x, y);
                    let sum_orig: f32 = orig.0.iter().sum::<f32>().max(EPSILON);

                    for channel in 0..3 {
                        let clipped = r.0[channel].clamp(low_val, high_val);
                        let norm_refl = (clipped - low_val) / refl_range;
                        let color_factor = (orig.0[channel] / sum_orig) * 3.0;
                        let restored = norm_refl * color_factor;
                        row_result[x as usize][channel] = restored.clamp(0.0, 1.0);
                    }
                }
                (y, row_result)
            })
            .collect();

        for (y, row_data) in rows {
            for x in 0..width {
                result.get_pixel_mut(x, y).0 = row_data[x as usize];
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for y in 0..height {
            for x in 0..width {
                let r = reflectance.get_pixel(x, y);
                let orig = original.get_pixel(x, y);
                let sum_orig: f32 = orig.0.iter().sum::<f32>().max(EPSILON);

                for channel in 0..3 {
                    let clipped = r.0[channel].clamp(low_val, high_val);
                    let norm_refl = (clipped - low_val) / refl_range;
                    let color_factor = (orig.0[channel] / sum_orig) * 3.0;
                    let restored = norm_refl * color_factor;
                    result.get_pixel_mut(x, y).0[channel] = restored.clamp(0.0, 1.0);
                }
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

fn float_to_rgb8(image: &Rgb32FImage) -> RgbImage {
    let (width, height) = image.dimensions();
    let mut output = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let r = (pixel.0[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let g = (pixel.0[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let b = (pixel.0[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            output.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    fn load_test_image() -> DynamicImage {
        image::open("images/house.jpg").expect("Failed to load test image")
    }

    #[test]
    fn test_single_scale_retinex_basic() {
        let img = load_test_image();
        let result = single_scale_retinex(&img, 15.0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_multi_scale_retinex_basic() {
        let img = load_test_image();
        // Use smaller sigmas for faster tests
        let result = multi_scale_retinex(&img, &[10.0, 40.0]);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_color_restored_single_scale() {
        let img = load_test_image();
        let result = single_scale_retinex_color_restored(&img, 15.0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_color_restored_multi_scale() {
        let img = load_test_image();
        // Use smaller sigmas for faster tests
        let result = multi_scale_retinex_color_restored(&img, &[10.0, 40.0]);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_extract_illumination() {
        let img = load_test_image();
        let result = extract_illumination(&img, 15.0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_single_scale_full_output() {
        let img = load_test_image();
        let result = single_scale_retinex_full(&img, 15.0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.reflectance.dimensions(), img.dimensions());
        assert_eq!(output.illumination.dimensions(), img.dimensions());
    }

    #[test]
    fn test_multi_scale_full_output() {
        let img = load_test_image();
        // Use smaller sigmas for faster tests
        let result = multi_scale_retinex_full(&img, &[10.0, 40.0]);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.reflectance.dimensions(), img.dimensions());
        assert_eq!(output.illumination.dimensions(), img.dimensions());
    }

    #[test]
    fn test_invalid_sigma_single_scale() {
        let img = load_test_image();
        let result = single_scale_retinex(&img, -5.0);
        assert!(matches!(result, Err(RetinexError::InvalidSigma(-5.0))));
    }

    #[test]
    fn test_invalid_sigma_multi_scale() {
        let img = load_test_image();
        let result = multi_scale_retinex(&img, &[10.0, -5.0, 40.0]);
        assert!(matches!(result, Err(RetinexError::InvalidSigma(-5.0))));
    }

    #[test]
    fn test_empty_sigma_set() {
        let img = load_test_image();
        let result = multi_scale_retinex(&img, &[]);
        assert!(matches!(result, Err(RetinexError::EmptySigmaSet)));
    }

    #[test]
    fn test_normalize_reflectance() {
        let img = load_test_image();
        let rgb = img.to_rgb32f();
        let output = normalize_reflectance(&rgb);
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_normalize_per_channel() {
        let img = load_test_image();
        let rgb = img.to_rgb32f();
        let output = normalize_per_channel(&rgb);
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_clamp_reflectance() {
        let img = load_test_image();
        let mut output = single_scale_retinex_full(&img, 15.0).unwrap();

        // Store original max reflectance
        let original_max = output
            .reflectance
            .pixels()
            .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        clamp_reflectance(&mut output.reflectance, &mut output.illumination);

        // After clamping, max reflectance should be <= 0
        let new_max = output
            .reflectance
            .pixels()
            .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        assert!(new_max <= 0.0 || original_max <= 0.0);
    }

    #[test]
    fn test_illumination_range() {
        let img = load_test_image();
        let output = single_scale_retinex_full(&img, 15.0).unwrap();

        // Illumination should be in [0, 1] range
        for pixel in output.illumination.pixels() {
            for channel in 0..3 {
                assert!(
                    pixel.0[channel] >= 0.0 && pixel.0[channel] <= 1.0,
                    "Illumination value {} out of range [0, 1]",
                    pixel.0[channel]
                );
            }
        }
    }
}
