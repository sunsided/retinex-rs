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
//! ```text
//! R = log(I) - log(G_σ * I)
//! ```
//!
//! ## Multi-Scale Retinex (MSR)
//! ```text
//! R = Σ w_i [ log(I) - log(G_σi * I) ]
//! ```
//!
//! ## MSRCR (Multi-Scale Retinex with Color Restoration)
//! ```text
//! R_final = R_msr × (I_c / sum(I)) × 3
//! ```

use image::Rgb32FImage;

pub(crate) const EPSILON: f32 = 1e-6;

/// Errors that can occur during Retinex processing
#[derive(Debug)]
pub enum RetinexError {
    /// No sigma values were provided for multi-scale processing
    EmptySigmaSet,
    /// A sigma value was not positive
    InvalidSigma(f32),
    /// The sparse LDL^T factorization failed (gradient mode)
    #[cfg(feature = "gradient")]
    SolverFailed(String),
    /// A threshold value was not positive (gradient mode)
    #[cfg(feature = "gradient")]
    InvalidThreshold(f32),
}

impl std::fmt::Display for RetinexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetinexError::EmptySigmaSet => write!(f, "expected at least one sigma value"),
            RetinexError::InvalidSigma(value) => write!(f, "sigma must be positive, got {value}"),
            #[cfg(feature = "gradient")]
            RetinexError::SolverFailed(msg) => write!(f, "sparse solver failed: {msg}"),
            #[cfg(feature = "gradient")]
            RetinexError::InvalidThreshold(value) => {
                write!(f, "threshold must be positive, got {value}")
            }
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

/// Output from gradient-based intrinsic image decomposition
///
/// Unlike [`RetinexOutput`], reflectance is in **linear space** ([0, 1])
/// and directly displayable without normalization.
#[cfg(feature = "gradient")]
#[derive(Debug, Clone)]
pub struct IntrinsicOutput {
    /// Reflectance in linear space, values in [0, 1]
    pub reflectance: Rgb32FImage,
    /// Shading (illumination) in linear space, values in [0, 1]
    pub shading: Rgb32FImage,
}

mod retinex;
pub use retinex::{
    clamp_reflectance, extract_illumination, multi_scale_retinex,
    multi_scale_retinex_color_restored, multi_scale_retinex_full, normalize_per_channel,
    normalize_reflectance, single_scale_retinex, single_scale_retinex_color_restored,
    single_scale_retinex_full, suggest_sigma, suggest_sigmas,
};

#[cfg(test)]
mod tests {
    #[cfg(feature = "gradient")]
    use super::*;

    #[test]
    #[cfg(feature = "gradient")]
    fn test_intrinsic_output_fields_exist() {
        let out = IntrinsicOutput {
            reflectance: image::Rgb32FImage::new(1, 1),
            shading: image::Rgb32FImage::new(1, 1),
        };
        assert_eq!(out.reflectance.dimensions(), (1, 1));
        assert_eq!(out.shading.dimensions(), (1, 1));
    }

    #[test]
    #[cfg(feature = "gradient")]
    fn test_solver_failed_error_display() {
        let e = RetinexError::SolverFailed("singular matrix".into());
        assert!(e.to_string().contains("singular matrix"));
    }

    #[test]
    #[cfg(feature = "gradient")]
    fn test_invalid_threshold_error_display() {
        let e = RetinexError::InvalidThreshold(-0.1);
        assert!(e.to_string().contains("-0.1"));
    }
}
