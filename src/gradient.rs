use image::{DynamicImage, Rgb32FImage};

use crate::{EPSILON, IntrinsicOutput, RetinexError, RetinexResult};

const DEFAULT_THRESHOLD: f32 = 0.05;

pub fn gradient_intrinsic_decomp(
    _image: &DynamicImage,
    _threshold: Option<f32>,
) -> RetinexResult<IntrinsicOutput> {
    todo!()
}

fn to_log_luma(image: &Rgb32FImage) -> Vec<f32> {
    image
        .pixels()
        .map(|p| {
            let l = 0.2126 * p.0[0] + 0.7152 * p.0[1] + 0.0722 * p.0[2];
            (l + EPSILON).ln()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, Rgb32FImage};

    #[test]
    fn test_log_luma_red_pixel() {
        // Pure red at 1.0: luma = 0.2126 (Rec. 709)
        let mut img = Rgb32FImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([1.0f32, 0.0, 0.0]));
        let luma = to_log_luma(&img);
        let expected = (0.2126f32 + 1e-6).ln();
        assert!((luma[0] - expected).abs() < 1e-5, "got {}", luma[0]);
    }

    #[test]
    fn test_log_luma_white_pixel() {
        // White: luma = 0.2126 + 0.7152 + 0.0722 = 1.0
        let mut img = Rgb32FImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([1.0f32, 1.0, 1.0]));
        let luma = to_log_luma(&img);
        let expected = (1.0f32 + 1e-6).ln();
        assert!((luma[0] - expected).abs() < 1e-5, "got {}", luma[0]);
    }

    #[test]
    fn test_log_luma_length() {
        let img = Rgb32FImage::new(4, 3);
        let luma = to_log_luma(&img);
        assert_eq!(luma.len(), 12);
    }
}
