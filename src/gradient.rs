use image::{DynamicImage, Rgb32FImage};

use crate::{EPSILON, IntrinsicOutput, RetinexError, RetinexResult};

const DEFAULT_THRESHOLD: f32 = 0.05;

struct Constraint {
    p: usize, // "from" pixel index
    q: usize, // "to" pixel index (always q > p)
    rhs: f32, // 0.0 for shading; actual gradient for reflectance edge
}

fn classify_gradients(
    log_luma: &[f32],
    width: u32,
    height: u32,
    threshold: f32,
) -> Vec<Constraint> {
    let w = width as usize;
    let h = height as usize;
    let mut constraints = Vec::with_capacity((w - 1) * h + w * (h - 1));

    // Horizontal pairs: pixel (x,y) -> (x+1,y)
    for y in 0..h {
        for x in 0..(w - 1) {
            let p = y * w + x;
            let q = y * w + x + 1;
            let g = log_luma[q] - log_luma[p];
            let rhs = if g.abs() > threshold { g } else { 0.0 };
            constraints.push(Constraint { p, q, rhs });
        }
    }

    // Vertical pairs: pixel (x,y) -> (x,y+1)
    for y in 0..(h - 1) {
        for x in 0..w {
            let p = y * w + x;
            let q = (y + 1) * w + x;
            let g = log_luma[q] - log_luma[p];
            let rhs = if g.abs() > threshold { g } else { 0.0 };
            constraints.push(Constraint { p, q, rhs });
        }
    }

    constraints
}

// Computes y = (A'A + λI) x matrix-free, in O(constraints) time.
// A'A is the graph Laplacian: for each constraint (p,q), diagonal +1 each, off-diagonal -1.
fn matvec(constraints: &[Constraint], lambda: f64, x: &[f64], y: &mut [f64]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(yi, xi)| *yi = lambda * xi);
    for c in constraints {
        let diff = x[c.p] - x[c.q];
        y[c.p] += diff;
        y[c.q] -= diff;
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(yi, xi)| *yi += alpha * xi);
}

fn solve_log_reflectance(constraints: &[Constraint], n: usize, lambda: f64) -> Vec<f64> {
    // Assemble A'b: for each constraint (p, q, rhs), A'b[p] -= rhs, A'b[q] += rhs.
    let mut atb = vec![0.0f64; n];
    for c in constraints {
        let rhs = c.rhs as f64;
        atb[c.p] -= rhs;
        atb[c.q] += rhs;
    }

    let b_norm_sq = dot(&atb, &atb);
    if b_norm_sq < f64::EPSILON {
        return vec![0.0; n];
    }

    // Jacobi (diagonal) preconditioner: M[i] = degree(i) + λ.
    let mut diag = vec![lambda; n];
    for c in constraints {
        diag[c.p] += 1.0;
        diag[c.q] += 1.0;
    }

    // Preconditioned conjugate gradient.
    // With λ ≈ 1e-3, condition number ≈ 4/λ ≈ 4000 → ~300-500 iterations to converge.
    let tol_sq = 1e-8 * b_norm_sq;
    let max_iter = (n as f64).sqrt() as usize * 4 + 1000;

    let mut x = vec![0.0f64; n];
    let mut r = atb; // r₀ = b − A·0 = b
    let mut z: Vec<f64> = r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    let mut ap = vec![0.0f64; n];

    for _ in 0..max_iter {
        ap.fill(0.0);
        matvec(constraints, lambda, &p, &mut ap);

        let pap = dot(&p, &ap);
        if pap < 1e-14 {
            break;
        }

        let alpha = rz / pap;
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);

        if dot(&r, &r) < tol_sq {
            break;
        }

        z.iter_mut()
            .zip(r.iter())
            .zip(diag.iter())
            .for_each(|((zi, ri), di)| *zi = ri / di);
        let rz_new = dot(&r, &z);

        let beta = rz_new / rz;
        p.iter_mut()
            .zip(z.iter())
            .for_each(|(pi, zi)| *pi = zi + beta * *pi);
        rz = rz_new;
    }

    x
}

fn enforce_physical_constraint(log_refl: &mut [f64]) {
    let max_val = log_refl.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val > 0.0 {
        for v in log_refl.iter_mut() {
            *v -= max_val;
        }
    }
}

fn reconstruct_color(
    log_refl: &[f64],
    original: &Rgb32FImage,
    width: u32,
    height: u32,
) -> (Rgb32FImage, Rgb32FImage) {
    let mut reflectance = Rgb32FImage::new(width, height);
    let mut shading = Rgb32FImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let orig = original.get_pixel(x, y);

            let luma = 0.2126 * orig.0[0] + 0.7152 * orig.0[1] + 0.0722 * orig.0[2];
            let refl_luma = (log_refl[idx] as f32).exp();
            let ratio = refl_luma / (luma + EPSILON);

            for ch in 0..3 {
                let r = (orig.0[ch] * ratio).clamp(0.0, 1.0);
                // Shading can exceed 1 in dark/low-reflectance regions; only prevent negatives.
                // Clamping to 1 here would break the multiplicative identity I = R * S.
                let s = (orig.0[ch] / (r + EPSILON)).max(0.0);
                reflectance.get_pixel_mut(x, y).0[ch] = r;
                shading.get_pixel_mut(x, y).0[ch] = s;
            }
        }
    }

    (reflectance, shading)
}

pub fn gradient_intrinsic_decomp(
    image: &DynamicImage,
    threshold: Option<f32>,
) -> RetinexResult<IntrinsicOutput> {
    let threshold = threshold.unwrap_or(DEFAULT_THRESHOLD);
    if threshold <= 0.0 {
        return Err(RetinexError::InvalidThreshold(threshold));
    }

    let rgb = image.to_rgb32f();
    let (width, height) = rgb.dimensions();
    let n = (width * height) as usize;

    let log_luma = to_log_luma(&rgb);
    let constraints = classify_gradients(&log_luma, width, height, threshold);
    let mut log_refl = solve_log_reflectance(&constraints, n, 1e-3);
    enforce_physical_constraint(&mut log_refl);

    let (reflectance, shading) = reconstruct_color(&log_refl, &rgb, width, height);

    Ok(IntrinsicOutput {
        reflectance,
        shading,
    })
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
    use image::{GenericImageView, Rgb, Rgb32FImage};

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

    #[test]
    fn test_classify_reflectance_edge() {
        // Large gradient: dark (0.1) to bright (0.9) — should be reflectance edge
        let log_luma = vec![(0.1f32 + 1e-6).ln(), (0.9f32 + 1e-6).ln()];
        let constraints = classify_gradients(&log_luma, 2, 1, 0.05);
        // g ≈ ln(0.9) - ln(0.1) ≈ 2.197, which is >> 0.05
        assert_eq!(constraints.len(), 1);
        let g = log_luma[1] - log_luma[0];
        assert!(
            (constraints[0].rhs - g).abs() < 1e-5,
            "expected rhs={g}, got {}",
            constraints[0].rhs
        );
    }

    #[test]
    fn test_classify_shading_region() {
        // Tiny gradient: nearly equal values — should be shading (rhs = 0)
        let log_luma = vec![(0.5f32 + 1e-6).ln(), (0.5001f32 + 1e-6).ln()];
        let constraints = classify_gradients(&log_luma, 2, 1, 0.05);
        assert_eq!(constraints.len(), 1);
        assert!(
            constraints[0].rhs.abs() < 1e-6,
            "expected rhs=0, got {}",
            constraints[0].rhs
        );
    }

    #[test]
    fn test_constraint_count_2x2() {
        // 2x2 image: 2 horizontal pairs + 2 vertical pairs = 4 constraints
        let log_luma = vec![0.0f32; 4];
        let constraints = classify_gradients(&log_luma, 2, 2, 0.05);
        assert_eq!(constraints.len(), 4);
    }

    #[test]
    fn test_constraint_pixel_indices() {
        // 3x1 image: two horizontal constraints
        // pixels 0,1,2; constraint 0: p=0 q=1; constraint 1: p=1 q=2
        let log_luma = vec![0.0f32; 3];
        let constraints = classify_gradients(&log_luma, 3, 1, 0.05);
        assert_eq!(constraints.len(), 2);
        assert_eq!(constraints[0].p, 0);
        assert_eq!(constraints[0].q, 1);
        assert_eq!(constraints[1].p, 1);
        assert_eq!(constraints[1].q, 2);
    }

    #[test]
    fn test_solver_follows_reflectance_edge() {
        // 2 pixels, large gradient → reflectance edge
        // Solution diff should ≈ log(0.8) - log(0.3)
        let log_luma = vec![(0.3f32 + 1e-6).ln(), (0.8f32 + 1e-6).ln()];
        let constraints = classify_gradients(&log_luma, 2, 1, 0.05);
        let solution = solve_log_reflectance(&constraints, 2, 1e-3);
        assert_eq!(solution.len(), 2);
        let diff = solution[1] - solution[0];
        let expected = (log_luma[1] - log_luma[0]) as f64;
        assert!(
            (diff - expected).abs() < 1e-3,
            "expected diff ≈ {expected:.4}, got {diff:.4}"
        );
    }

    #[test]
    fn test_solver_smooth_for_shading_region() {
        // 2 pixels, tiny gradient → shading → reflectance should be flat
        let log_luma = vec![(0.5f32 + 1e-6).ln(), (0.5001f32 + 1e-6).ln()];
        let constraints = classify_gradients(&log_luma, 2, 1, 0.05);
        let solution = solve_log_reflectance(&constraints, 2, 1e-3);
        let diff = (solution[1] - solution[0]).abs();
        assert!(diff < 1e-3, "reflectance should be smooth, diff = {diff}");
    }

    fn load_test_image() -> image::DynamicImage {
        image::open("images/house.jpg").expect("Failed to load test image")
    }

    #[test]
    fn test_gradient_decomp_basic() {
        let img = load_test_image();
        let result = gradient_intrinsic_decomp(&img, None);
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        let out = result.unwrap();
        assert_eq!(out.reflectance.dimensions(), img.dimensions());
        assert_eq!(out.shading.dimensions(), img.dimensions());
    }

    #[test]
    fn test_gradient_decomp_physical_constraint() {
        let img = load_test_image();
        let out = gradient_intrinsic_decomp(&img, None).unwrap();
        for pixel in out.reflectance.pixels() {
            for ch in 0..3 {
                assert!(
                    pixel.0[ch] <= 1.0 + 1e-5,
                    "reflectance channel {} = {} exceeds 1.0",
                    ch,
                    pixel.0[ch]
                );
            }
        }
    }

    #[test]
    fn test_gradient_decomp_invalid_threshold() {
        let img = load_test_image();
        let result = gradient_intrinsic_decomp(&img, Some(-0.1));
        assert!(matches!(result, Err(RetinexError::InvalidThreshold(_))));
    }

    #[test]
    fn test_gradient_decomp_zero_threshold() {
        let img = load_test_image();
        let result = gradient_intrinsic_decomp(&img, Some(0.0));
        assert!(matches!(result, Err(RetinexError::InvalidThreshold(_))));
    }

    #[test]
    fn test_gradient_decomp_synthetic_step_edge() {
        // 4x4 image with hard luminance step at x=2.
        // Left two columns: ~0.2 (v=51), right two columns: ~0.8 (v=204).
        // After decomp, reflectance should differ significantly across the edge.
        let mut img = image::RgbImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v: u8 = if x < 2 { 51 } else { 204 };
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let out = gradient_intrinsic_decomp(&dyn_img, Some(0.05)).unwrap();

        let r_left = out.reflectance.get_pixel(1, 2).0[0];
        let r_right = out.reflectance.get_pixel(2, 2).0[0];
        assert!(
            (r_right - r_left).abs() > 0.1,
            "expected reflectance edge at x=1/2, got left={r_left:.3}, right={r_right:.3}"
        );
    }
}
