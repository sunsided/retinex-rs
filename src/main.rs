use clap::{Parser, ValueEnum};
#[cfg(feature = "gradient")]
use retinex::gradient_intrinsic_decomp;
use retinex::{
    extract_illumination, multi_scale_retinex, multi_scale_retinex_color_restored,
    multi_scale_retinex_full, normalize_reflectance, single_scale_retinex,
    single_scale_retinex_color_restored, single_scale_retinex_full, suggest_sigma, suggest_sigmas,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "retinex")]
#[command(about = "Apply Retinex or gradient intrinsic decomposition to images", long_about = None)]
struct Cli {
    #[arg(help = "Input image file path")]
    input: PathBuf,

    #[arg(help = "Output image file path (reflectance result)")]
    output: PathBuf,

    #[arg(short, long, value_enum, default_value = "single")]
    mode: Mode,

    /// Comma-separated Gaussian blur radii, e.g. `15,80,250` (single/multi modes only)
    #[arg(long, value_delimiter = ',')]
    sigmas: Option<Vec<f32>>,

    /// Auto-select sigma values from image dimensions (single/multi modes only)
    #[arg(long)]
    auto_sigmas: bool,

    /// Gradient classification threshold for reflectance edges (gradient mode only)
    #[arg(long)]
    threshold: Option<f32>,

    /// Save the estimated illumination or shading to a separate file
    #[arg(long)]
    illumination: Option<PathBuf>,

    /// Save the raw reflectance to a separate file
    #[arg(long)]
    reflectance: Option<PathBuf>,

    /// Enable color restoration/MSRCR (single/multi modes only; gradient mode always produces color)
    #[arg(long, default_value_t = false)]
    color_restore: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Single,
    Multi,
    #[cfg(feature = "gradient")]
    Gradient,
}

#[cfg(feature = "gradient")]
fn float32_to_rgb8(img: &image::Rgb32FImage) -> image::RgbImage {
    let (w, h) = img.dimensions();
    let mut out = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            let r = (p.0[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let g = (p.0[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let b = (p.0[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    out
}

fn main() {
    let cli = Cli::parse();

    #[cfg(feature = "gradient")]
    let is_gradient = cli.mode == Mode::Gradient;
    #[cfg(not(feature = "gradient"))]
    let is_gradient = false;

    if cli.auto_sigmas && cli.sigmas.is_some() {
        eprintln!("Cannot use --sigmas and --auto-sigmas together");
        std::process::exit(1);
    }
    if is_gradient && (cli.sigmas.is_some() || cli.auto_sigmas) {
        eprintln!("Warning: --sigmas and --auto-sigmas are ignored in gradient mode");
    }
    if is_gradient && cli.color_restore {
        eprintln!("Warning: --color-restore is ignored in gradient mode (always color output)");
    }

    let image = match image::open(&cli.input) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("Failed to open {}: {err}", cli.input.display());
            std::process::exit(1);
        }
    };

    // Gradient dispatch
    #[cfg(feature = "gradient")]
    if is_gradient {
        let result = gradient_intrinsic_decomp(&image, cli.threshold);
        match result {
            Ok(out) => {
                if let Some(illum_path) = &cli.illumination {
                    let shading_rgb = float32_to_rgb8(&out.shading);
                    if let Err(err) = shading_rgb.save(illum_path) {
                        eprintln!("Failed to save shading {}: {err}", illum_path.display());
                        std::process::exit(1);
                    }
                    println!("Saved shading to {}", illum_path.display());
                }
                let refl_rgb = float32_to_rgb8(&out.reflectance);
                if let Err(err) = refl_rgb.save(&cli.output) {
                    eprintln!("Failed to save {}: {err}", cli.output.display());
                    std::process::exit(1);
                }
                println!("Saved gradient reflectance to {}", cli.output.display());
            }
            Err(err) => {
                eprintln!("Gradient decomposition failed: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    // SSR / MSR path
    let sigmas: Vec<f32> = if cli.auto_sigmas {
        let selected = match cli.mode {
            Mode::Single => vec![suggest_sigma(&image)],
            Mode::Multi => suggest_sigmas(&image).to_vec(),
            #[cfg(feature = "gradient")]
            Mode::Gradient => unreachable!(),
        };
        println!("Auto-selected sigmas: {selected:?}");
        selected
    } else if let Some(provided) = cli.sigmas {
        if let Some(&invalid) = provided.iter().find(|&&sigma| sigma <= 0.0) {
            eprintln!("Sigmas must be positive, got {invalid}");
            std::process::exit(1);
        }
        provided
    } else {
        vec![15.0]
    };

    if let Some(illum_path) = &cli.illumination {
        match extract_illumination(&image, sigmas[0]) {
            Ok(illum) => {
                if let Err(err) = illum.save(illum_path) {
                    eprintln!(
                        "Failed to save illumination {}: {err}",
                        illum_path.display()
                    );
                    std::process::exit(1);
                }
                println!("Saved illumination to {}", illum_path.display());
            }
            Err(err) => {
                eprintln!("Failed to extract illumination: {err}");
                std::process::exit(1);
            }
        }
    }

    if let Some(refl_path) = &cli.reflectance {
        let refl_result = match cli.mode {
            Mode::Single => {
                let output = single_scale_retinex_full(&image, sigmas[0]);
                output.map(|o| normalize_reflectance(&o.reflectance))
            }
            Mode::Multi => {
                let output = multi_scale_retinex_full(&image, &sigmas);
                output.map(|o| normalize_reflectance(&o.reflectance))
            }
            #[cfg(feature = "gradient")]
            Mode::Gradient => unreachable!(),
        };
        match refl_result {
            Ok(refl) => {
                if let Err(err) = refl.save(refl_path) {
                    eprintln!("Failed to save reflectance {}: {err}", refl_path.display());
                    std::process::exit(1);
                }
                println!("Saved raw reflectance to {}", refl_path.display());
            }
            Err(err) => {
                eprintln!("Failed to extract reflectance: {err}");
                std::process::exit(1);
            }
        }
    }

    let result = if cli.color_restore {
        match cli.mode {
            Mode::Single => single_scale_retinex_color_restored(&image, sigmas[0]),
            Mode::Multi => multi_scale_retinex_color_restored(&image, &sigmas),
            #[cfg(feature = "gradient")]
            Mode::Gradient => unreachable!(),
        }
    } else {
        match cli.mode {
            Mode::Single => single_scale_retinex(&image, sigmas[0]),
            Mode::Multi => multi_scale_retinex(&image, &sigmas),
            #[cfg(feature = "gradient")]
            Mode::Gradient => unreachable!(),
        }
    };

    match result {
        Ok(processed) => {
            if let Err(err) = processed.save(&cli.output) {
                eprintln!("Failed to save {}: {err}", cli.output.display());
                std::process::exit(1);
            }
            println!("Saved result to {}", cli.output.display());
        }
        Err(err) => {
            eprintln!("Processing failed: {err}");
            std::process::exit(1);
        }
    }
}
