use clap::{Parser, ValueEnum};
use retinex::{
    extract_illumination, multi_scale_retinex, multi_scale_retinex_color_restored,
    multi_scale_retinex_full, normalize_reflectance, single_scale_retinex,
    single_scale_retinex_color_restored, single_scale_retinex_full,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "retinex")]
#[command(about = "Apply Retinex enhancement to images", long_about = None)]
struct Cli {
    #[arg(help = "Input image file path")]
    input: PathBuf,

    #[arg(help = "Output image file path (reflectance result)")]
    output: PathBuf,

    #[arg(short, long, value_enum, default_value = "single")]
    mode: Mode,

    #[arg(long, value_delimiter = ',', default_value = "15.0")]
    sigmas: Vec<f32>,

    /// Save the estimated illumination to a separate file
    #[arg(long)]
    illumination: Option<PathBuf>,

    /// Save the raw reflectance (before color restoration) to a separate file
    #[arg(long)]
    reflectance: Option<PathBuf>,

    /// Enable color restoration (MSRCR) to prevent grayscale output
    #[arg(long, default_value_t = false)]
    color_restore: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Single,
    Multi,
}

fn main() {
    let cli = Cli::parse();

    if cli.sigmas.is_empty() {
        eprintln!("Sigmas cannot be empty");
        std::process::exit(1);
    }

    if let Some(&invalid) = cli.sigmas.iter().find(|&&sigma| sigma <= 0.0) {
        eprintln!("Sigmas must be positive, got {invalid}");
        std::process::exit(1);
    }

    let image = match image::open(&cli.input) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("Failed to open {}: {err}", cli.input.display());
            std::process::exit(1);
        }
    };

    // Save illumination if requested
    if let Some(illum_path) = &cli.illumination {
        let illum_result = match cli.mode {
            Mode::Single => extract_illumination(&image, cli.sigmas[0]),
            Mode::Multi => {
                // For multi-scale, use the first sigma for illumination visualization
                // or we could average them - using first for simplicity
                extract_illumination(&image, cli.sigmas[0])
            }
        };

        match illum_result {
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

    // Save raw reflectance if requested
    if let Some(refl_path) = &cli.reflectance {
        let refl_result = match cli.mode {
            Mode::Single => {
                let output = single_scale_retinex_full(&image, cli.sigmas[0]);
                output.map(|o| normalize_reflectance(&o.reflectance))
            }
            Mode::Multi => {
                let output = multi_scale_retinex_full(&image, &cli.sigmas);
                output.map(|o| normalize_reflectance(&o.reflectance))
            }
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

    // Process the main output
    let result = if cli.color_restore {
        match cli.mode {
            Mode::Single => single_scale_retinex_color_restored(&image, cli.sigmas[0], 0.0, 0.0),
            Mode::Multi => multi_scale_retinex_color_restored(&image, &cli.sigmas, 0.0, 0.0),
        }
    } else {
        match cli.mode {
            Mode::Single => single_scale_retinex(&image, cli.sigmas[0]),
            Mode::Multi => multi_scale_retinex(&image, &cli.sigmas),
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
