use std::fs;
use std::io::{self, IsTerminal, Read, Write};

use anyhow::Result;
use braken::{
    AmbiguousRulePolicy, BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar,
    DerivationSemantics, FloatWidth, calculate_with_progress, generate_seed,
};
use braken_viz::{
    Turtle2dConfig, Turtle3dConfig, Visualization, VisualizationContext, VisualizeRequest,
    VisualizerBackend, VisualizerConfig, VisualizerKind,
    targets::{Palette, svg, tui},
    visualize_with_backend, visualizer_metadata,
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(version, author, about = "Derive and render L-systems with Braken")]
struct Args {
    /// Read an .lsys grammar from a file. Reads stdin when omitted.
    #[arg(short, long)]
    file: Option<String>,

    /// The number of iterations to calculate.
    #[arg(short, long, default_value_t = 1)]
    iterations: usize,

    /// Grammar derivation backend.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,

    /// Visualizer to use. Source metadata is used when omitted.
    #[arg(long)]
    visualizer: Option<VisualizerKind>,

    /// Visualization backend, selected independently from grammar derivation.
    #[arg(long, value_enum, default_value_t = VisualizerBackendArg::Auto)]
    visualizer_backend: VisualizerBackendArg,

    /// Output target.
    #[arg(long, value_enum, default_value_t = Target::Auto)]
    target: Target,

    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<String>,

    /// Turtle turn angle in degrees. Source metadata or 90 degrees is used when omitted.
    #[arg(long)]
    angle: Option<f64>,

    /// Random seed. A random seed is selected when omitted.
    #[arg(long)]
    seed: Option<u64>,

    /// Floating-point width for derivation semantics.
    #[arg(long, value_enum, default_value_t = FloatWidthArg::F64)]
    float_width: FloatWidthArg,

    /// Policy for multiple applicable unweighted productions.
    #[arg(long, value_enum, default_value_t = AmbiguousRulesArg::Uniform)]
    ambiguous_rules: AmbiguousRulesArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    Auto,
    Cpu,
    Cuda,
    Wgpu,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FloatWidthArg {
    F32,
    F64,
}

impl From<FloatWidthArg> for FloatWidth {
    fn from(value: FloatWidthArg) -> Self {
        match value {
            FloatWidthArg::F32 => Self::F32,
            FloatWidthArg::F64 => Self::F64,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AmbiguousRulesArg {
    First,
    Error,
    Uniform,
}

impl From<AmbiguousRulesArg> for AmbiguousRulePolicy {
    fn from(value: AmbiguousRulesArg) -> Self {
        match value {
            AmbiguousRulesArg::First => Self::First,
            AmbiguousRulesArg::Error => Self::Error,
            AmbiguousRulesArg::Uniform => Self::Uniform,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VisualizerBackendArg {
    Auto,
    Cpu,
    Cuda,
}

impl From<VisualizerBackendArg> for VisualizerBackend {
    fn from(value: VisualizerBackendArg) -> Self {
        match value {
            VisualizerBackendArg::Auto => Self::Auto,
            VisualizerBackendArg::Cpu => Self::Cpu,
            VisualizerBackendArg::Cuda => Self::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Target {
    Auto,
    Tui,
    Svg,
}

fn run() -> Result<()> {
    let args = Args::parse();
    let seed = match args.seed {
        Some(seed) => seed,
        None => generate_seed()
            .map_err(|error| anyhow::anyhow!("could not generate a random seed: {error}"))?,
    };
    eprintln!("Seed: {seed}");

    let source = match args.file {
        Some(path) => fs::read_to_string(path)?,
        None => {
            let mut source = String::new();
            io::stdin().read_to_string(&mut source)?;
            source
        }
    };

    let interactive_progress = io::stderr().is_terminal();
    let result = calculate_with_progress(
        CalculationRequest {
            grammar: CompiledGrammar::parse(&source)?,
            iterations: args.iterations,
            backend: match args.backend {
                Backend::Auto => BackendChoice::Auto,
                Backend::Cpu => BackendChoice::Cpu,
                Backend::Cuda => BackendChoice::Cuda,
                Backend::Wgpu => BackendChoice::Wgpu,
            },
            seed,
            semantics: DerivationSemantics {
                float_width: args.float_width.into(),
                ambiguous_rules: args.ambiguous_rules.into(),
            },
            limits: CalculationLimits::unbounded_production(),
        },
        |progress| {
            let message = format!(
                "Deriving {}/{} · {} modules · {} items · {:.1?}",
                progress.completed_iterations,
                progress.total_iterations,
                progress.modules,
                progress.items,
                progress.elapsed,
            );
            if interactive_progress {
                eprint!("\r\x1b[2K{message}");
                let _ = io::stderr().flush();
            } else if progress.completed_iterations == 0
                || progress.completed_iterations == progress.total_iterations
            {
                eprintln!("{message}");
            }
            true
        },
    )?;
    if interactive_progress {
        eprintln!();
    }
    eprintln!(
        "Derivation: {}",
        derivation_backend_name(&result.backend_used)
    );

    let visualizer = resolve_visualizer(args.visualizer, &source)?;
    let angle = args
        .angle
        .or_else(|| numeric_metadata(&source, "Angle"))
        .unwrap_or(90.0);
    let turtle = Turtle2dConfig {
        turn_angle: angle.to_radians(),
        ..Turtle2dConfig::default()
    }
    .with_source_metadata(&source);
    let config = match visualizer {
        VisualizerKind::Turtle2d => VisualizerConfig::Turtle2d(turtle),
        VisualizerKind::Turtle3d => VisualizerConfig::Turtle3d(Turtle3dConfig::from(turtle)),
        _ => VisualizerConfig::for_kind(visualizer),
    };
    let context = VisualizationContext {
        iterations: result.stats.iterations,
        seed,
        derivation_backend: Some(derivation_backend_name(&result.backend_used)),
        elapsed: Some(result.stats.elapsed),
    };
    let output = visualize_with_backend(VisualizeRequest {
        generation: &result.generation,
        backend: args.visualizer_backend.into(),
        config,
        context,
    })?;
    eprintln!(
        "Visualization: {}",
        visualizer_backend_name(output.backend_used)
    );
    let target = match args.target {
        Target::Auto if visualizer == VisualizerKind::Inspector => Target::Tui,
        Target::Auto => Target::Svg,
        target => target,
    };
    let output = match (target, output.visualization) {
        (Target::Tui, Visualization::Scene2d(scene)) => tui::encode(&scene, (80, 24)),
        (Target::Tui, Visualization::Scene3d(scene)) => tui::encode_3d(&scene, (80, 24)),
        (Target::Svg, Visualization::Scene2d(scene)) => svg::encode(&scene, Palette::Light),
        (Target::Svg, Visualization::Scene3d(scene)) => svg::encode_3d(&scene, Palette::Light),
        (Target::Auto, _) => unreachable!(),
    };
    if let Some(path) = args.output {
        fs::write(path, output)?;
    } else {
        print!("{output}");
    }

    Ok(())
}

fn visualizer_backend_name(backend: VisualizerBackend) -> &'static str {
    match backend {
        VisualizerBackend::Auto => "auto",
        VisualizerBackend::Cpu => "cpu",
        VisualizerBackend::Cuda => "cuda",
    }
}

fn derivation_backend_name(backend: &BackendChoice) -> &'static str {
    match backend {
        BackendChoice::Auto => "auto",
        BackendChoice::Cpu => "cpu",
        BackendChoice::Cuda => "cuda",
        BackendChoice::Wgpu => "wgpu",
        BackendChoice::Named(_) => "named",
    }
}

fn numeric_metadata(source: &str, key: &str) -> Option<f64> {
    let prefix = format!("# {key}:");
    source.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix(&prefix)
            .and_then(|value| value.trim().parse().ok())
    })
}

fn resolve_visualizer(
    command_line: Option<VisualizerKind>,
    source: &str,
) -> Result<VisualizerKind, braken_viz::VisualizerMetadataError> {
    if let Some(visualizer) = command_line {
        return Ok(visualizer);
    }
    Ok(visualizer_metadata(source)?.unwrap_or_default())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_an_explicit_seed() {
        let args = Args::try_parse_from(["braken", "--seed", "42"]).unwrap();
        assert_eq!(args.seed, Some(42));
    }

    #[test]
    fn rejects_a_seed_outside_u64() {
        assert!(Args::try_parse_from(["braken", "--seed", "18446744073709551616"]).is_err());
    }

    #[test]
    fn accepts_an_explicit_wgpu_backend() {
        let args = Args::try_parse_from(["braken", "--backend", "wgpu"]).unwrap();
        assert!(matches!(args.backend, Backend::Wgpu));
    }

    #[test]
    fn accepts_an_explicit_derivation_semantic_profile() {
        let args = Args::try_parse_from([
            "braken",
            "--float-width",
            "f32",
            "--ambiguous-rules",
            "error",
        ])
        .unwrap();
        assert!(matches!(args.float_width, FloatWidthArg::F32));
        assert!(matches!(args.ambiguous_rules, AmbiguousRulesArg::Error));
    }

    #[test]
    fn accepts_an_independent_visualizer_backend() {
        let args = Args::try_parse_from([
            "braken",
            "--backend",
            "wgpu",
            "--visualizer-backend",
            "cuda",
        ])
        .unwrap();
        assert!(matches!(args.backend, Backend::Wgpu));
        assert!(matches!(
            args.visualizer_backend,
            VisualizerBackendArg::Cuda
        ));
    }

    #[test]
    fn automatic_backend_does_not_replace_an_unimplemented_visualizer_kind() {
        let generation = braken::Generation::default();
        let error = visualize_with_backend(VisualizeRequest {
            generation: &generation,
            backend: VisualizerBackend::Auto,
            config: VisualizerConfig::for_kind(VisualizerKind::Plot),
            context: VisualizationContext::default(),
        })
        .unwrap_err();
        assert!(matches!(
            error,
            braken_viz::VisualizeError::Unimplemented {
                visualizer: VisualizerKind::Plot,
                backend: VisualizerBackend::Cpu,
            }
        ));
    }

    #[test]
    fn visualizer_precedence_is_cli_then_source_then_inspector() {
        assert_eq!(
            resolve_visualizer(Some(VisualizerKind::Plot), "# Visualizer: turtle_2d").unwrap(),
            VisualizerKind::Plot
        );
        assert_eq!(
            resolve_visualizer(Some(VisualizerKind::Plot), "# Visualizer: not_a_visualizer")
                .unwrap(),
            VisualizerKind::Plot
        );
        assert_eq!(
            resolve_visualizer(None, "# Visualizer: turtle_2d").unwrap(),
            VisualizerKind::Turtle2d
        );
        assert_eq!(
            resolve_visualizer(None, "axiom F;").unwrap(),
            VisualizerKind::Inspector
        );
    }

    #[test]
    fn command_line_angle_overrides_source_metadata() {
        let source_angle = numeric_metadata("# Angle: 45", "Angle");
        assert_eq!(Some(30.0).or(source_angle).unwrap_or(90.0), 30.0);
        assert_eq!(None.or(source_angle).unwrap_or(90.0), 45.0);
        assert_eq!(
            None.or(numeric_metadata("axiom F;", "Angle"))
                .unwrap_or(90.0),
            90.0
        );
    }
}
