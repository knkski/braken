//! Regenerate the checked-in web corpus fixtures and GUI presets.

use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};
use braken_viz::{
    Primitive2d, Primitive3d, Turtle2dConfig, Turtle3dConfig, Visualization, VisualizationContext,
    VisualizeRequest, VisualizerBackend, VisualizerConfig, VisualizerKind, visualize,
};
use serde::Deserialize;
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    env, fs,
    path::Path,
};

#[derive(Deserialize)]
struct Catalog {
    systems: Vec<System>,
}

#[derive(Deserialize)]
struct System {
    id: String,
    name: String,
    file: String,
    profile: String,
    render_iterations: Option<u32>,
    #[serde(default)]
    render_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RenderReport {
    systems: Vec<RenderRecord>,
}

#[derive(Deserialize)]
struct RenderRecord {
    id: String,
    iterations: Option<u32>,
}

#[derive(Clone, Copy)]
struct PresetCopy {
    name: &'static str,
    summary: &'static str,
}

const GOTHIC_CARRIER_FILE_NAME: &str = "gothic-starwork.lsys";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("lib should be inside the workspace");
    let corpus = workspace.join("scratch/well-known-lsystems-corpus/lsystem-corpus");
    if !corpus.is_dir() {
        return Err(format!("corpus not found at {}", corpus.display()).into());
    }

    let catalog: Catalog = read_json(&corpus.join("catalog/catalog.json"))?;
    let report: RenderReport = read_json(&corpus.join("catalog/render-report.json"))?;
    let preset_dir = workspace.join("gui/presets/web");
    let iteration_limits = read_checked_in_iteration_limits(&preset_dir)?;
    let renders = report
        .systems
        .into_iter()
        .map(|record| (record.id.clone(), record))
        .collect::<HashMap<_, _>>();

    let fixture_dir = workspace.join("lib/tests/fixtures/web/pass");
    recreate_dir(&fixture_dir)?;
    recreate_dir(&preset_dir)?;

    let mut preset_names_seen = HashSet::new();
    let mut preset_grammars_seen = HashSet::new();
    seed_curated_presets(
        &workspace.join("gui/presets"),
        &mut preset_names_seen,
        &mut preset_grammars_seen,
    )?;

    let mut fixture_names = Vec::with_capacity(catalog.systems.len());
    let mut preset_names = Vec::new();
    for system in &catalog.systems {
        let source_path = corpus.join(&system.file);
        let file_name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("grammar filename is not UTF-8")?;
        let source = fs::read_to_string(&source_path)?;

        fs::write(fixture_dir.join(file_name), &source)?;
        fixture_names.push(file_name.to_owned());

        let Some(render) = renders.get(&system.id) else {
            continue;
        };
        let Some(visualizer) = supported_turtle_visualizer(&system.profile) else {
            continue;
        };
        if !admitted_gui_catalog_name(&system.name, visualizer)
            || has_incomplete_source_fidelity(&source)
            || metadata(&source, "Angle")
                .and_then(|angle| angle.parse::<f64>().ok())
                .is_none()
            || has_rewritten_turtle_constants(&source)
            || uses_external_turtle_resources(&source)
        {
            continue;
        }
        let Some(copy) = curated_preset_copy(&system.name) else {
            continue;
        };
        if !preset_names_seen.insert(normalized_name(copy.name))
            || !preset_grammars_seen.insert(normalized_grammar(&source))
        {
            continue;
        }

        let iterations = preset_iterations(system, render);
        let validation_source = format_preset(system, copy, &source, iterations, iterations, None);
        if !renders_geometry(&validation_source, iterations, visualizer) {
            continue;
        }
        let default_preview_iterations = iterations.min(2);
        let preview_iterations =
            preset_preview_iterations(&system.name).unwrap_or(default_preview_iterations);
        let preview_override =
            if renders_geometry(&validation_source, preview_iterations, visualizer) {
                (preview_iterations != default_preview_iterations).then_some(preview_iterations)
            } else {
                Some(iterations)
            };
        let preset_name = format!("web-{file_name}");
        let max_iterations = iteration_limits
            .get(&preset_name)
            .copied()
            .unwrap_or(iterations)
            .max(iterations);
        let preset = format_preset(
            system,
            copy,
            &source,
            iterations,
            max_iterations,
            preview_override,
        );
        fs::write(preset_dir.join(&preset_name), preset)?;
        preset_names.push(preset_name);
    }

    fixture_names.sort();
    preset_names.sort();
    fs::copy(
        corpus.join("NOTICE.md"),
        workspace.join("lib/tests/fixtures/web/NOTICE.md"),
    )?;
    write_fixture_manifest(workspace, &fixture_names)?;
    write_preset_manifest(workspace, &preset_names)?;

    println!(
        "imported {} test fixtures and {} GUI presets",
        fixture_names.len(),
        preset_names.len()
    );
    Ok(())
}

fn read_checked_in_iteration_limits(
    directory: &Path,
) -> Result<HashMap<String, u32>, Box<dyn std::error::Error>> {
    let mut limits = HashMap::new();
    if !directory.is_dir() {
        return Ok(limits);
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().is_none_or(|extension| extension != "lsys") {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let source = fs::read_to_string(&path)?;
        let Some(max_iterations) =
            metadata(&source, "Max Iterations").and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        limits.insert(file_name.to_owned(), max_iterations);
    }
    Ok(limits)
}

fn seed_curated_presets(
    directory: &Path,
    names: &mut HashSet<String>,
    grammars: &mut HashSet<String>,
) -> std::io::Result<()> {
    let mut paths = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "lsys")
        })
        .collect::<Vec<_>>();
    paths.sort();

    for path in paths {
        if path.file_name().and_then(|name| name.to_str()) == Some(GOTHIC_CARRIER_FILE_NAME) {
            continue;
        }
        let source = fs::read_to_string(path)?;
        if let Some(name) = metadata(&source, "Name") {
            names.insert(normalized_name(name));
        }
        grammars.insert(normalized_grammar(&source));
    }
    Ok(())
}

fn normalized_name(name: &str) -> String {
    let ascii = name
        .chars()
        .map(|character| match character.to_ascii_lowercase() {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ń' | 'ñ' => 'n',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            character if character.is_ascii_alphanumeric() => character,
            _ => ' ',
        })
        .collect::<String>();
    let mut words = ascii
        .split_whitespace()
        .filter(|word| !matches!(*word, "classic" | "curve" | "l" | "system" | "the" | "von"))
        .collect::<Vec<_>>();
    if words.contains(&"sierpinski") && words.contains(&"arrowhead") {
        words.retain(|word| *word != "sierpinski");
    }
    if words.contains(&"sierpinski") && words.contains(&"gasket") {
        words.retain(|word| *word != "gasket");
        words.push("triangle");
    }
    words.concat()
}

fn normalized_grammar(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn recreate_dir(path: &Path) -> std::io::Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    fs::create_dir_all(path)
}

fn format_preset(
    system: &System,
    copy: PresetCopy,
    source: &str,
    iterations: u32,
    max_iterations: u32,
    preview_iterations: Option<u32>,
) -> String {
    let visualizer = supported_turtle_visualizer(&system.profile)
        .expect("format_preset is called only for supported turtle profiles");
    let mut output = format!(
        "# Name: {}\n# Summary: {}\n# Visualizer: {}\n# Angle: {}\n# Iterations: {iterations}\n# Max Iterations: {max_iterations}\n{}\n",
        copy.name,
        copy.summary,
        visualizer.as_str(),
        metadata(source, "Angle").unwrap_or("90"),
        preset_initial_color(&system.name)
            .map(|color| format!("# Initial Color: {color}\n"))
            .unwrap_or_default(),
    );
    if let Some(preview_iterations) = preview_iterations {
        output.push_str(&format!("# Preview Iterations: {preview_iterations}\n\n"));
    }
    if !system.render_map.is_empty() {
        output.push_str("# Render Map: ");
        for (index, (module, action)) in system.render_map.iter().enumerate() {
            if index > 0 {
                output.push(' ');
            }
            output.push_str(module);
            output.push('=');
            output.push_str(action);
        }
        output.push_str("\n\n");
    }
    for line in source.lines() {
        if [
            "Name",
            "Parse",
            "Execute",
            "Render",
            "Angle",
            "Preview Iterations",
        ]
        .iter()
        .any(|key| is_metadata_line(line, key))
        {
            continue;
        }
        output.push_str(&curated_preset_source_line(&system.name, line));
        output.push('\n');
    }
    output
}

/// Give every recursive Circular Tile tier its own palette index while
/// preserving the imported geometry and provenance comments.
fn curated_preset_source_line<'a>(system_name: &str, line: &'a str) -> Cow<'a, str> {
    if system_name != "CircularTile" {
        return Cow::Borrowed(line);
    }

    if line.starts_with("axiom X Plus ") {
        return Cow::Owned(line.replace('X', "X(0)"));
    }
    if let Some(successor) = line.strip_prefix("match X then [ ") {
        return Cow::Owned(format!(
            "match X(c) then [ Color(c) {}",
            successor.replacen(" X Minus Y ", " X(c + 1) Minus Y(c + 1) ", 1)
        ));
    }
    if let Some(successor) = line.strip_prefix("match Y then [ ") {
        return Cow::Owned(format!(
            "match Y(c) then [ Color(c) {}",
            successor.replacen(" Minus Minus Minus Y ]", " Minus Minus Minus Y(c + 1) ]", 1,)
        ));
    }

    Cow::Borrowed(line)
}

fn renders_geometry(source: &str, iterations: u32, visualizer: VisualizerKind) -> bool {
    let Ok(grammar) = CompiledGrammar::parse(source) else {
        return false;
    };
    let Ok(calculation) = calculate(CalculationRequest {
        grammar,
        iterations: iterations as usize,
        backend: BackendChoice::Cpu,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits::unbounded_production(),
    }) else {
        return false;
    };
    let angle = metadata(source, "Angle")
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(90.0)
        .to_radians();
    let turtle = Turtle2dConfig {
        turn_angle: angle,
        ..Turtle2dConfig::default()
    }
    .with_source_metadata(source);
    let config = match visualizer {
        VisualizerKind::Turtle2d => VisualizerConfig::Turtle2d(turtle),
        VisualizerKind::Turtle3d => VisualizerConfig::Turtle3d(Turtle3dConfig::from(turtle)),
        _ => return false,
    };
    let Ok(visualization) = visualize(VisualizeRequest {
        generation: &calculation.generation,
        backend: VisualizerBackend::Cpu,
        config,
        context: VisualizationContext::default(),
    }) else {
        return false;
    };
    match visualization {
        Visualization::Scene2d(scene) => scene
            .primitives
            .iter()
            .any(|primitive| matches!(primitive, Primitive2d::Line(_) | Primitive2d::Polygon(_))),
        Visualization::Scene3d(scene) => scene
            .primitives
            .iter()
            .any(|primitive| matches!(primitive, Primitive3d::Line(_) | Primitive3d::Polygon(_))),
    }
}

fn supported_turtle_visualizer(profile: &str) -> Option<VisualizerKind> {
    match profile {
        "context-turtle2d"
        | "cut-turtle2d"
        | "parametric-agent-turtle2d"
        | "parametric-context-turtle2d"
        | "parametric-turtle2d"
        | "turtle2d-fractint"
        | "turtle2d-generic" => Some(VisualizerKind::Turtle2d),
        "turtle3d" | "turtle3d-l3d" | "turtle3d-propfeds" => Some(VisualizerKind::Turtle3d),
        _ => None,
    }
}

/// Three-dimensional presets are admitted explicitly because the source
/// catalogs contain other 3D profiles whose meshes, materials, external
/// models, or source-specific command mappings are not represented by a
/// self-contained `Scene3d` preset.
fn admitted_gui_catalog_name(name: &str, visualizer: VisualizerKind) -> bool {
    if visualizer == VisualizerKind::Turtle2d {
        return !excluded_from_gui_catalog(name);
    }
    matches!(name, "Blackboard tree" | "Fern (3-D)" | "Tree4")
}

fn curated_preset_copy(name: &str) -> Option<PresetCopy> {
    let (name, summary) = match name {
        "a binary tree with scale and thickness" => (
            "Tapered Binary Tree",
            "A symmetric binary crown grows through level-by-level changes in branch length and width.",
        ),
        "An ordinary fractal tree of various thicknesses of branches" => (
            "Tiered Fractal Tree",
            "Repeated side forks build a compact symmetric crown with level-varying stroke widths.",
        ),
        "Anklets of Krishna" => (
            "Anklets of Krishna",
            "Four interlocking square coils form a balanced ornamental knot.",
        ),
        "Anti Square Koch" => (
            "Anti-Square Koch Curve",
            "An inward-stepping Koch boundary builds a deeply notched shape with fourfold symmetry.",
        ),
        "Blackboard tree" => (
            "Spatial Four-Way Tree",
            "A central trunk repeatedly divides into four yawing and pitching limbs.",
        ),
        "Board" => (
            "Sierpiński Square Curve",
            "A continuous right-angled path subdivides a square into nested rooms and gridded corners.",
        ),
        "Bush" => (
            "Forked Bush",
            "Repeated clustered forks grow a compact shrub around a strong central stem.",
        ),
        "Carpet" => (
            "Fractal Carpet",
            "Bracketed right-angle branches weave a dense carpet of mirrored corridors and chambers.",
        ),
        "CircularTile" => (
            "Circular Tile",
            "Nested tiers of square curls interlock into a color-banded circular tile.",
        ),
        "Citric.Circles" => (
            "Citric Circles",
            "Five ringed lobes cluster around a smaller central rosette.",
        ),
        "ColorTriangGasket" => (
            "Colored Triangle Gasket",
            "Color-shifting triangular loops assemble a bright Sierpiński-style gasket.",
        ),
        "Cross dragon curve" => (
            "Cross Dragon Curve",
            "A folded dragon path builds a hooked cross-shaped orthogonal motif.",
        ),
        "Crystal" => (
            "Koch Crystal",
            "A fourfold quadratic Koch curve forms a crystalline square frame.",
        ),
        "Fass1" => (
            "FASS Curve",
            "A right-angled self-similar path packs repeating Greek-key motifs into a square.",
        ),
        "Fern (3-D)" => (
            "Spatial Fern",
            "A spatial fern unfurls paired leaflets along a gently curved stem.",
        ),
        "five lines as star" => (
            "Fiveflake",
            "Five scaled copies bloom from every segment into a branching pentagonal flake.",
        ),
        "FracRhombusTile" => (
            "Fractal Rhombus Tile",
            "Mirrored diagonal branches repeat into a tile of nested rhombus fragments.",
        ),
        "Gothic.Outline" => (
            "Gothic Starwork",
            "Nested stars, decagons, and onion-dome forms create intricate Gothic tracery.",
        ),
        "Hex1" => (
            "Hexagonal Honeycomb",
            "Seven linked hexagonal cells grow into a compact honeycomb rosette.",
        ),
        "Koch anti snowflake" => (
            "Koch Anti-Snowflake",
            "Three inward-growing Koch curves create a deeply notched triangular snowflake.",
        ),
        "Organic tree (scale, width)" => (
            "Organic Tapered Tree",
            "An asymmetric crown grows from branches that change length and width at every fork.",
        ),
        "Pentagram" => (
            "Recursive Pentagram",
            "Golden-ratio branches repeat a five-pointed star inside its surrounding pentagon.",
        ),
        "Pentaplex" => (
            "Pentaplex Curve",
            "Five interlocking pentagonal loops form a dense fivefold rosette.",
        ),
        "Pythagorean tree" => (
            "Pythagorean Tree",
            "Scaled square branches grow from a right-triangle scaffold into a geometric canopy.",
        ),
        "QuadKoch" => (
            "Quad Koch Curve",
            "A long orthogonal Koch path folds into a dense fourfold labyrinth.",
        ),
        "Quadratic Gosper" => (
            "Quadratic Gosper Curve",
            "A right-angled space-filling path packs interlocking meanders into a square.",
        ),
        "Quadratic Koch Island" => (
            "Quadratic Koch Island",
            "A closed right-angled Koch boundary grows four matching crenellated lobes.",
        ),
        "Quadratic Koch snowflake" => (
            "Quadratic Koch Snowflake",
            "Four orthogonal Koch sides form a deeply notched square snowflake.",
        ),
        "RhombusTile" => (
            "Rhombus Tile",
            "Mirrored 30-degree branches assemble an elongated tile of nested rhombi.",
        ),
        "scaled fractal pentagram as a Eulerian graph" => (
            "Scaled Fractal Pentagram",
            "Scaled color-stepped segments turn a pentagram into a nested Eulerian star.",
        ),
        "Sierpiński median curve" => (
            "Sierpiński Median Curve",
            "A diagonal Sierpiński path folds into a four-armed diamond-centered cross.",
        ),
        "Sierpiński triangle (scaled)" => (
            "Sierpiński Triangle",
            "Scaled triangular paths nest into the classic self-similar gasket.",
        ),
        "Sign." => (
            "Scalloped Hexagon",
            "Alternating companion curves ripple around a six-lobed snowflake-like ring.",
        ),
        "Sixfold snowflake" => (
            "Stochastic Sixfold Snowflake",
            "Random branch lengths give six fernlike arms a distinct crystalline texture.",
        ),
        "Snow flake" => (
            "Branched Snowflake",
            "Six recursively forked arms radiate into a delicate dendritic snowflake.",
        ),
        "Sphinx" => (
            "Sphinx Rep-Tile",
            "Smaller copies recursively assemble the asymmetric six-triangle sphinx figure.",
        ),
        "Spiral tiling" => (
            "Spiral Tiling",
            "Repeated square curls sweep around a center to form a circular woven tile.",
        ),
        "Terdragon boundary" => (
            "Terdragon Boundary",
            "Two mutually recursive paths trace the crenellated perimeter of a terdragon.",
        ),
        "TrapezoidTile" => (
            "Trapezoid Tile",
            "A 60-degree substitution stretches a simple trapezoid into a finely segmented frame.",
        ),
        "Tree4" => (
            "Spatial Tiered Tree",
            "A rising trunk advances through repeated tiers of opposed spatial branches.",
        ),
        "Vertigo1" => (
            "Vertigo Spiral",
            "A color-cycling path turns and shrinks into a tightly layered spiral.",
        ),
        _ => return None,
    };
    Some(PresetCopy { name, summary })
}

fn preset_iterations(system: &System, render: &RenderRecord) -> u32 {
    match system.name.as_str() {
        "CircularTile" => 6,
        "Fern (3-D)" => 7,
        "Sierpiński median curve" | "TrapezoidTile" => 8,
        "Vertigo1" => 14,
        _ => render
            .iterations
            .or(system.render_iterations)
            .unwrap_or(1)
            .max(1),
    }
}

fn preset_preview_iterations(name: &str) -> Option<u32> {
    match name {
        "CircularTile" => Some(3),
        _ => None,
    }
}

/// Assign visually distinct theme-palette colors to imported presets which do
/// not already select colors in their grammar.
fn preset_initial_color(name: &str) -> Option<u16> {
    match name {
        "Crystal" | "Quadratic Koch Island" | "Snow flake" => Some(0),
        "Fass1" | "Sierpiński median curve" | "Sierpiński triangle (scaled)" | "Sphinx" => {
            Some(1)
        }
        "Gothic.Outline" | "Pentagram" | "Pentaplex" => Some(2),
        "Cross dragon curve" | "Terdragon boundary" => Some(3),
        "Anti Square Koch" | "Koch anti snowflake" | "QuadKoch" | "Quadratic Koch snowflake" => {
            Some(4)
        }
        "Anklets of Krishna" | "five lines as star" | "Sixfold snowflake" => Some(5),
        "a binary tree with scale and thickness"
        | "Bush"
        | "Fern (3-D)"
        | "Organic tree (scale, width)"
        | "Pythagorean tree"
        | "An ordinary fractal tree of various thicknesses of branches"
        | "Blackboard tree"
        | "Tree4" => Some(6),
        "Citric.Circles" | "Spiral tiling" => Some(7),
        "Quadratic Gosper" => Some(8),
        "Board" | "Carpet" | "FracRhombusTile" | "RhombusTile" | "TrapezoidTile" => Some(9),
        "Hex1" => Some(10),
        "Sign." => Some(11),
        _ => None,
    }
}

/// Keep the public preset picker curated. These bounds mirror the
/// case-insensitive source-catalog ordering and prevent excluded equivalents
/// from reappearing after grammar deduplication.
fn excluded_from_gui_catalog(name: &str) -> bool {
    const EXACT: &[&str] = &[
        "2isosceles",
        "2tile1",
        "2tile1med",
        "32 segment curve",
        "32-segment curve",
        "5-rep-tile",
        "7-rep-tile",
        "26: bush/tree",
        "a colored tree (young parts are green, old are brown)",
        "a curved bush, rotate to see it nicely",
        "a leaf",
        "a nice decent tree!",
        "a three-dimensionsional bush-like structure",
        "a tree",
        "a tree bent to the left",
        "a triangular grid",
        "a triangular pattern",
        "a water weed",
        "a weed",
        "a weird tree that looks good from some angles",
        "adh105d3b",
        "adh105d5",
        "adh105d5a",
        "adh105d5c",
        "adh122zzi",
        "an ordinary fractal tree with scale",
        "an ordinary tree",
        "arrow weed",
        "barbells",
        "bird's nest",
        "birds-nest2",
        "boat",
        "border1",
        "box fractal",
        "boundary",
        "borchert–honda resource-allocation branching model",
        "bourke kolem",
        "branch shedding and dynamic-equilibrium palm skeleton",
        "branched acropetal signal",
        "calendula",
        "cantor set",
        "cantordust",
        "cellular botanica (propfeds parametric manual)",
        "cesaro",
        "cesaro fractal",
        "classic sierpinski curve",
        "cross",
        "fass2",
        "generated random l-system 06",
        "greek cross fractal",
        "grid",
        "growingleaf2bush-120",
        "hex",
        "hilbert curve (3-d)",
        "hilbert curve ii",
        "hilbert curve in three dimensions",
        "hollow lily pad",
        "koch",
        "netlogo tree2",
        "pentigree",
        "pine tree",
        "quadgosper",
        "quadratic koch curve",
        "quadratic snowflake",
        "quartet",
        "r.i.p.around",
        "seaweed",
        "serp2",
        "sierpiński tree",
        "signal propagation (propfeds parametric manual)",
        "smoke",
        "snake kolam",
        "snezh",
        "terdragon (davis and knuth)",
        "tree simple and efficient",
        "tree2",
        "variation to koch curve",
        "vertigo2",
        "weed",
        "willow",
    ];
    const RANGES: &[(&str, &str)] = &[
        ("adh105d5d", "adh119za"),
        ("adh119zc", "adh119zd"),
        ("adh119zf", "adh119zzi"),
        ("adh119zzk", "adh119zzm"),
        ("adh120a", "adh121zzl"),
        ("adh122a", "adh122za"),
        ("adh122zc", "adh122zzf"),
        ("adh122zzk", "adh123zzk"),
        ("adh124a1", "adh131a"),
        ("adh135", "an ordinary binary tree"),
        (
            "binary tree (young parts are green, old are brown)",
            "blackboard tree",
        ),
        ("botched cultivar ff", "botched cultivar xexf"),
        ("crystals.ls", "dragonmed1"),
        (
            "fass (space-filling, self-avoiding, simple, self-similar)",
            "fass curve 3",
        ),
        ("fern (3-d)", "fir tree"),
        (
            "five pointed star variant of koch's snowflake",
            "four branches tree",
        ),
        ("fractal pentagram (scaled) drawn in one stroke", "gilbert"),
        ("grid1", "heighway dragon"),
        ("hexa-grid", "hilbert curve (3-d)"),
        ("hilbert curve in three dimensions", "hiwaymed-x"),
        ("islands and lakes", "kitesanddarts"),
        ("koch curve", "organic tree"),
        (
            "oriented scaled fractal pentagram as a eulerian graph",
            "pentagon as a planar eulerian graph",
        ),
        ("pentagram recursively", "pentant"),
        ("pentive", "plant11"),
        ("rhombustile1", "root"),
        ("schneider", "seaweed06"),
        ("snowflake1", "spacefillingtree"),
        ("sporrer", "terdragnm"),
        ("terdragonalt", "trainsmoke"),
        ("trapezoidtile1", "untitled"),
    ];

    let name = name.to_lowercase();
    name.starts_with("adh")
        || EXACT.contains(&name.as_str())
        || RANGES
            .iter()
            .any(|&(start, end)| start <= name.as_str() && name.as_str() <= end)
        || (name.starts_with("bush") && name != "bush")
        || (("sextet"..="sierpiński triangle (scaled)").contains(&name.as_str())
            && !matches!(
                name.as_str(),
                "sierpiński median curve" | "sierpiński triangle (scaled)"
            ))
        || name.as_str() >= "weed 1"
}

fn has_incomplete_source_fidelity(source: &str) -> bool {
    metadata(source, "Fidelity") == Some("projection")
        || metadata(source, "Source-token warnings").is_some()
        || metadata(source, "Features").is_some_and(|features| {
            features
                .split(',')
                .map(str::trim)
                .any(|feature| matches!(feature, "known-broken-upstream" | "quarantine"))
        })
}

/// Older corpus builds accidentally converted numeric turtle-command
/// arguments into countdown productions. Such files cannot recover the
/// original command value, so do not present them as faithful presets. A
/// corrected corpus has no productions whose predecessor is a turtle command.
fn has_rewritten_turtle_constants(source: &str) -> bool {
    const COMMANDS: &[&str] = &[
        "Color",
        "ColorIncrement",
        "ExplicitMinus",
        "ExplicitPlus",
        "Scale",
    ];
    source.lines().any(|line| {
        let Some(predecessor) = line.trim_start().strip_prefix("match ") else {
            return false;
        };
        COMMANDS.iter().any(|command| {
            predecessor
                .strip_prefix(command)
                .is_some_and(|rest| rest.starts_with('(') || rest.starts_with(' '))
        })
    })
}

/// Object, Surface, and Query need source assets or an environment protocol
/// that the normalized `.lsys` file does not contain. The 3D interpreter shows
/// object/surface placement markers, but that approximation is not sufficient
/// for automatic preset admission.
fn uses_external_turtle_resources(source: &str) -> bool {
    source
        .lines()
        .filter(|line| line.starts_with("axiom ") || line.starts_with("match "))
        .flat_map(|line| {
            line.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        })
        .any(|word| matches!(word, "Object" | "Surface" | "Query"))
}

fn metadata<'a>(source: &'a str, key: &str) -> Option<&'a str> {
    source.lines().find_map(|line| {
        line.strip_prefix(&format!("# {key}"))?
            .trim_start()
            .strip_prefix(':')
            .map(str::trim)
    })
}

fn is_metadata_line(line: &str, key: &str) -> bool {
    line.strip_prefix("# ")
        .and_then(|line| line.split_once(':'))
        .is_some_and(|(found, _)| found.trim() == key)
}

fn write_fixture_manifest(workspace: &Path, names: &[String]) -> std::io::Result<()> {
    let mut manifest = String::from("# Expected native parser/CPU outcome: pass\n");
    for name in names {
        manifest.push_str(name);
        manifest.push('\n');
    }
    fs::write(
        workspace.join("lib/tests/fixtures/web/manifest.txt"),
        manifest,
    )
}

fn write_preset_manifest(workspace: &Path, names: &[String]) -> std::io::Result<()> {
    let mut rust = String::from(
        "// Generated by `cargo run -p braken --example import_web_corpus`.\n\
         pub const WEB_PRESET_SOURCES: &[&str] = &[\n",
    );
    for name in names {
        let include = format!("    include_str!(\"../presets/web/{name}\"),\n");
        if include.trim_end().chars().count() <= 100 {
            rust.push_str(&include);
        } else {
            rust.push_str(&format!(
                "    include_str!(\n        \"../presets/web/{name}\"\n    ),\n"
            ));
        }
    }
    rust.push_str("];\n");
    fs::write(workspace.join("gui/src/generated_web_presets.rs"), rust)
}
