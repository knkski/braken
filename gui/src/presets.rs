//! Curated preset metadata and executable source for the graphical catalogs.
//!
//! Hand-authored presets own their display copy in their `.lsys` metadata.
//! Imported display copy is maintained by the web-corpus importer documented
//! in `CONTRIBUTING.md`; generated preset files and preview assets must not be
//! edited directly.

use crate::orientation::OrientationAnchor;
use braken_viz::{Turtle2dConfig, VisualizerKind, visualizer_metadata};

#[path = "generated_web_presets.rs"]
mod generated_web_presets;
use generated_web_presets::WEB_PRESET_SOURCES;

#[derive(Clone)]
pub struct Preset {
    pub name: String,
    pub summary: String,
    pub source: String,
    pub angle: f64,
    pub iters: u32,
    /// Iteration count used by the offline thumbnail generator.
    pub preview_iters: u32,
    pub max_iters: u32,
    pub visualizer: VisualizerKind,
    pub turtle_config: Turtle2dConfig,
    /// Optional geometric landmark used to keep recursive curves upright.
    pub orientation_anchor: Option<OrientationAnchor>,
    search_text: String,
}

const GOTHIC_IMPORTED_ID: &str = "LSYS-92A83BA22EA0";
const GOTHIC_STARWORK_SOURCE: &str = include_str!("../presets/gothic-starwork.lsys");
const BRAKEN_SOURCE: &str = include_str!("../branding/app-icon.lsys");

const HAND_AUTHORED_PRESET_SOURCES: &[&str] = &[
    include_str!("../presets/3d-hilbert-curve.lsys"),
    BRAKEN_SOURCE,
    include_str!("../presets/cordate-leaf.lsys"),
    include_str!("../presets/hilbert-curve.lsys"),
    include_str!("../presets/koch-snowflake.lsys"),
    include_str!("../presets/levy-c-curve.lsys"),
    include_str!("../presets/dragon-curve.lsys"),
    include_str!("../presets/terdragon.lsys"),
    include_str!("../presets/pentaflake.lsys"),
    include_str!("../presets/stochastic-plant.lsys"),
    include_str!("../presets/gosper-curve.lsys"),
    include_str!("../presets/virus.lsys"),
    include_str!("../presets/arrowhead.lsys"),
    include_str!("../presets/fancy-triangle.lsys"),
    include_str!("../presets/penrose-tiling.lsys"),
    include_str!("../presets/island.lsys"),
    include_str!("../presets/lily-of-the-valley-flower.lsys"),
    include_str!("../presets/rose-leaf.lsys"),
    include_str!("../presets/simple-leaf-family.lsys"),
];

pub fn get_presets() -> Vec<Preset> {
    let mut presets = HAND_AUTHORED_PRESET_SOURCES
        .iter()
        .copied()
        .map(|source| parse_preset(source, false))
        .chain(
            WEB_PRESET_SOURCES
                .iter()
                .copied()
                .map(|source| parse_preset(source, true)),
        )
        .collect::<Vec<_>>();

    presets.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.name.cmp(&right.name))
    });
    presets
}

impl Preset {
    /// Compact dimensional label for turtle preset cards.
    ///
    /// Other visualizer kinds deliberately omit the badge rather than being
    /// presented as either planar or spatial.
    pub fn dimension_badge_label(&self) -> Option<&'static str> {
        match self.visualizer {
            VisualizerKind::Turtle2d => Some("2D"),
            VisualizerKind::Turtle3d => Some("3D"),
            _ => None,
        }
    }

    pub fn matches_search_terms(&self, terms: &[String]) -> bool {
        terms.iter().all(|term| self.search_text.contains(term))
    }
}

fn parse_preset(source: &str, imported: bool) -> Preset {
    let name = required_metadata(source, "Name");
    let summary = required_metadata(source, "Summary");
    let is_gothic_starwork =
        imported && metadata(source, "ID").as_deref() == Some(GOTHIC_IMPORTED_ID);
    let executable_source = if is_gothic_starwork {
        GOTHIC_STARWORK_SOURCE
    } else {
        source
    };
    let angle = required_metadata(source, "Angle")
        .parse::<f64>()
        .unwrap_or_else(|error| panic!("{name} has invalid Angle metadata: {error}"));
    let iters = required_metadata(source, "Iterations")
        .parse::<u32>()
        .unwrap_or_else(|error| panic!("{name} has invalid Iterations metadata: {error}"));
    let preview_iters = metadata(source, "Preview Iterations")
        .map(|value| {
            value.parse::<u32>().unwrap_or_else(|error| {
                panic!("{name} has invalid Preview Iterations metadata: {error}")
            })
        })
        .unwrap_or_else(|| iters.min(2));
    let max_iters = required_metadata(source, "Max Iterations")
        .parse::<u32>()
        .unwrap_or_else(|error| panic!("{name} has invalid Max Iterations metadata: {error}"));
    let visualizer = visualizer_metadata(source)
        .unwrap_or_else(|error| panic!("{name} has invalid Visualizer metadata: {error}"))
        .unwrap_or_default();
    let turtle_config = Turtle2dConfig {
        turn_angle: angle.to_radians(),
        ..Turtle2dConfig::default()
    }
    .with_source_metadata(executable_source);
    let orientation_anchor = metadata(source, "Orientation Anchor").map(|value| {
        value.parse().unwrap_or_else(|error| {
            panic!("{name} has invalid Orientation Anchor metadata: {error}")
        })
    });
    let features = metadata(source, "Features").unwrap_or_default();
    let search_text = format!("{name} {summary} {features}").to_lowercase();
    Preset {
        name,
        summary,
        source: strip_comments(executable_source),
        angle,
        iters,
        preview_iters,
        max_iters,
        visualizer,
        turtle_config,
        orientation_anchor,
        search_text,
    }
}

fn required_metadata(source: &str, key: &str) -> String {
    metadata(source, key).unwrap_or_else(|| panic!("preset is missing {key} metadata"))
}

fn metadata(source: &str, key: &str) -> Option<String> {
    source.lines().find_map(|line| {
        let line = line.strip_prefix('#')?.trim_start();
        let (line_key, value) = line.split_once(':')?;
        (line_key.trim() == key).then(|| value.trim().to_owned())
    })
}

fn strip_comments(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use braken::{
        BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate,
    };
    use braken_viz::{
        Primitive2d, Primitive3d, StrokeColor, Turtle3dConfig, Visualization, VisualizationContext,
        VisualizeRequest, VisualizerBackend, VisualizerConfig, visualize,
    };

    #[test]
    fn every_preset_has_required_metadata() {
        for source in HAND_AUTHORED_PRESET_SOURCES
            .iter()
            .chain(WEB_PRESET_SOURCES)
        {
            for key in [
                "Name",
                "Summary",
                "Angle",
                "Iterations",
                "Max Iterations",
                "Visualizer",
            ] {
                assert!(
                    metadata(source, key).is_some(),
                    "preset source is missing {key} metadata"
                );
            }
        }
    }

    #[test]
    fn preview_iterations_use_an_optional_override() {
        let hilbert = get_presets()
            .into_iter()
            .find(|preset| preset.name == "Hilbert Curve")
            .expect("Hilbert Curve should be available");
        assert_eq!(hilbert.preview_iters, 2);

        let fallback = parse_preset(
            "# Name: Preview fallback\n\
             # Summary: Test preset.\n\
             # Visualizer: turtle_2d\n\
             # Angle: 90\n\
             # Iterations: 1\n\
             # Max Iterations: 1\n\n\
             axiom Draw;",
            false,
        );
        assert_eq!(fallback.preview_iters, 1);
    }

    #[test]
    fn circular_tile_uses_one_palette_color_per_recursive_tier() {
        let preset = get_presets()
            .into_iter()
            .find(|preset| preset.name == "Circular Tile")
            .expect("Circular Tile should be available");
        assert_eq!(preset.iters, 6);
        assert_eq!(preset.preview_iters, 3);

        let calculation = calculate(CalculationRequest {
            grammar: CompiledGrammar::parse(&preset.source).unwrap(),
            iterations: preset.iters as usize,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::unbounded_production(),
        })
        .unwrap();
        let Visualization::Scene2d(scene) = visualize(VisualizeRequest {
            generation: &calculation.generation,
            backend: VisualizerBackend::Cpu,
            config: VisualizerConfig::Turtle2d(preset.turtle_config),
            context: VisualizationContext::default(),
        })
        .unwrap() else {
            panic!("Circular Tile did not produce a planar scene");
        };
        let mut color_counts = [0; 6];
        for line in scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive2d::Line(line) => Some(line),
                Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
            })
        {
            match line.color {
                StrokeColor::PaletteIndex(index @ 0..=5) => {
                    color_counts[usize::from(index)] += 1;
                }
                color => panic!("unexpected Circular Tile preview color {color:?}"),
            }
        }

        assert_eq!(color_counts, [216, 432, 648, 864, 1_080, 1_296]);
    }

    #[test]
    #[should_panic(expected = "invalid Preview Iterations metadata")]
    fn invalid_preview_iterations_are_rejected() {
        parse_preset(
            "# Name: Invalid preview\n\
             # Summary: Test preset.\n\
             # Visualizer: turtle_2d\n\
             # Angle: 90\n\
             # Iterations: 1\n\
             # Preview Iterations: nope\n\
             # Max Iterations: 1\n\n\
             axiom Draw;",
            false,
        );
    }

    #[test]
    fn all_presets_parse_and_calculate() {
        for preset in get_presets() {
            let grammar = CompiledGrammar::parse(&preset.source)
                .unwrap_or_else(|error| panic!("{} failed to parse: {error}", preset.name));

            calculate(CalculationRequest {
                grammar,
                iterations: preset.iters.min(2) as usize,
                backend: BackendChoice::Cpu,
                seed: 0,
                semantics: Default::default(),
                limits: CalculationLimits::default(),
            })
            .unwrap_or_else(|error| panic!("{} failed to calculate: {error}", preset.name));
        }
    }

    #[test]
    fn preset_sources_do_not_use_legacy_conversion_syntax() {
        for preset in get_presets() {
            assert!(
                !preset.source.contains("=>"),
                "{} still contains legacy production syntax",
                preset.name
            );
            assert!(
                !preset.source.contains("Converted from"),
                "{} still contains generated conversion comments",
                preset.name
            );
        }
    }

    #[test]
    fn loaded_preset_sources_do_not_include_comments() {
        for preset in get_presets() {
            assert!(
                !preset
                    .source
                    .lines()
                    .any(|line| line.trim_start().starts_with('#')),
                "{} still includes comments in editor source",
                preset.name
            );
        }
    }

    #[test]
    fn generated_web_preset_set_is_complete() {
        assert_eq!(WEB_PRESET_SOURCES.len(), 41);
        assert!(
            WEB_PRESET_SOURCES
                .iter()
                .all(|source| metadata(source, "Name").as_deref() != Some("Hollow lily pad"))
        );
    }

    #[test]
    fn presets_are_one_alphabetically_sorted_catalog() {
        let presets = get_presets();
        assert_eq!(presets.len(), 60);
        let names = presets
            .iter()
            .map(|preset| preset.name.to_lowercase())
            .collect::<Vec<_>>();
        assert!(names.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn preset_catalog_copy_is_complete_and_regularized() {
        let presets = get_presets();
        let mut names = std::collections::HashSet::new();
        let generic_summaries = [
            "A recursively generated line drawing.",
            "A randomized, recursively generated line drawing.",
            "A recursively generated three-dimensional turtle drawing.",
            "A randomized, recursively generated three-dimensional turtle drawing.",
        ];

        for preset in presets {
            assert_eq!(preset.name, preset.name.trim(), "{}", preset.name);
            assert_eq!(preset.summary, preset.summary.trim(), "{}", preset.name);
            assert!(!preset.name.ends_with('.'), "{}", preset.name);
            assert!(preset.summary.ends_with('.'), "{}", preset.name);
            assert!(!preset.summary.contains('\n'), "{}", preset.name);
            assert!(
                !generic_summaries.contains(&preset.summary.as_str()),
                "{} retains an uncurated summary",
                preset.name
            );
            assert!(
                names.insert(preset.name.to_lowercase()),
                "duplicate preset name {}",
                preset.name
            );
        }
    }

    #[test]
    fn three_dimensional_hilbert_is_the_default_and_two_dimensional_hilbert_remains() {
        let presets = get_presets();
        assert_eq!(presets[0].name, "3D Hilbert Curve");
        assert_eq!(presets[0].visualizer, VisualizerKind::Turtle3d);
        let planar = presets
            .iter()
            .find(|preset| preset.name == "Hilbert Curve")
            .expect("the hand-authored 2D Hilbert should remain available");
        assert_eq!(planar.visualizer, VisualizerKind::Turtle2d);
    }

    #[test]
    fn every_preset_defines_a_color() {
        for preset in get_presets() {
            assert!(
                preset.turtle_config.initial_color != StrokeColor::ThemeDefault
                    || preset.source.contains("Color"),
                "{} does not define an initial color or color command",
                preset.name
            );
        }
    }

    #[test]
    fn turtle_presets_have_dimension_badge_labels() {
        let mut presets = get_presets();
        for preset in &presets {
            match preset.visualizer {
                VisualizerKind::Turtle2d => {
                    assert_eq!(preset.dimension_badge_label(), Some("2D"));
                }
                VisualizerKind::Turtle3d => {
                    assert_eq!(preset.dimension_badge_label(), Some("3D"));
                }
                visualizer => panic!("unexpected preset visualizer {visualizer}"),
            }
        }

        presets[0].visualizer = VisualizerKind::Inspector;
        assert_eq!(presets[0].dimension_badge_label(), None);
    }

    #[test]
    fn abop_surface_presets_emit_their_complete_frameworks_and_fills() {
        let expected = [
            ("Cordate Leaf", 272, 34),
            ("Lanceolate Leaf", 200, 1),
            ("Rose Leaflet", 676, 50),
        ];
        let presets = get_presets();
        for (name, expected_lines, expected_polygons) in expected {
            let preset = presets
                .iter()
                .find(|preset| preset.name == name)
                .unwrap_or_else(|| panic!("missing {name} preset"));
            assert_eq!(preset.turtle_config.draw_modules, ["G"]);
            let calculation = calculate(CalculationRequest {
                grammar: CompiledGrammar::parse(&preset.source).unwrap(),
                iterations: preset.iters as usize,
                backend: BackendChoice::Cpu,
                seed: 0,
                semantics: Default::default(),
                limits: CalculationLimits::unbounded_production(),
            })
            .unwrap();
            let Visualization::Scene2d(scene) = visualize(VisualizeRequest {
                generation: &calculation.generation,
                backend: VisualizerBackend::Cpu,
                config: VisualizerConfig::Turtle2d(preset.turtle_config.clone()),
                context: VisualizationContext::default(),
            })
            .unwrap() else {
                panic!("{name} did not produce a planar scene");
            };
            assert_eq!(
                scene
                    .primitives
                    .iter()
                    .filter(|primitive| matches!(primitive, Primitive2d::Line(_)))
                    .count(),
                expected_lines,
                "{name} framework line count changed",
            );
            assert_eq!(
                scene
                    .primitives
                    .iter()
                    .filter(|primitive| matches!(primitive, Primitive2d::Polygon(_)))
                    .count(),
                expected_polygons,
                "{name} polygon count changed",
            );
        }

        let lily = presets
            .iter()
            .find(|preset| preset.name == "Lily-of-the-Valley Flower")
            .expect("missing Lily-of-the-Valley Flower preset");
        let calculation = calculate(CalculationRequest {
            grammar: CompiledGrammar::parse(&lily.source).unwrap(),
            iterations: lily.iters as usize,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::unbounded_production(),
        })
        .unwrap();
        let Visualization::Scene3d(scene) = visualize(VisualizeRequest {
            generation: &calculation.generation,
            backend: VisualizerBackend::Cpu,
            config: VisualizerConfig::Turtle3d(Turtle3dConfig::from(lily.turtle_config.clone())),
            context: VisualizationContext::default(),
        })
        .unwrap() else {
            panic!("Lily-of-the-Valley Flower did not produce a spatial scene");
        };
        assert_eq!(
            scene
                .primitives
                .iter()
                .filter(|primitive| matches!(primitive, Primitive3d::Line(_)))
                .count(),
            100,
        );
        assert_eq!(
            scene
                .primitives
                .iter()
                .filter(|primitive| matches!(primitive, Primitive3d::Polygon(_)))
                .count(),
            50,
        );
    }

    #[test]
    fn three_dimensional_hilbert_fills_a_cube_without_long_connectors() {
        let preset = get_presets()
            .into_iter()
            .find(|preset| preset.name == "3D Hilbert Curve")
            .expect("the 3D Hilbert curve should be available");
        assert_eq!(preset.turtle_config.draw_modules, ["F"]);

        for iterations in 1_u32..=preset.max_iters {
            let calculation = calculate(CalculationRequest {
                grammar: CompiledGrammar::parse(&preset.source).unwrap(),
                iterations: iterations as usize,
                backend: BackendChoice::Cpu,
                seed: 0,
                semantics: Default::default(),
                limits: CalculationLimits::default(),
            })
            .unwrap();
            let Visualization::Scene3d(scene) = visualize(VisualizeRequest {
                generation: &calculation.generation,
                backend: VisualizerBackend::Cpu,
                config: VisualizerConfig::Turtle3d(Turtle3dConfig::from(
                    preset.turtle_config.clone(),
                )),
                context: VisualizationContext::default(),
            })
            .unwrap() else {
                panic!("the 3D Hilbert preset should produce a spatial scene");
            };

            let lines = scene
                .primitives
                .iter()
                .filter_map(|primitive| match primitive {
                    Primitive3d::Line(line) => Some(line.line),
                    Primitive3d::Polygon(_) => None,
                })
                .collect::<Vec<_>>();
            let side = (1_u32 << iterations) - 1;
            let point_count = 8_usize.pow(iterations);
            assert_eq!(lines.len(), point_count - 1);

            let mut minimum = [f64::INFINITY; 3];
            let mut maximum = [f64::NEG_INFINITY; 3];
            let mut lattice_points = std::collections::BTreeSet::new();
            let mut previous_endpoint = None;
            for line in lines {
                if let Some(previous) = previous_endpoint {
                    assert_eq!(
                        line.0, previous,
                        "iteration {iterations} is not one continuous path"
                    );
                }
                previous_endpoint = Some(line.1);
                let delta = [
                    line.1.0 - line.0.0,
                    line.1.1 - line.0.1,
                    line.1.2 - line.0.2,
                ];
                let length_squared = delta.iter().map(|value| value * value).sum::<f64>();
                assert!(
                    (length_squared - 1.0).abs() < 1.0e-9,
                    "iteration {iterations} contains a non-unit connector"
                );
                for point in [line.0, line.1] {
                    let coordinates = [point.0, point.1, point.2];
                    for axis in 0..3 {
                        assert!(
                            (coordinates[axis] - coordinates[axis].round()).abs() < 1.0e-9,
                            "iteration {iterations} leaves the cubic lattice"
                        );
                        minimum[axis] = minimum[axis].min(coordinates[axis]);
                        maximum[axis] = maximum[axis].max(coordinates[axis]);
                    }
                    lattice_points.insert((
                        point.0.round() as i64,
                        point.1.round() as i64,
                        point.2.round() as i64,
                    ));
                }
            }

            assert_eq!(lattice_points.len(), point_count);
            for axis in 0..3 {
                assert!(
                    (maximum[axis] - minimum[axis] - f64::from(side)).abs() < 1.0e-9,
                    "iteration {iterations} does not fill a cube along axis {axis}"
                );
            }
        }
    }

    #[test]
    fn imported_catalog_uses_only_complete_supported_sources() {
        let expected_3d = [
            "Spatial Fern",
            "Spatial Four-Way Tree",
            "Spatial Tiered Tree",
        ];
        let mut found_3d = Vec::new();
        for source in WEB_PRESET_SOURCES {
            match metadata(source, "Visualizer").as_deref() {
                Some("turtle_2d") => {}
                Some("turtle_3d") => found_3d.push(required_metadata(source, "Name")),
                visualizer => panic!("unexpected imported visualizer {visualizer:?}"),
            }
            assert_ne!(metadata(source, "Fidelity").as_deref(), Some("projection"));
            assert!(metadata(source, "Source-token warnings").is_none());
            let features = metadata(source, "Features").unwrap_or_default();
            assert!(
                !features
                    .split(',')
                    .map(str::trim)
                    .any(|feature| matches!(feature, "known-broken-upstream" | "quarantine"))
            );
        }
        found_3d.sort();
        assert_eq!(found_3d, expected_3d);
    }

    #[test]
    fn curated_imported_names_and_iterations_are_applied() {
        let presets = get_presets();
        for name in [
            "FASS Curve",
            "Fiveflake",
            "Sierpiński Median Curve",
            "Sierpiński Triangle",
            "Scaled Fractal Pentagram",
            "Tiered Fractal Tree",
            "Spatial Four-Way Tree",
        ] {
            assert!(
                presets.iter().any(|preset| preset.name == name),
                "missing {name}"
            );
        }
        let vertigo = presets
            .iter()
            .find(|preset| preset.name == "Vertigo Spiral")
            .expect("Vertigo Spiral should remain available");
        assert_eq!(vertigo.iters, 14);
        assert_eq!(vertigo.max_iters, 14);

        for (name, iterations) in [
            ("Circular Tile", 6),
            ("Spatial Fern", 7),
            ("Sierpiński Median Curve", 8),
            ("Trapezoid Tile", 8),
        ] {
            let preset = presets
                .iter()
                .find(|preset| preset.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(preset.iters, iterations, "{name}");
            assert!(preset.max_iters >= iterations, "{name}");
        }

        for removed in [
            "2Isosceles",
            "2Tile1",
            "2Tile1Med",
            "32 Segment curve",
            "32-segment curve",
            "5-rep-tile",
            "7-rep-tile",
            "A tree",
            "A tree bent to the left",
            "A triangular grid",
            "A triangular pattern",
            "A water weed",
            "A weed",
            "An ordinary fractal tree of various thicknesses of branches",
            "Bird's nest",
            "Birds-Nest2",
            "Blackboard tree",
            "Boat",
            "Border1",
            "Boundary",
            "Bourke Kolem",
            "Cantor set",
            "Cesaro fractal",
            "Classic Sierpinski curve",
            "Cross",
            "Fass1",
            "five lines as star",
            "Generated random L-system 06",
            "Grid",
            "GrowingLeaf2Bush-120",
            "Hilbert curve (3-D)",
            "Hilbert curve II",
            "pine tree",
            "scaled fractal pentagram as a Eulerian graph",
            "Seaweed",
            "serp2",
            "Sierpiński tree",
            "Sierpiński triangle (scaled)",
            "tree simple and efficient",
            "Tree2",
            "Vertigo1",
            "Vertigo2",
            "Willow",
            "Branched acropetal signal",
            "Cellular botanica (propfeds parametric manual)",
            "a curved bush, rotate to see it nicely",
            "a nice decent tree!",
            "a weird tree that looks good from some angles",
        ] {
            assert!(
                presets
                    .iter()
                    .all(|preset| !preset.name.eq_ignore_ascii_case(removed)),
                "removed or renamed preset {removed:?} is still visible"
            );
        }
        assert!(
            presets
                .iter()
                .all(|preset| !preset.name.to_lowercase().starts_with("adh"))
        );
    }

    #[test]
    fn gothic_starwork_preserves_its_imported_identity_and_is_searchable() {
        let presets = get_presets();
        let gothic = presets
            .iter()
            .find(|preset| preset.name == "Gothic Starwork")
            .expect("promoted Gothic preset should exist");

        assert!(gothic.summary.contains("Nested stars"));
        assert!(gothic.matches_search_terms(&[String::from("gothic"), String::from("starwork")]));
        assert!(!gothic.matches_search_terms(&[String::from("gothic.outline")]));
        assert_eq!(gothic.turtle_config.draw_modules, ["X"]);
        assert_eq!(gothic.turtle_config.move_modules, ["Y"]);
        assert!(!gothic.source.contains("match F then nothing"));
        let imported_source = WEB_PRESET_SOURCES
            .iter()
            .find(|source| metadata(source, "ID").as_deref() == Some(GOTHIC_IMPORTED_ID))
            .expect("generated Gothic source should remain available");
        assert_eq!(
            metadata(imported_source, "Name").as_deref(),
            Some("Gothic Starwork")
        );
        assert!(imported_source.contains("# Source   :"));
    }

    #[test]
    fn retired_preset_names_are_not_search_aliases() {
        let presets = get_presets();
        for retired in ["quadkoch", "tree4", "gothic.outline"] {
            assert!(
                presets
                    .iter()
                    .all(|preset| !preset.matches_search_terms(&[retired.to_owned()])),
                "retired name {retired:?} remains searchable"
            );
        }
    }

    #[test]
    fn rotating_curves_declare_stable_orientation_anchors() {
        let presets = get_presets();
        for name in [
            "Heighway Dragon",
            "Sierpiński Arrowhead",
            "Orthogonal Virus",
        ] {
            let preset = presets.iter().find(|preset| preset.name == name).unwrap();
            assert_eq!(
                preset.orientation_anchor,
                Some(OrientationAnchor::Endpoint),
                "{name}"
            );
        }
    }

    #[test]
    fn gothic_carrier_rewrite_preserves_the_imported_turtle_geometry() {
        let imported_source = WEB_PRESET_SOURCES
            .iter()
            .find(|source| metadata(source, "ID").as_deref() == Some(GOTHIC_IMPORTED_ID))
            .expect("generated Gothic source should remain available");
        let imported_config = Turtle2dConfig {
            turn_angle: 36.0_f64.to_radians(),
            ..Turtle2dConfig::default()
        }
        .with_source_metadata(imported_source);
        let carrier_config = Turtle2dConfig {
            turn_angle: 36.0_f64.to_radians(),
            ..Turtle2dConfig::default()
        }
        .with_source_metadata(GOTHIC_STARWORK_SOURCE);

        for iterations in 0..=2 {
            let render = |source: &str, config: Turtle2dConfig| {
                let calculation = calculate(CalculationRequest {
                    grammar: CompiledGrammar::parse(source).unwrap(),
                    iterations,
                    backend: BackendChoice::Cpu,
                    seed: 0,
                    semantics: Default::default(),
                    limits: CalculationLimits::default(),
                })
                .unwrap();
                visualize(VisualizeRequest {
                    generation: &calculation.generation,
                    backend: VisualizerBackend::Cpu,
                    config: VisualizerConfig::Turtle2d(config),
                    context: VisualizationContext::default(),
                })
                .unwrap()
            };

            assert_eq!(
                render(imported_source, imported_config.clone()),
                render(GOTHIC_STARWORK_SOURCE, carrier_config.clone()),
                "Gothic carrier rewrite changed iteration {iterations}",
            );
        }
    }

    #[test]
    fn width_capable_presets_are_available() {
        let mut names = get_presets()
            .into_iter()
            .filter(|preset| {
                preset
                    .source
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .any(|word| matches!(word, "Width" | "WidthIncrease" | "WidthDecrease"))
            })
            .map(|preset| preset.name)
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            [
                "Organic Tapered Tree",
                "Spatial Tiered Tree",
                "Tapered Binary Tree",
                "Tiered Fractal Tree",
            ]
        );
    }

    #[test]
    fn every_generated_web_preset_renders_lines() {
        let mut failures = Vec::new();
        let mut spatial_presets = 0usize;
        let mut presets_with_depth = 0usize;
        for source in WEB_PRESET_SOURCES {
            let preset = parse_preset(source, true);
            let result = CompiledGrammar::parse(&preset.source)
                .map_err(|error| error.to_string())
                .and_then(|grammar| {
                    calculate(CalculationRequest {
                        grammar,
                        iterations: preset.iters as usize,
                        backend: BackendChoice::Cpu,
                        seed: 0,
                        semantics: Default::default(),
                        limits: CalculationLimits::unbounded_production(),
                    })
                    .map_err(|error| error.to_string())
                })
                .and_then(|calculation| {
                    let config = match preset.visualizer {
                        VisualizerKind::Turtle2d => {
                            VisualizerConfig::Turtle2d(preset.turtle_config)
                        }
                        VisualizerKind::Turtle3d => {
                            VisualizerConfig::Turtle3d(Turtle3dConfig::from(preset.turtle_config))
                        }
                        visualizer => VisualizerConfig::for_kind(visualizer),
                    };
                    visualize(VisualizeRequest {
                        generation: &calculation.generation,
                        backend: VisualizerBackend::Cpu,
                        config,
                        context: VisualizationContext::default(),
                    })
                    .map_err(|error| error.to_string())
                });
            match result {
                Ok(Visualization::Scene2d(scene))
                    if scene.primitives.iter().any(|primitive| {
                        matches!(primitive, Primitive2d::Line(_) | Primitive2d::Polygon(_))
                    }) => {}
                Ok(Visualization::Scene3d(scene))
                    if scene.primitives.iter().any(|primitive| {
                        matches!(primitive, Primitive3d::Line(_) | Primitive3d::Polygon(_))
                    }) =>
                {
                    spatial_presets += 1;
                    let mut minimum = f64::INFINITY;
                    let mut maximum = f64::NEG_INFINITY;
                    for primitive in &scene.primitives {
                        match primitive {
                            Primitive3d::Line(line) => {
                                minimum = minimum.min(line.line.0.2).min(line.line.1.2);
                                maximum = maximum.max(line.line.0.2).max(line.line.1.2);
                            }
                            Primitive3d::Polygon(polygon) => {
                                for point in &polygon.vertices {
                                    minimum = minimum.min(point.2);
                                    maximum = maximum.max(point.2);
                                }
                            }
                        }
                    }
                    if maximum - minimum > 1.0e-9 {
                        presets_with_depth += 1;
                    }
                }
                Ok(_) => failures.push(format!("{}: rendered no geometry", preset.name)),
                Err(error) => failures.push(format!("{}: {error}", preset.name)),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
        assert_eq!(spatial_presets, 3);
        assert_eq!(presets_with_depth, 2);
    }
}
