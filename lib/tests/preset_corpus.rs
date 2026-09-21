use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};
use std::fs;
use std::path::{Path, PathBuf};

const EXPECTED_FIXTURES: &[&str] = &[
    "fancy-triangle.lsys",
    "h-tree.lsys",
    "koch-snowflake.lsys",
    "levy-c-curve.lsys",
    "penrose-tiling.lsys",
    "pentaflake.lsys",
    "sierpinski-triangle.lsys",
    "stochastic-plant.lsys",
    "terdragon.lsys",
    "tree-row.lsys",
    "virus.lsys",
];

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/presets/pass")
}

fn fixture_files() -> Vec<PathBuf> {
    let mut files = fs::read_dir(fixture_root())
        .expect("preset fixture directory should exist")
        .map(|entry| entry.expect("preset fixture should be readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "lsys")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

#[test]
fn preset_fixture_set_is_complete() {
    let names = fixture_files()
        .into_iter()
        .map(|path| path.file_name().unwrap().to_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(names, EXPECTED_FIXTURES);
}

#[test]
fn every_preset_fixture_parses_and_executes() {
    for path in fixture_files() {
        let source = fs::read_to_string(&path).unwrap();
        let grammar = CompiledGrammar::parse(&source)
            .unwrap_or_else(|error| panic!("{} failed to parse: {error}", path.display()));
        calculate(CalculationRequest {
            grammar,
            iterations: 2,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        })
        .unwrap_or_else(|error| panic!("{} failed to execute: {error}", path.display()));
    }
}
