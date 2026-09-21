use std::{
    fs,
    path::{Path, PathBuf},
};

use braken::Grammar;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/abop")
}

fn lsys_files(directory: &Path) -> Vec<PathBuf> {
    let mut files = fs::read_dir(directory)
        .expect("fixture directory should exist")
        .map(|entry| entry.expect("fixture entry should be readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "lsys")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn fixture_with_id(id: &str) -> PathBuf {
    let prefix = format!("{}-", id.to_ascii_lowercase());
    lsys_files(&fixture_root().join("pass"))
        .into_iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix))
        })
        .unwrap_or_else(|| panic!("missing fixture {id}"))
}

fn parse_path(path: &Path) -> Result<Grammar, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Grammar::parse(&source).map_err(|error| format!("{}:\n{error}", path.display()))
}

#[test]
fn every_pass_fixture_parses() {
    let mut failures = Vec::new();
    for path in lsys_files(&fixture_root().join("pass")) {
        if let Err(error) = parse_path(&path) {
            failures.push(error);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn every_failure_fixture_is_rejected() {
    let mut unexpected = Vec::new();
    for path in lsys_files(&fixture_root().join("fail")) {
        if parse_path(&path).is_ok() {
            unexpected.push(path.display().to_string());
        }
    }
    assert!(
        unexpected.is_empty(),
        "fixtures unexpectedly parsed:\n{}",
        unexpected.join("\n"),
    );
}

#[test]
fn fibonacci_fixture_executes() {
    let grammar = parse_path(&fixture_with_id("ABOP-001")).unwrap();
    let program = grammar.compile_cpu().unwrap();
    assert_eq!(program.run(5).unwrap().to_string(), "a b a a b a b a");
}

#[test]
fn parametric_fixture_executes() {
    let grammar = parse_path(&fixture_with_id("ABOP-048")).unwrap();
    let program = grammar.compile_cpu().unwrap();
    let generation = program.run(4).unwrap();
    assert!(!generation.items().is_empty());
}

#[test]
fn branch_context_fixture_executes() {
    let grammar = parse_path(&fixture_with_id("ABOP-037")).unwrap();
    let program = grammar.compile_cpu().unwrap();
    let generation = program.run(1).unwrap();
    assert!(generation.to_string().contains("Fb"));
}

#[test]
fn stochastic_fixture_is_seed_reproducible() {
    let grammar = parse_path(&fixture_with_id("ABOP-034")).unwrap();
    let program = grammar.compile_cpu().unwrap();
    assert_eq!(
        program.run_with_seed(4, 1234).unwrap(),
        program.run_with_seed(4, 1234).unwrap(),
    );
}

#[test]
fn partial_fixture_uses_uniform_random_choice_by_default() {
    let grammar = parse_path(&fixture_with_id("ABOP-058")).unwrap();
    let program = grammar.compile_cpu().unwrap();
    let generation = program.run_with_seed(4, 7).unwrap();
    assert!(!generation.items().is_empty());
}
