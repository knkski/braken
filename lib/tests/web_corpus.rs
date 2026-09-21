use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/web")
}

#[test]
fn web_fixture_manifest_matches_files() {
    let root = fixture_root();
    let expected = fs::read_to_string(root.join("manifest.txt")).unwrap();
    let expected = expected
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut actual = fs::read_dir(root.join("pass"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".lsys"))
        .collect::<Vec<_>>();
    actual.sort();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 1_011);
}

#[test]
fn every_web_fixture_parses_and_executes_one_iteration() {
    let root = fixture_root();
    let manifest = fs::read_to_string(root.join("manifest.txt")).unwrap();
    let mut failures = Vec::new();
    for name in manifest
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let path = root.join("pass").join(name);
        let result = fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|source| CompiledGrammar::parse(&source).map_err(|error| error.to_string()))
            .and_then(|grammar| {
                calculate(CalculationRequest {
                    grammar,
                    iterations: 1,
                    backend: BackendChoice::Cpu,
                    seed: 0,
                    semantics: Default::default(),
                    limits: CalculationLimits::default(),
                })
                .map_err(|error| error.to_string())
            });
        if let Err(error) = result {
            failures.push(format!("{name}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn lsystem_files_do_not_use_proposed_heading() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut offenders = Vec::new();
    for root in [
        workspace.join("lib/tests/fixtures"),
        workspace.join("gui/presets"),
    ] {
        collect_proposed_markers(&root, &mut offenders);
    }
    assert!(
        offenders.is_empty(),
        "forbidden heading in:\n{}",
        offenders.join("\n")
    );
}

fn collect_proposed_markers(directory: &Path, offenders: &mut Vec<String>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_proposed_markers(&path, offenders);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "lsys")
            && fs::read_to_string(&path)
                .unwrap()
                .lines()
                .any(|line| line == "# Proposed .lsys:")
        {
            offenders.push(path.display().to_string());
        }
    }
}
