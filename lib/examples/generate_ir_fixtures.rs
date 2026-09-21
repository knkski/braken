//! Generate or check the curated derivation-IR fixture artifacts.

#[path = "../ir_fixtures.rs"]
#[allow(dead_code)]
mod ir_fixtures;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::Path;

use ir_fixtures::{
    GeneratedCase, expected_disassembly_path, expected_json_path, expected_output_paths,
    files_with_extension, fixture_root, generate_case, load_manifest, render_coverage,
    render_gallery, validate_run_expectations,
};

enum Mode {
    Check,
    Update,
    Show(String),
}

fn main() -> Result<(), String> {
    let mode = parse_args()?;
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = fixture_root(manifest_dir);
    let manifest = load_manifest(&root)?;
    let mut generated = BTreeMap::new();
    for case in &manifest.cases {
        generated.insert(case.id.clone(), generate_case(&root, case)?);
    }

    if let Mode::Show(id) = &mode {
        let case = manifest
            .cases
            .iter()
            .find(|case| &case.id == id)
            .ok_or_else(|| format!("unknown IR fixture {id:?}"))?;
        let generated = &generated[&case.id];
        println!(
            "# {}\n\n## Source\n\n{}\n## Disassembly\n\n{}\n## JSON\n\n{}",
            case.title, generated.source, generated.disassembly, generated.json
        );
        return Ok(());
    }

    validate_declared_expectations(&manifest, &generated)?;
    let generated_values = manifest
        .cases
        .iter()
        .map(|case| &generated[&case.id])
        .collect::<Vec<_>>();
    let gallery = render_gallery(&manifest, &generated);
    let coverage = render_coverage(manifest_dir, &generated_values)?;
    let gallery_path = root.join("GALLERY.md");
    let coverage_path = root.join("COVERAGE.md");

    match mode {
        Mode::Check => {
            let mut stale = Vec::new();
            for case in &manifest.cases {
                check_file(
                    &expected_json_path(&root, case),
                    &generated[&case.id].json,
                    &mut stale,
                );
                check_file(
                    &expected_disassembly_path(&root, case),
                    &generated[&case.id].disassembly,
                    &mut stale,
                );
            }
            check_file(&gallery_path, &gallery, &mut stale);
            check_file(&coverage_path, &coverage, &mut stale);
            let expected = expected_output_paths(&root, &manifest);
            let actual: BTreeSet<_> = files_with_extension(&root.join("expected"), "json")?
                .into_iter()
                .chain(files_with_extension(&root.join("expected"), "txt")?)
                .collect();
            for orphan in actual.difference(&expected) {
                stale.push(format!("orphan generated output {}", orphan.display()));
            }
            if stale.is_empty() {
                println!("all {} IR fixtures are current", manifest.cases.len());
                Ok(())
            } else {
                Err(format!(
                    "IR fixture artifacts are stale:\n{}\n\nRun with --update and review both semantic JSON and disassembly changes.",
                    stale.join("\n")
                ))
            }
        }
        Mode::Update => {
            fs::create_dir_all(root.join("expected"))
                .map_err(|error| format!("could not create expected output directory: {error}"))?;
            for case in &manifest.cases {
                write_file(&expected_json_path(&root, case), &generated[&case.id].json)?;
                write_file(
                    &expected_disassembly_path(&root, case),
                    &generated[&case.id].disassembly,
                )?;
            }
            write_file(&gallery_path, &gallery)?;
            write_file(&coverage_path, &coverage)?;
            let expected = expected_output_paths(&root, &manifest);
            let actual: BTreeSet<_> = files_with_extension(&root.join("expected"), "json")?
                .into_iter()
                .chain(files_with_extension(&root.join("expected"), "txt")?)
                .collect();
            let orphans = actual.difference(&expected).collect::<Vec<_>>();
            if !orphans.is_empty() {
                return Err(format!(
                    "generated current outputs but found orphan files that must be reviewed and removed manually:\n{}",
                    orphans
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
            }
            println!("updated {} IR fixtures", manifest.cases.len());
            Ok(())
        }
        Mode::Show(_) => unreachable!(),
    }
}

fn parse_args() -> Result<Mode, String> {
    let mut arguments = env::args().skip(1);
    let Some(first) = arguments.next() else {
        return Err(usage());
    };
    let mode = match first.as_str() {
        "--check" => Mode::Check,
        "--update" => Mode::Update,
        "--case" => Mode::Show(
            arguments
                .next()
                .ok_or_else(|| String::from("--case requires a fixture id"))?,
        ),
        "-h" | "--help" => {
            println!("{}", usage());
            std::process::exit(0);
        }
        _ => return Err(format!("unknown argument {first:?}\n\n{}", usage())),
    };
    if let Some(extra) = arguments.next() {
        return Err(format!("unexpected argument {extra:?}\n\n{}", usage()));
    }
    Ok(mode)
}

fn usage() -> String {
    String::from(
        "Generate derivation-IR fixtures.\n\nUsage:\n  generate_ir_fixtures --check\n  generate_ir_fixtures --update\n  generate_ir_fixtures --case <id>",
    )
}

fn validate_declared_expectations(
    manifest: &ir_fixtures::Manifest,
    generated: &BTreeMap<String, GeneratedCase>,
) -> Result<(), String> {
    for case in &manifest.cases {
        let actual = &generated[&case.id];
        validate_run_expectations(case, actual)?;
        if actual.features != case.expected_features {
            return Err(format!(
                "{} feature declaration differs\nexpected: {:?}\nactual:   {:?}",
                case.id, case.expected_features, actual.features
            ));
        }
        for (backend, expected) in [
            ("cpu", &case.compatibility.cpu),
            ("cuda", &case.compatibility.cuda),
            ("wgpu", &case.compatibility.wgpu),
        ] {
            if &actual.compatibility[backend] != expected {
                return Err(format!(
                    "{} {backend} compatibility differs\nexpected: {expected:?}\nactual:   {:?}",
                    case.id, actual.compatibility[backend]
                ));
            }
        }
    }
    Ok(())
}

fn check_file(path: &Path, expected: &str, stale: &mut Vec<String>) {
    match fs::read_to_string(path) {
        Ok(actual) if actual == expected => {}
        Ok(_) => stale.push(format!("content differs: {}", path.display())),
        Err(error) => stale.push(format!("cannot read {}: {error}", path.display())),
    }
}

fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    fs::write(path, contents)
        .map_err(|error| format!("could not write {}: {error}", path.display()))
}
