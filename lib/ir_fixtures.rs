//! Shared support for the checked-in IR fixture generator and integration tests.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use braken::{
    CompiledGrammar, CpuBackend, DerivationIr, IrBackendProfile, IrDocument, IrRequirements,
};
use serde::Deserialize;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
pub struct Case {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub source: String,
    pub expected_features: Vec<String>,
    pub compatibility: CompatibilityExpectation,
    pub runs: Vec<RunExpectation>,
}

#[derive(Debug, Deserialize)]
pub struct CompatibilityExpectation {
    pub cpu: Vec<String>,
    pub cuda: Vec<String>,
    pub wgpu: Vec<String>,
}

impl CompatibilityExpectation {
    pub fn for_backend(&self, backend: IrBackendProfile) -> &[String] {
        match backend {
            IrBackendProfile::Cpu => &self.cpu,
            IrBackendProfile::CurrentCuda => &self.cuda,
            IrBackendProfile::CurrentWgpu => &self.wgpu,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RunExpectation {
    pub iterations: usize,
    pub seed: u64,
    pub generation: String,
    #[serde(default)]
    pub lineage: Option<Vec<usize>>,
}

pub struct GeneratedCase {
    pub source: String,
    pub document: IrDocument,
    pub json: String,
    pub disassembly: String,
    pub features: Vec<String>,
    pub compatibility: BTreeMap<String, Vec<String>>,
}

pub fn fixture_root(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("tests/fixtures/ir")
}

pub fn load_manifest(root: &Path) -> Result<Manifest, String> {
    let path = root.join("manifest.json");
    let json = fs::read_to_string(&path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let manifest: Manifest = serde_json::from_str(&json)
        .map_err(|error| format!("could not decode {}: {error}", path.display()))?;
    validate_manifest(root, &manifest)?;
    Ok(manifest)
}

fn validate_manifest(root: &Path, manifest: &Manifest) -> Result<(), String> {
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "IR fixture schema version {} is unsupported; expected {SCHEMA_VERSION}",
            manifest.schema_version
        ));
    }
    let mut ids = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut previous = None;
    for case in &manifest.cases {
        if case.id.is_empty() || !ids.insert(case.id.as_str()) {
            return Err(format!("empty or duplicate IR fixture id {:?}", case.id));
        }
        if previous.is_some_and(|previous: &str| previous >= case.id.as_str()) {
            return Err(String::from(
                "IR fixture manifest cases are not sorted by id",
            ));
        }
        previous = Some(case.id.as_str());
        let expected_source = format!("cases/{}.lsys", case.id);
        if case.source != expected_source {
            return Err(format!(
                "IR fixture {} should use source {expected_source:?}, found {:?}",
                case.id, case.source
            ));
        }
        if !sources.insert(case.source.as_str()) {
            return Err(format!("duplicate IR fixture source {:?}", case.source));
        }
        if !root.join(&case.source).is_file() {
            return Err(format!(
                "IR fixture {} is missing source {}",
                case.id,
                root.join(&case.source).display()
            ));
        }
        if !strictly_sorted(&case.expected_features) {
            return Err(format!(
                "expected_features for {} must be sorted and unique",
                case.id
            ));
        }
        if case.runs.is_empty() {
            return Err(format!(
                "IR fixture {} has no execution expectations",
                case.id
            ));
        }
    }

    let actual_sources = files_with_extension(&root.join("cases"), "lsys")?
        .into_iter()
        .map(|path| {
            path.strip_prefix(root)
                .expect("collected path is below fixture root")
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<BTreeSet<_>>();
    let declared_sources = sources
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if actual_sources != declared_sources {
        return Err(format!(
            "IR fixture sources differ from manifest\ndeclared: {declared_sources:#?}\nactual: {actual_sources:#?}"
        ));
    }
    Ok(())
}

fn strictly_sorted(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

pub fn generate_case(root: &Path, case: &Case) -> Result<GeneratedCase, String> {
    let source_path = root.join(&case.source);
    let source = fs::read_to_string(&source_path)
        .map_err(|error| format!("could not read {}: {error}", source_path.display()))?;
    let grammar = CompiledGrammar::parse(&source)
        .map_err(|error| format!("{} did not compile: {error}", source_path.display()))?;
    let document = grammar.to_ir_document();
    let json = document
        .to_json_pretty()
        .map_err(|error| format!("{} did not serialize: {error}", case.id))?;
    let ir = grammar.derivation_ir();
    Ok(GeneratedCase {
        source,
        document,
        json: format!("{json}\n"),
        disassembly: ir.disassemble(),
        features: requirement_features(ir.view().requirements()),
        compatibility: compatibility_codes(ir),
    })
}

pub fn validate_run_expectations(case: &Case, generated: &GeneratedCase) -> Result<(), String> {
    let grammar = CompiledGrammar::from_ir(generated.document.clone())
        .map_err(|error| format!("{} generated invalid IR: {error}", case.id))?;
    let program = CpuBackend::new()
        .compile_ir(grammar.derivation_ir())
        .map_err(|error| format!("{} IR did not compile for CPU: {error}", case.id))?;
    for run in &case.runs {
        let generation = program
            .run_with_seed(run.iterations, run.seed)
            .map_err(|error| format!("{} run failed: {error}", case.id))?;
        if generation.to_string() != run.generation {
            return Err(format!(
                "{} execution differs at iteration {} with seed {}\nexpected: {}\nactual:   {}",
                case.id, run.iterations, run.seed, run.generation, generation
            ));
        }
        if let Some(expected_lineage) = &run.lineage {
            let input_iterations = run.iterations.checked_sub(1).ok_or_else(|| {
                format!("{} cannot declare lineage for an axiom-only run", case.id)
            })?;
            let input = program
                .run_with_seed(input_iterations, run.seed)
                .map_err(|error| format!("{} lineage setup failed: {error}", case.id))?;
            let generation_index = u64::try_from(input_iterations)
                .map_err(|error| format!("{} iteration index is not portable: {error}", case.id))?;
            let (traced, _, lineage) = program
                .trace_rewrite_with_control(&input, generation_index, run.seed, || false, |_| {})
                .map_err(|error| format!("{} lineage trace failed: {error}", case.id))?;
            if traced != generation {
                return Err(format!(
                    "{} traced generation differs from its incremental execution",
                    case.id
                ));
            }
            if lineage.successor_modules_per_input() != expected_lineage {
                return Err(format!(
                    "{} lineage differs at iteration {} with seed {}\nexpected: {:?}\nactual:   {:?}",
                    case.id,
                    run.iterations,
                    run.seed,
                    expected_lineage,
                    lineage.successor_modules_per_input(),
                ));
            }
        }
    }
    Ok(())
}

pub fn expected_json_path(root: &Path, case: &Case) -> PathBuf {
    root.join("expected").join(format!("{}.ir.json", case.id))
}

pub fn expected_disassembly_path(root: &Path, case: &Case) -> PathBuf {
    root.join("expected").join(format!("{}.ir.txt", case.id))
}

pub fn expected_output_paths(root: &Path, manifest: &Manifest) -> BTreeSet<PathBuf> {
    manifest
        .cases
        .iter()
        .flat_map(|case| {
            [
                expected_json_path(root, case),
                expected_disassembly_path(root, case),
            ]
        })
        .collect()
}

pub fn requirement_features(requirements: &IrRequirements) -> Vec<String> {
    let mut features = BTreeSet::new();
    for (enabled, name) in [
        (requirements.parameterized_modules, "parameterized-modules"),
        (
            requirements.contextual_productions,
            "contextual-productions",
        ),
        (requirements.structural_contexts, "structural-contexts"),
        (requirements.context_filter, "context-filter"),
        (requirements.dynamic_conditions, "dynamic-conditions"),
        (requirements.dynamic_weights, "dynamic-weights"),
        (requirements.weighted_productions, "weighted-productions"),
        (requirements.structural_branches, "structural-branches"),
    ] {
        if enabled {
            features.insert(name.to_owned());
        }
    }
    features.extend(
        requirements
            .builtins
            .iter()
            .map(|builtin| format!("builtin:{builtin}")),
    );
    features.into_iter().collect()
}

pub fn compatibility_codes(ir: &DerivationIr) -> BTreeMap<String, Vec<String>> {
    [
        ("cpu", IrBackendProfile::Cpu),
        ("cuda", IrBackendProfile::CurrentCuda),
        ("wgpu", IrBackendProfile::CurrentWgpu),
    ]
    .into_iter()
    .map(|(name, backend)| {
        (
            name.to_owned(),
            ir.compatibility(backend)
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.code.to_owned())
                .collect(),
        )
    })
    .collect()
}

pub fn render_gallery(manifest: &Manifest, generated: &BTreeMap<String, GeneratedCase>) -> String {
    use std::fmt::Write as _;

    let mut output = String::from(
        "# Shared derivation IR gallery\n\n\
         This file is generated by `generate_ir_fixtures`. It places each focused source grammar\n\
         beside the checked-in IR disassembly used by the golden tests. Update it only through the\n\
         documented fixture command.\n\n\
         | Case | Focus | CPU | CUDA | WGPU |\n\
         | --- | --- | --- | --- | --- |\n",
    );
    for case in &manifest.cases {
        let generated = &generated[&case.id];
        let _ = writeln!(
            output,
            "| [{}](#{}) | {} | {} | {} | {} |",
            case.title,
            markdown_anchor(&case.title),
            feature_label(&case.expected_features),
            compatibility_label(&generated.compatibility["cpu"]),
            compatibility_label(&generated.compatibility["cuda"]),
            compatibility_label(&generated.compatibility["wgpu"]),
        );
    }
    for case in &manifest.cases {
        let generated = &generated[&case.id];
        let _ = write!(
            output,
            "\n## {}\n\n{}\n\n- Fixture: `{}`\n- Features: {}\n- Expected JSON: [`expected/{}.ir.json`](expected/{}.ir.json)\n- Compatibility: CPU {}; CUDA {}; WGPU {}\n\n### Source\n\n```text\n{}```\n\n### IR disassembly\n\n```text\n{}```\n\n### Expected executions\n\n| Iterations | Seed | Generation | Final-step lineage |\n| ---: | ---: | --- | --- |\n",
            case.title,
            case.summary,
            case.source,
            feature_label(&case.expected_features),
            case.id,
            case.id,
            compatibility_label(&generated.compatibility["cpu"]),
            compatibility_label(&generated.compatibility["cuda"]),
            compatibility_label(&generated.compatibility["wgpu"]),
            ensure_trailing_newline(&generated.source),
            ensure_trailing_newline(&generated.disassembly),
        );
        for run in &case.runs {
            let lineage = run.lineage.as_ref().map_or_else(
                || String::from("—"),
                |counts| {
                    counts
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                },
            );
            let _ = writeln!(
                output,
                "| {} | {} | `{}` | {} |",
                run.iterations,
                run.seed,
                run.generation.replace('|', "\\|"),
                lineage,
            );
        }
    }
    output
}

fn compatibility_label(codes: &[String]) -> String {
    if codes.is_empty() {
        String::from("compatible")
    } else {
        codes.join(", ")
    }
}

fn feature_label(features: &[String]) -> String {
    if features.is_empty() {
        String::from("none")
    } else {
        features.join(", ")
    }
}

fn markdown_anchor(title: &str) -> String {
    title
        .chars()
        .filter_map(|character| {
            if character.is_ascii_alphanumeric() {
                Some(character.to_ascii_lowercase())
            } else if character.is_whitespace() || character == '-' {
                Some('-')
            } else {
                None
            }
        })
        .collect()
}

fn ensure_trailing_newline(value: &str) -> String {
    if value.ends_with('\n') {
        value.to_owned()
    } else {
        format!("{value}\n")
    }
}

#[derive(Default)]
struct Coverage {
    grammars: usize,
    parameterized: usize,
    contextual: usize,
    structural_contexts: usize,
    filters: usize,
    dynamic_conditions: usize,
    dynamic_weights: usize,
    weighted: usize,
    branches: usize,
    builtins: usize,
}

impl Coverage {
    fn include(&mut self, requirements: &IrRequirements) {
        self.grammars += 1;
        self.parameterized += usize::from(requirements.parameterized_modules);
        self.contextual += usize::from(requirements.contextual_productions);
        self.structural_contexts += usize::from(requirements.structural_contexts);
        self.filters += usize::from(requirements.context_filter);
        self.dynamic_conditions += usize::from(requirements.dynamic_conditions);
        self.dynamic_weights += usize::from(requirements.dynamic_weights);
        self.weighted += usize::from(requirements.weighted_productions);
        self.branches += usize::from(requirements.structural_branches);
        self.builtins += usize::from(!requirements.builtins.is_empty());
    }
}

pub fn render_coverage(manifest_dir: &Path, curated: &[&GeneratedCase]) -> Result<String, String> {
    use std::fmt::Write as _;

    let corpora = [
        ("Curated IR", None, Some(curated)),
        (
            "ABOP pass",
            Some(manifest_dir.join("tests/fixtures/abop/pass")),
            None,
        ),
        (
            "Preset pass",
            Some(manifest_dir.join("tests/fixtures/presets/pass")),
            None,
        ),
        (
            "Web pass",
            Some(manifest_dir.join("tests/fixtures/web/pass")),
            None,
        ),
    ];
    let mut rows = Vec::new();
    for (name, path, generated) in corpora {
        let mut coverage = Coverage::default();
        if let Some(generated) = generated {
            for case in generated {
                let ir = case
                    .document
                    .clone()
                    .validate()
                    .map_err(|error| error.to_string())?;
                coverage.include(ir.view().requirements());
            }
        }
        if let Some(path) = path {
            for source_path in files_with_extension(&path, "lsys")? {
                let source = fs::read_to_string(&source_path).map_err(|error| {
                    format!("could not read {}: {error}", source_path.display())
                })?;
                let grammar = CompiledGrammar::parse(&source).map_err(|error| {
                    format!("{} did not compile: {error}", source_path.display())
                })?;
                coverage.include(grammar.ir().requirements());
            }
        }
        rows.push((name, coverage));
    }

    let mut output = String::from(
        "# Shared derivation IR corpus coverage\n\n\
         This generated report counts grammars that exercise each IR feature. It is a breadth\n\
         signal, not a substitute for the exact curated goldens in [`GALLERY.md`](GALLERY.md).\n\n\
         | Corpus | Grammars | Parameters | Context | Structural context | Filters | Dynamic conditions | Dynamic weights | Weighted | Branches | Builtins |\n\
         | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for (name, coverage) in rows {
        let _ = writeln!(
            output,
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            coverage.grammars,
            coverage.parameterized,
            coverage.contextual,
            coverage.structural_contexts,
            coverage.filters,
            coverage.dynamic_conditions,
            coverage.dynamic_weights,
            coverage.weighted,
            coverage.branches,
            coverage.builtins,
        );
    }
    Ok(output)
}

pub fn files_with_extension(directory: &Path, extension: &str) -> Result<Vec<PathBuf>, String> {
    fn visit(directory: &Path, extension: &str, output: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in fs::read_dir(directory)
            .map_err(|error| format!("could not read {}: {error}", directory.display()))?
        {
            let path = entry
                .map_err(|error| format!("could not read {} entry: {error}", directory.display()))?
                .path();
            if path.is_dir() {
                visit(&path, extension, output)?;
            } else if path
                .extension()
                .is_some_and(|candidate| candidate == extension)
            {
                output.push(path);
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(directory, extension, &mut output)?;
    output.sort();
    Ok(output)
}
