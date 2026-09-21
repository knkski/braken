#[path = "../ir_fixtures.rs"]
#[allow(dead_code)]
mod support;

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use braken::{
    BindingSlot, CompiledGrammar, CpuBackend, FloatWidth, GlobalId, IrBackendProfile,
    IrDiagnosticSeverity, IrDocument, IrExpr, IrInstruction, IrJsonError, IrModuleExpr,
    IrPatternArgument, IrProgram, IrWordItem, SymbolId,
};
use support::{
    expected_disassembly_path, expected_json_path, files_with_extension, fixture_root,
    generate_case, load_manifest, requirement_features, validate_run_expectations,
};

#[test]
fn curated_documents_disassembly_execution_and_lineage_match_the_goldens() {
    let root = fixture_root(Path::new(env!("CARGO_MANIFEST_DIR")));
    let manifest = load_manifest(&root).unwrap();

    for case in &manifest.cases {
        let generated = generate_case(&root, case).unwrap();
        assert_eq!(
            fs::read_to_string(expected_json_path(&root, case)).unwrap(),
            generated.json,
            "{} semantic JSON changed",
            case.id
        );
        assert_eq!(
            fs::read_to_string(expected_disassembly_path(&root, case)).unwrap(),
            generated.disassembly,
            "{} disassembly changed",
            case.id
        );
        assert_eq!(generated.features, case.expected_features, "{}", case.id);
        validate_run_expectations(case, &generated).unwrap();

        let parsed = IrDocument::from_json_with_limit(&generated.json, generated.json.len())
            .unwrap_or_else(|error| panic!("{} exact-limit JSON failed: {error}", case.id));
        assert_eq!(parsed, generated.document, "{} JSON round trip", case.id);
        assert!(matches!(
            IrDocument::from_json_with_limit(&generated.json, generated.json.len() - 1),
            Err(IrJsonError::SizeLimitExceeded { .. })
        ));

        let validated = parsed.clone().validate().unwrap();
        assert_eq!(
            validated.disassemble(),
            generated.disassembly,
            "{}",
            case.id
        );
        verify_programs(validated.view().programs(), &case.id);
        verify_production_groups(&validated, &case.id);

        for backend in [
            IrBackendProfile::Cpu,
            IrBackendProfile::CurrentCuda,
            IrBackendProfile::CurrentWgpu,
        ] {
            let report = validated.compatibility(backend);
            let codes = report
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.code.to_owned())
                .collect::<Vec<_>>();
            assert_eq!(
                codes,
                case.compatibility.for_backend(backend),
                "{} {backend} diagnostics",
                case.id
            );
            assert_eq!(
                report.compatible,
                !report
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.severity == IrDiagnosticSeverity::Error),
                "{} {backend} compatibility flag",
                case.id
            );
        }

        let imported = CompiledGrammar::from_ir(parsed.clone()).unwrap();
        let mut normalized = parsed;
        normalized.source = None;
        normalized.source_fingerprint = None;
        assert_eq!(
            IrDocument::from_grammar(imported.grammar()),
            normalized,
            "{} raise/lower semantic round trip",
            case.id
        );

        let legacy = CompiledGrammar::parse(&generated.source).unwrap();
        for float_width in [FloatWidth::F32, FloatWidth::F64] {
            let backend = CpuBackend::new().float_width(float_width);
            let legacy_program = backend.compile(legacy.grammar()).unwrap();
            let ir_program = backend.compile_ir(imported.derivation_ir()).unwrap();
            for run in &case.runs {
                let legacy_generation = legacy_program
                    .run_with_seed(run.iterations, run.seed)
                    .unwrap();
                let ir_generation = ir_program.run_with_seed(run.iterations, run.seed).unwrap();
                assert_eq!(
                    legacy_generation, ir_generation,
                    "{} {float_width} CPU/IR",
                    case.id
                );
                if float_width == FloatWidth::F64 {
                    assert_eq!(ir_generation.to_string(), run.generation, "{}", case.id);
                }
            }
        }
    }
}

#[test]
fn imported_documents_reject_independent_metadata_reference_and_semantic_mutations() {
    let root = fixture_root(Path::new(env!("CARGO_MANIFEST_DIR")));
    let manifest = load_manifest(&root).unwrap();
    let document = |id: &str| {
        let case = manifest.cases.iter().find(|case| case.id == id).unwrap();
        generate_case(&root, case).unwrap().document
    };

    let mut mutated = document("01-deterministic-flat");
    mutated.format = String::from("some-other-ir");
    expect_diagnostic(mutated, "format-name", None);

    let mut mutated = document("01-deterministic-flat");
    mutated.format_version += 1;
    expect_diagnostic(mutated, "format-version", None);

    let mut mutated = document("01-deterministic-flat");
    mutated.semantics_version += 1;
    expect_diagnostic(mutated, "semantics-version", None);

    let mut mutated = document("01-deterministic-flat");
    mutated.source_fingerprint = Some(String::from("fnv1a64:0000000000000000"));
    expect_diagnostic(mutated, "source-fingerprint", None);

    let mut mutated = document("03-parameters-and-patterns");
    first_axiom_module(&mut mutated).symbol = SymbolId(u32::MAX);
    expect_diagnostic(mutated, "symbol-reference", Some("axiom"));

    let mut mutated = document("03-parameters-and-patterns");
    first_axiom_module(&mut mutated)
        .arguments
        .push(IrExpr::Number(0.0_f64.to_bits()));
    expect_diagnostic(mutated, "symbol-arity", Some("axiom"));

    let mut mutated = document("02-globals-and-expressions");
    mutated.globals[0].value = IrExpr::Global(GlobalId(u32::MAX));
    expect_diagnostic(mutated, "global-reference", Some("global g0"));

    let mut mutated = document("03-parameters-and-patterns");
    mutated.productions[0].condition = Some(IrExpr::Binding(BindingSlot(u32::MAX)));
    expect_diagnostic(mutated, "binding-reference", Some("production p0"));

    let mut mutated = document("03-parameters-and-patterns");
    mutated.productions[0].center.arguments[0] = IrPatternArgument::Bind(BindingSlot(u32::MAX));
    expect_diagnostic(mutated, "binding-definition", Some("production p0"));

    let mut mutated = document("03-parameters-and-patterns");
    let duplicate = mutated.productions[0].bindings[0].clone();
    mutated.productions[0].bindings.push(duplicate);
    expect_diagnostic(mutated, "invalid-binding", Some("production p0"));

    let mut mutated = document("05-stochastic-wildcard");
    mutated.productions[0].weight = Some(IrExpr::Number((-1.0_f64).to_bits()));
    expect_diagnostic(mutated, "grammar-validation", None);
}

#[test]
fn interchange_json_rejects_malformed_syntax_and_noncanonical_number_bits() {
    assert!(matches!(
        IrDocument::from_json_with_limit("{", 1),
        Err(IrJsonError::InvalidJson(_))
    ));

    let root = fixture_root(Path::new(env!("CARGO_MANIFEST_DIR")));
    let manifest = load_manifest(&root).unwrap();
    let case = manifest
        .cases
        .iter()
        .find(|case| case.id == "02-globals-and-expressions")
        .unwrap();
    let json = generate_case(&root, case).unwrap().json;
    let malformed_bits = json.replacen("0x", "0X", 1);
    assert!(matches!(
        IrDocument::from_json_with_limit(&malformed_bits, malformed_bits.len()),
        Err(IrJsonError::InvalidJson(_))
    ));
}

#[test]
fn every_existing_valid_fixture_round_trips_through_the_shared_ir() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let roots = [
        manifest_dir.join("tests/fixtures/abop/pass"),
        manifest_dir.join("tests/fixtures/presets/pass"),
        manifest_dir.join("tests/fixtures/web/pass"),
    ];
    let mut grammar_count = 0_usize;
    let mut feature_counts = [0_usize; 9];

    for root in roots {
        for source_path in files_with_extension(&root, "lsys").unwrap() {
            let source = fs::read_to_string(&source_path).unwrap();
            let compiled = CompiledGrammar::parse(&source)
                .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
            let document = compiled.to_ir_document();
            let json = document.to_json_pretty().unwrap();
            let decoded = IrDocument::from_json_with_limit(&json, json.len()).unwrap();
            assert_eq!(
                decoded,
                document,
                "{} JSON round trip",
                source_path.display()
            );
            let imported = CompiledGrammar::from_ir(decoded.clone())
                .unwrap_or_else(|error| panic!("{}: {error}", source_path.display()));
            assert_eq!(
                imported.ir().disassemble(),
                compiled.ir().disassemble(),
                "{} disassembly",
                source_path.display()
            );
            verify_programs(imported.ir().programs(), &source_path.display().to_string());

            let mut normalized = decoded;
            normalized.source = None;
            normalized.source_fingerprint = None;
            assert_eq!(
                IrDocument::from_grammar(imported.grammar()),
                normalized,
                "{} raise/lower semantic round trip",
                source_path.display()
            );

            let requirements = imported.ir().requirements();
            for (index, enabled) in [
                requirements.parameterized_modules,
                requirements.contextual_productions,
                requirements.structural_contexts,
                requirements.context_filter,
                requirements.dynamic_conditions,
                requirements.dynamic_weights,
                requirements.weighted_productions,
                requirements.structural_branches,
                !requirements.builtins.is_empty(),
            ]
            .into_iter()
            .enumerate()
            {
                feature_counts[index] += usize::from(enabled);
            }
            grammar_count += 1;
        }
    }

    assert!(grammar_count >= 1_000, "fixture corpus unexpectedly shrank");
    for (name, count) in [
        "parameters",
        "context",
        "structural context",
        "filters",
        "dynamic conditions",
        "weighted productions",
        "branches",
        "builtins",
    ]
    .into_iter()
    .zip(
        feature_counts
            .into_iter()
            .enumerate()
            .filter_map(|(index, count)| (index != 5).then_some(count)),
    ) {
        assert!(count > 0, "fixture corpus no longer covers {name}");
    }
}

fn first_axiom_module(document: &mut IrDocument) -> &mut IrModuleExpr {
    match document.axiom.items.first_mut().unwrap() {
        IrWordItem::Module(module) => module,
        IrWordItem::Branch(_) => panic!("fixture unexpectedly starts with a branch"),
    }
}

fn expect_diagnostic(mutated: IrDocument, code: &str, entity: Option<&str>) {
    let errors = match mutated.validate() {
        Ok(_) => panic!("mutation unexpectedly validated for diagnostic {code}"),
        Err(errors) => errors,
    };
    assert!(
        errors.0.iter().any(|diagnostic| {
            diagnostic.code == code && diagnostic.entity.as_deref() == entity
        }),
        "missing [{code}] on {entity:?}; found {:#?}",
        errors.0
    );
}

fn verify_programs(programs: &[IrProgram], case: &str) {
    for (index, program) in programs.iter().enumerate() {
        assert_eq!(usize::try_from(program.id.0).unwrap(), index, "{case}");
        assert!(!program.instructions.is_empty(), "{case} {}", program.id);
        let mut depths = vec![None; program.instructions.len()];
        let mut pending = VecDeque::from([(0_usize, 0_u32)]);
        let mut maximum = 0_u32;
        let mut returned = false;
        while let Some((instruction_index, depth)) = pending.pop_front() {
            assert!(
                instruction_index < program.instructions.len(),
                "{case} {} jumps past its program",
                program.id
            );
            if let Some(previous) = depths[instruction_index] {
                assert_eq!(previous, depth, "{case} {} stack merge", program.id);
                continue;
            }
            depths[instruction_index] = Some(depth);
            maximum = maximum.max(depth);
            let instruction = &program.instructions[instruction_index];
            let next = instruction_index + 1;
            match instruction {
                IrInstruction::PushNumber(_)
                | IrInstruction::PushBool(_)
                | IrInstruction::LoadGlobal(_)
                | IrInstruction::LoadBinding(_) => {
                    let depth = depth.checked_add(1).expect("IR stack depth overflow");
                    maximum = maximum.max(depth);
                    pending.push_back((next, depth));
                }
                IrInstruction::Call { arity, .. } => {
                    assert!(depth >= *arity, "{case} {} call underflow", program.id);
                    let depth = depth - arity + 1;
                    maximum = maximum.max(depth);
                    pending.push_back((next, depth));
                }
                IrInstruction::Unary(_) => {
                    assert!(depth >= 1, "{case} {} unary underflow", program.id);
                    pending.push_back((next, depth));
                }
                IrInstruction::Binary(_) => {
                    assert!(depth >= 2, "{case} {} binary underflow", program.id);
                    pending.push_back((next, depth - 1));
                }
                IrInstruction::JumpIfFalse(target) | IrInstruction::JumpIfTrue(target) => {
                    assert!(depth >= 1, "{case} {} branch underflow", program.id);
                    pending.push_back((usize::try_from(*target).unwrap(), depth));
                    pending.push_back((next, depth));
                }
                IrInstruction::Pop => {
                    assert!(depth >= 1, "{case} {} pop underflow", program.id);
                    pending.push_back((next, depth - 1));
                }
                IrInstruction::Return => {
                    assert_eq!(depth, 1, "{case} {} return stack", program.id);
                    returned = true;
                }
            }
        }
        assert!(returned, "{case} {} never returns", program.id);
        assert!(
            depths.iter().all(Option::is_some),
            "{case} {} contains unreachable bytecode",
            program.id
        );
        assert_eq!(
            program.max_stack, maximum,
            "{case} {} max stack",
            program.id
        );
    }
}

fn verify_production_groups(ir: &braken::DerivationIr, case: &str) {
    let view = ir.view();
    let groups = view.production_groups();
    assert!(
        groups
            .windows(2)
            .all(|pair| pair[0].symbol < pair[1].symbol),
        "{case} production groups are not symbol ordered"
    );
    let mut productions = Vec::new();
    for group in groups {
        for production in &group.productions {
            assert_eq!(
                view.production(*production).unwrap().center.symbol,
                group.symbol,
                "{case} production group membership"
            );
            productions.push(production.0);
        }
    }
    productions.sort_unstable();
    assert_eq!(
        productions,
        (0..u32::try_from(view.document().productions.len()).unwrap()).collect::<Vec<_>>(),
        "{case} production grouping completeness"
    );
}

#[test]
fn curated_feature_names_are_derived_from_requirements() {
    let root = fixture_root(Path::new(env!("CARGO_MANIFEST_DIR")));
    let manifest = load_manifest(&root).unwrap();
    for case in &manifest.cases {
        let generated = generate_case(&root, case).unwrap();
        assert_eq!(
            requirement_features(
                generated
                    .document
                    .clone()
                    .validate()
                    .unwrap()
                    .view()
                    .requirements()
            ),
            case.expected_features,
            "{}",
            case.id
        );
    }
}
