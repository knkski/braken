use braken::{
    AmbiguousRulePolicy, BackendChoice, CalculationError, CalculationLimits, CalculationRequest,
    CompiledGrammar, CpuBackend, CpuExecutionError, Grammar, calculate, calculate_with_progress,
};

fn run(source: &str, iterations: usize) -> String {
    let grammar = CompiledGrammar::parse(source).unwrap();
    let result = calculate(CalculationRequest {
        grammar,
        iterations,
        backend: BackendChoice::Cpu,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits::default(),
    })
    .unwrap();

    result.generation.to_string()
}

#[test]
fn reports_each_cpu_iteration_and_can_cancel() {
    let grammar = CompiledGrammar::parse("axiom F; match F then F F;").unwrap();
    let mut seen = Vec::new();
    let error = calculate_with_progress(
        CalculationRequest {
            grammar,
            iterations: 5,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        },
        |progress| {
            seen.push((progress.completed_iterations, progress.modules));
            progress.completed_iterations < 2
        },
    )
    .unwrap_err();

    assert!(matches!(error, CalculationError::Cancelled));
    assert_eq!(seen, vec![(0, 1), (1, 2), (2, 4)]);
}

#[test]
fn parses_generated_sample() {
    let source = include_str!("fixtures/abop/pass/abop-002-anabaena-catenula-dol-model.lsys");
    Grammar::parse(source).unwrap();
}

#[test]
fn rewrites_basic_system() {
    let source = r#"
        axiom b;
        match a then a b;
        match b then a;
    "#;

    assert_eq!(run(source, 0), "b");
    assert_eq!(run(source, 1), "a");
    assert_eq!(run(source, 2), "a b");
    assert_eq!(run(source, 3), "a b a");
}

#[test]
fn rewrites_parameters_and_conditions() {
    let source = r#"
        axiom B(2) A(4, 4);
        match A(x, y) when y <= 3 then A(x * 2, x + y);
        match A(x, y) when y > 3 then B(x) A(x / y, 0);
        match B(x) when x < 1 then C;
        match B(x) when x >= 1 then B(x - 1);
    "#;

    assert_eq!(run(source, 0), "B(2) A(4,4)");
    assert_eq!(run(source, 1), "B(1) B(4) A(1,0)");
    assert_eq!(run(source, 2), "B(0) B(3) A(2,1)");
    assert_eq!(run(source, 3), "C B(2) A(4,3)");
}

#[test]
fn preserves_branches_structurally() {
    let source = r#"
        axiom A(1);
        match A(s) then F(s) [ Turn(25) A(s / 2) ] [ Turn(-25) A(s / 2) ];
    "#;

    assert_eq!(
        run(source, 1),
        "F(1) [ Turn(25) A(0.5) ] [ Turn(-25) A(0.5) ]"
    );
}

#[test]
fn applies_left_and_right_context() {
    let source = r#"
        axiom a b c b;
        match b left c then y;
        match b then x;
    "#;

    let grammar = Grammar::parse(source).unwrap();
    let program = CpuBackend::new()
        .ambiguous_rules(AmbiguousRulePolicy::First)
        .compile(&grammar)
        .unwrap();
    assert_eq!(program.run(1).unwrap().to_string(), "a x c y");
}

#[test]
fn supports_only_context_filter() {
    let source = r#"
        only F;
        axiom F H F;
        match H left F right F then X;
    "#;

    assert_eq!(run(source, 1), "F X F");
}

#[test]
fn supports_empty_successor() {
    let source = r#"
        axiom A B;
        match A then nothing;
    "#;

    assert_eq!(run(source, 1), "B");
}

#[test]
fn enforces_item_limit() {
    let grammar = CompiledGrammar::parse(
        r#"
            axiom F;
            match F then F F F;
        "#,
    )
    .unwrap();

    let error = calculate(CalculationRequest {
        grammar,
        iterations: 3,
        backend: BackendChoice::Cpu,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits {
            max_items: 10,
            ..CalculationLimits::default()
        },
    })
    .unwrap_err();

    assert!(matches!(
        error,
        CalculationError::Execution(CpuExecutionError { .. })
    ));
}

#[cfg(any(not(feature = "cuda"), target_arch = "wasm32"))]
#[test]
fn explicit_cuda_reports_unavailable_without_cuda_feature() {
    let grammar = CompiledGrammar::parse(
        r#"
            axiom F;
            match F then F F;
        "#,
    )
    .unwrap();

    let error = calculate(CalculationRequest {
        grammar,
        iterations: 1,
        backend: BackendChoice::Cuda,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits::default(),
    })
    .unwrap_err();

    assert!(matches!(
        error,
        CalculationError::BackendUnavailable {
            backend: BackendChoice::Cuda,
            ..
        }
    ));
}
