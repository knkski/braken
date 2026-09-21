use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};

fn main() {
    let grammar = CompiledGrammar::parse(
        r#"
            axiom b;
            match a then a b;
            match b then a;
        "#,
    )
    .unwrap();

    for iterations in 0..10 {
        let result = calculate(CalculationRequest {
            grammar: grammar.clone(),
            iterations,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        })
        .unwrap();

        println!("Step #{iterations}: {}", result.generation);
    }
}
