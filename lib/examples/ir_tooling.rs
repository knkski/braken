use braken::{
    AmbiguousRulePolicy, BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar,
    DerivationSemantics, FloatWidth, IrBackendProfile, IrDocument, calculate,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grammar = CompiledGrammar::parse("axiom A(1); match A(x) when x < 4 then A(x + 1) B;")?;

    println!("{}", grammar.ir().disassemble());
    for float_width in [FloatWidth::F32, FloatWidth::F64] {
        let semantics = DerivationSemantics {
            float_width,
            ambiguous_rules: AmbiguousRulePolicy::Uniform,
        };
        for backend in [
            IrBackendProfile::Cpu,
            IrBackendProfile::CurrentCuda,
            IrBackendProfile::CurrentWgpu,
        ] {
            let report = grammar
                .ir()
                .compatibility_with_semantics(backend, semantics);
            println!("{backend}/{float_width}: compatible={}", report.compatible);
        }
    }

    let json = grammar.to_ir_document().to_json_pretty()?;
    let imported = CompiledGrammar::from_ir(IrDocument::from_json(&json)?)?;
    let result = calculate(CalculationRequest {
        grammar: imported,
        iterations: 3,
        backend: BackendChoice::Cpu,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits::default(),
    })?;
    println!("generation: {}", result.generation);
    Ok(())
}
