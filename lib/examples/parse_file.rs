//! Read, parse, and derive an L-system stored in a file.
//!
//! Pass a different file as the first argument, or omit it to use the bundled
//! Fibonacci example.

use std::{env, error::Error, fs, path::PathBuf};

use braken::{BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/files/fibonacci.lsys")
    });
    let source = fs::read_to_string(&path)?;
    let grammar = CompiledGrammar::parse(&source)?;

    let result = calculate(CalculationRequest {
        grammar,
        iterations: 8,
        backend: BackendChoice::Auto,
        seed: 0,
        semantics: Default::default(),
        limits: CalculationLimits::default(),
    })?;

    println!("{} via {:?}", path.display(), result.backend_used);
    println!("{}", result.generation);
    Ok(())
}
