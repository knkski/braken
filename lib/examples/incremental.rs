//! Compile once, then retain and advance an execution state one step at a time.

use std::error::Error;

use braken::Grammar;

fn main() -> Result<(), Box<dyn Error>> {
    let grammar = Grammar::parse(
        r#"
            axiom b;
            match a then a b;
            match b then a;
        "#,
    )?;
    let program = grammar.compile_cpu()?;
    let mut state = program.start();

    println!("step {}: {}", state.generation_index(), state.generation());
    for _ in 0..8 {
        let stats = state.step()?;
        println!(
            "step {} ({} modules): {}",
            stats.generation,
            stats.output_modules,
            state.generation()
        );
    }

    Ok(())
}
