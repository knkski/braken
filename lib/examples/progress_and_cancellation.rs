//! Observe structured progress and cooperatively cancel a calculation.

use std::error::Error;

use braken::{
    BackendChoice, CalculationError, CalculationLimits, CalculationRequest, CancellationToken,
    CompiledGrammar, calculate_with_control,
};

fn main() -> Result<(), Box<dyn Error>> {
    let grammar = CompiledGrammar::parse("axiom f; match f then f f;")?;
    let cancellation = CancellationToken::new();
    let cancel_from_callback = cancellation.clone();
    let mut last_reported_iteration = None;

    let outcome = calculate_with_control(
        CalculationRequest {
            grammar,
            iterations: 30,
            backend: BackendChoice::Cpu,
            seed: 42,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        },
        &cancellation,
        |progress| {
            if last_reported_iteration != Some(progress.completed_iterations) {
                eprintln!(
                    "iteration {}/{}: {} items ({:?})",
                    progress.completed_iterations,
                    progress.total_iterations,
                    progress.items,
                    progress.phase,
                );
                last_reported_iteration = Some(progress.completed_iterations);
            }
            if progress.completed_iterations >= 5 {
                cancel_from_callback.cancel();
            }
            !cancellation.is_cancelled()
        },
    );

    match outcome {
        Err(CalculationError::Cancelled) => println!("cancelled after five iterations"),
        Ok(result) => println!("completed with {} modules", result.stats.modules),
        Err(error) => return Err(error.into()),
    }

    Ok(())
}
