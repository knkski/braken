//! Grammar parser and backend-selectable derivation engine for L-systems.
//!
//! This crate turns a validated grammar into a backend-neutral [`Generation`].
//! Converting that generation into drawing primitives and rasterizing those
//! primitives are downstream visualization and display stages; selecting CUDA
//! or WGPU here does not imply that either later stage uses the same backend.
//!
//! A [`CompiledGrammar`] also owns validated [`DerivationIr`]. The [`ir`]
//! module exposes versioned interchange, invariant checking, deterministic
//! disassembly, static backend diagnostics, and CPU-reference rewrite tracing;
//! device-specific packed layouts remain private.

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
pub mod cuda_backend;
pub mod execution;
pub mod grammar;
pub mod ir;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
pub use cuda_backend::{CudaBackend, CudaLimitEstimate, CudaProgram, CudaState};
pub use execution::{
    BackendChoice, CalculationError, CalculationLimits, CalculationPhase, CalculationProgress,
    CalculationRequest, CalculationResult, CancellationToken, CompiledGrammar, DerivationSemantics,
    ExecutionStats, calculate, calculate_rewrite_lineage_with_control, calculate_with_control,
    calculate_with_progress,
};
pub use grammar::{
    AmbiguousRulePolicy, BinaryOp, Binding, ContextFilter, CpuBackend, CpuProgram, CpuState,
    CpuStepPhase, CpuStepProgress, Document, ExecutionError as CpuExecutionError,
    ExecutionErrorKind as CpuExecutionErrorKind, ExecutionLimits, Expr, FloatWidth, Generation,
    GenerationItem, Grammar, GrammarError, Identifier, Item, Module, ModuleExpr, ModulePattern,
    PatternArgument, PatternItem, PatternWord, Production, RewriteLineage, StepStats, SyntaxError,
    UnaryOp, ValidationError, ValidationErrors, Value, Word, WordItem,
};
pub use ir::{
    BindingSlot, DerivationIr, GlobalId, IR_FORMAT_NAME, IR_FORMAT_VERSION, IR_SEMANTICS_VERSION,
    IrBackendProfile, IrBinaryOp, IrBuilder, IrCompatibilityReport, IrContextFilter, IrDiagnostic,
    IrDiagnosticSeverity, IrDocument, IrExpr, IrGlobal, IrInstruction, IrJsonError, IrModuleExpr,
    IrModulePattern, IrPatternArgument, IrPatternItem, IrPatternWord, IrProduction,
    IrProductionGroup, IrProgram, IrProgramOwner, IrRequirements, IrRewriteTrace, IrSymbol,
    IrUnaryOp, IrValidationErrors, IrView, IrWord, IrWordItem, ProductionId, ProgramId, SymbolId,
};
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{WgpuBackend, WgpuBackendOptions, WgpuError, WgpuProgram, WgpuState};

/// Generates a seed from the operating system or browser entropy source.
pub fn generate_seed() -> Result<u64, getrandom::Error> {
    getrandom::u64()
}
