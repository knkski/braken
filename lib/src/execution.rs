//! Backend-independent requests for deriving L-system generations.
//!
//! [`BackendChoice`] selects only the derivation stage. Every successful backend
//! returns the same backend-neutral [`Generation`] representation; visualization
//! and display select their own implementations downstream.
//!
//! On native builds, [`BackendChoice::Auto`] preflights CUDA, then a hardware
//! WGPU adapter, and finally CPU. It may fall through when a candidate is
//! unavailable, cannot compile the grammar, or cannot satisfy preflight resource
//! requirements. Once a GPU backend starts executing a request, an error is
//! returned instead of silently repeating the work on another backend. Explicit
//! backend choices never fall back.
//!
//! Caller limits are semantic output limits, not GPU dispatch or batching sizes.
//! Their defaults impose no policy cap; allocation, address-space, and device
//! limits can still produce a typed resource error. Cancellation is cooperative:
//! CPU work checks within bounded chunks, while GPU work checks between bounded
//! submissions and synchronization points.
//!
//! [`DerivationSemantics`] makes floating-point width and ambiguous-rule
//! handling explicit. Automatic selection preflights those semantics along
//! with grammar requirements: CPU implements both widths, CUDA currently
//! implements f64, and WGPU currently implements f32.

use std::error::Error;
use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use web_time::Instant;

use crate::grammar::{
    AmbiguousRulePolicy, CpuBackend, ExecutionError as CpuExecutionError, ExecutionLimits,
    FloatWidth, Generation, Grammar, GrammarError,
};
use crate::ir::{DerivationIr, IrDocument, IrValidationErrors, IrView};

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
static NATIVE_WGPU_BACKEND: OnceLock<Mutex<Option<crate::wgpu_backend::WgpuBackend>>> =
    OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
/// Selects the backend used to derive a [`Generation`].
///
/// This choice does not select a visualizer or a GUI renderer.
pub enum BackendChoice {
    /// Prefer usable native CUDA, then hardware WGPU, then CPU.
    ///
    /// Browser callers that need asynchronous WebGPU should use
    /// `wgpu_backend::WgpuBackend` directly from a worker instead of this
    /// synchronous dispatcher.
    Auto,
    /// Use the full-grammar CPU implementation.
    Cpu,
    /// Require the native CUDA implementation and its supported grammar subset.
    Cuda,
    /// Require WGPU compute and its supported grammar subset.
    Wgpu,
    /// Reserved for externally registered backends. The built-in dispatcher
    /// currently reports [`CalculationError::BackendUnsupported`].
    Named(String),
}

/// A parsed and semantically validated grammar ready for a calculation request.
#[derive(Debug, Clone)]
pub struct CompiledGrammar {
    grammar: Grammar,
    ir: DerivationIr,
}

impl CompiledGrammar {
    /// Parses and validates grammar source.
    pub fn parse(source: &str) -> Result<Self, GrammarError> {
        let grammar = Grammar::parse(source)?;
        let ir = DerivationIr::from_grammar_source(&grammar, Some(source));
        Ok(Self { grammar, ir })
    }

    /// Returns the validated grammar used by backend compilers.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// Returns a read-only view of the validated backend-neutral grammar IR.
    pub fn ir(&self) -> IrView<'_> {
        self.ir.view()
    }

    /// Returns the shareable validated IR owned by this compiled grammar.
    pub fn derivation_ir(&self) -> &DerivationIr {
        &self.ir
    }

    /// Validates an interchange document and makes it executable by every
    /// backend whose preflight accepts its requirements.
    pub fn from_ir(document: IrDocument) -> Result<Self, IrValidationErrors> {
        let ir = document.validate()?;
        let grammar = ir.to_grammar()?;
        Ok(Self { grammar, ir })
    }

    /// Clones the versioned interchange representation for serialization or editing.
    pub fn to_ir_document(&self) -> IrDocument {
        self.ir.document().clone()
    }
}

impl From<Grammar> for CompiledGrammar {
    fn from(grammar: Grammar) -> Self {
        let ir = DerivationIr::from_grammar_source(&grammar, None);
        Self { grammar, ir }
    }
}

/// Numeric and rule-selection semantics required of a derivation backend.
///
/// Backend parity is defined within one of these profiles. A backend that
/// cannot implement the selected width or ambiguity policy must reject the
/// request during preflight instead of silently changing its meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DerivationSemantics {
    /// Floating-point width used for every numeric operation and stored argument.
    pub float_width: FloatWidth,
    /// Resolution policy for multiple applicable unweighted productions.
    pub ambiguous_rules: AmbiguousRulePolicy,
}

impl Default for DerivationSemantics {
    fn default() -> Self {
        Self {
            float_width: FloatWidth::F64,
            ambiguous_rules: AmbiguousRulePolicy::Uniform,
        }
    }
}

#[derive(Debug, Clone)]
/// Complete input to one derivation request.
pub struct CalculationRequest {
    /// Parsed and validated grammar to derive.
    pub grammar: CompiledGrammar,
    /// Number of production rewrites after the axiom.
    pub iterations: usize,
    /// Derivation backend policy for this request.
    pub backend: BackendChoice,
    /// Stable key for stochastic production choices.
    pub seed: u64,
    /// Numeric width and ambiguous-rule selection semantics.
    pub semantics: DerivationSemantics,
    /// Optional semantic limits applied to every generated result.
    pub limits: CalculationLimits,
}

/// Caller-selected semantic limits for a derived generation.
///
/// These values are not scheduling quanta and must not be inferred from GPU
/// buffer or launch sizes. [`Default`] and [`Self::unbounded_production`] impose
/// no caller policy limits, although real resource constraints still apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalculationLimits {
    /// Maximum number of modules in a generation.
    pub max_modules: usize,
    /// Maximum number of modules plus structural branch items.
    pub max_items: usize,
    /// Maximum structural branch nesting depth.
    pub max_branch_depth: usize,
}

impl Default for CalculationLimits {
    fn default() -> Self {
        Self {
            max_modules: usize::MAX,
            max_items: usize::MAX,
            max_branch_depth: usize::MAX,
        }
    }
}

impl CalculationLimits {
    /// Uses no caller-selected semantic generation limits.
    ///
    /// Allocator, address-space, and backend device constraints still apply.
    pub fn unbounded_production() -> Self {
        Self::default()
    }
}

/// A cheap, cloneable, one-shot cancellation signal shared by coordinators and workers.
///
/// Clones observe the same monotonic flag. Cancellation cannot be reset and does
/// not forcibly interrupt an in-flight device kernel; backends observe it at
/// safe checkpoints and then return [`CalculationError::Cancelled`].
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Creates a token in the running state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks this token and all of its clones as cancelled.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Reports whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl From<CalculationLimits> for ExecutionLimits {
    fn from(limits: CalculationLimits) -> Self {
        Self {
            max_modules: limits.max_modules,
            max_items: limits.max_items,
            max_branch_depth: limits.max_branch_depth,
        }
    }
}

#[derive(Debug, Clone)]
/// A completed derivation and the backend that actually produced it.
pub struct CalculationResult {
    /// Exact backend-neutral generation.
    pub generation: Generation,
    /// Concrete derivation backend, including the backend selected by [`BackendChoice::Auto`].
    pub backend_used: BackendChoice,
    /// Final exact counts and total derivation time.
    pub stats: ExecutionStats,
}

/// Summary of a completed derivation.
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Number of requested and completed rewrites.
    pub iterations: usize,
    /// Exact module count in the final generation.
    pub modules: usize,
    /// Exact module-plus-branch-item count in the final generation.
    pub items: usize,
    /// Wall-clock time spent deriving and materializing the result.
    pub elapsed: Duration,
}

/// Backend-neutral phase of a derivation progress update.
///
/// Backends may omit phases that do not apply to their algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalculationPhase {
    /// Initial validation, compilation, or device setup.
    Preparing,
    /// Counting and validating the input generation.
    InspectingInput,
    /// Building context or dispatch indexes.
    Indexing,
    /// Selecting productions for the current input.
    SelectingProductions,
    /// Materializing the selected successors.
    Rewriting,
    /// Counting and validating a completed output.
    ValidatingOutput,
    /// Moving a completed result between device and host memory.
    Transferring,
    /// The requested derivation is complete.
    Complete,
}

/// A point-in-time, backend-neutral derivation progress update.
///
/// `completed_iterations` is monotonic. `phase_completed` is local to `phase`,
/// and `phase_total` is absent when a backend cannot know the total cheaply.
/// While a flat GPU generation remains device-resident, `modules` and `items`
/// can both report its token count; the final [`ExecutionStats`] are exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalculationProgress {
    /// Work currently being performed.
    pub phase: CalculationPhase,
    /// Completed work units within the current phase.
    pub phase_completed: usize,
    /// Total work units within the current phase, when known.
    pub phase_total: Option<usize>,
    /// Fully completed production rewrites.
    pub completed_iterations: usize,
    /// Total requested production rewrites.
    pub total_iterations: usize,
    /// Best currently available module count.
    pub modules: usize,
    /// Best currently available item count.
    pub items: usize,
    /// Wall-clock time since this calculation started.
    pub elapsed: Duration,
}

/// Typed failure from parsing, selecting, or running a derivation backend.
#[derive(Debug, Clone)]
pub enum CalculationError {
    /// Source parsing or semantic validation failed.
    Grammar(GrammarError),
    /// The CPU execution engine returned a typed evaluation or resource error.
    Execution(CpuExecutionError),
    /// The requested backend was compiled in but no usable runtime/device was available.
    BackendUnavailable {
        backend: BackendChoice,
        reason: String,
    },
    /// The built-in dispatcher does not implement the requested backend identifier.
    BackendUnsupported(BackendChoice),
    /// The backend is available but cannot preserve the grammar's semantics.
    UnsupportedGrammar {
        backend: BackendChoice,
        reason: String,
    },
    /// A selected backend failed during setup or execution.
    BackendFailed {
        backend: BackendChoice,
        reason: String,
    },
    /// A caller limit or real host/device resource constraint was exceeded.
    ResourceExhausted {
        backend: BackendChoice,
        resource: &'static str,
        reason: String,
    },
    /// The cancellation token or progress callback stopped the request.
    Cancelled,
}

impl fmt::Display for CalculationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grammar(error) => write!(f, "{error}"),
            Self::Execution(error) => write!(f, "{error}"),
            Self::BackendUnavailable { backend, reason } => {
                write!(f, "backend is not available: {backend:?}: {reason}")
            }
            Self::BackendUnsupported(backend) => write!(f, "backend is not available: {backend:?}"),
            Self::UnsupportedGrammar { backend, reason } => {
                write!(
                    f,
                    "backend {backend:?} does not support this grammar: {reason}"
                )
            }
            Self::BackendFailed { backend, reason } => {
                write!(f, "backend {backend:?} failed while executing: {reason}")
            }
            Self::ResourceExhausted {
                backend,
                resource,
                reason,
            } => write!(f, "backend {backend:?} exhausted {resource}: {reason}"),
            Self::Cancelled => f.write_str("calculation cancelled"),
        }
    }
}

impl Error for CalculationError {}

impl From<GrammarError> for CalculationError {
    fn from(error: GrammarError) -> Self {
        Self::Grammar(error)
    }
}

impl From<CpuExecutionError> for CalculationError {
    fn from(error: CpuExecutionError) -> Self {
        if error.is_cancelled() {
            Self::Cancelled
        } else {
            Self::Execution(error)
        }
    }
}

/// Derives a generation without progress callbacks or external cancellation.
pub fn calculate(request: CalculationRequest) -> Result<CalculationResult, CalculationError> {
    calculate_with_control(request, &CancellationToken::new(), |_| true)
}

/// Derives a generation while reporting completed iterations.
///
/// For compatibility, the callback is invoked at most once per completed
/// iteration even though [`calculate_with_control`] can report intra-iteration
/// phases. Returning `false` requests cooperative cancellation.
pub fn calculate_with_progress(
    request: CalculationRequest,
    mut on_progress: impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    let mut last_completed = None;
    calculate_with_control(request, &CancellationToken::new(), |progress| {
        if last_completed == Some(progress.completed_iterations) {
            true
        } else {
            last_completed = Some(progress.completed_iterations);
            on_progress(progress)
        }
    })
}

/// Derives a generation with structured progress and a cancellation signal that
/// may be triggered from another thread.
///
/// Returning `false` from `on_progress` has the same effect as cancelling the
/// token. CPU work observes cancellation within bounded chunks. GPU backends
/// observe it at submission and synchronization boundaries; device work already
/// submitted to a driver finishes normally.
pub fn calculate_with_control(
    request: CalculationRequest,
    cancellation: &CancellationToken,
    mut on_progress: impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    if cancellation.is_cancelled() {
        return Err(CalculationError::Cancelled);
    }
    match request.backend {
        BackendChoice::Auto => calculate_auto(request, cancellation, &mut on_progress),
        BackendChoice::Cpu => calculate_cpu(request, cancellation, &mut on_progress),
        BackendChoice::Cuda => calculate_cuda(request, cancellation, &mut on_progress),
        BackendChoice::Wgpu => calculate_wgpu(request, cancellation, &mut on_progress),
        BackendChoice::Named(_) => Err(CalculationError::BackendUnsupported(request.backend)),
    }
}

/// Derives through `request.iterations` and returns exact selected-successor
/// lineage for only the final rewrite.
///
/// This opt-in operation is intended for adjacent-generation consumers. It
/// preserves the ordinary backend-selection and no-runtime-fallback policy,
/// while allowing WGPU to download the actual selected decisions. The request
/// must contain at least one iteration.
pub fn calculate_rewrite_lineage_with_control(
    request: CalculationRequest,
    cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    if request.iterations == 0 {
        return Err(CalculationError::BackendFailed {
            backend: request.backend,
            reason: String::from("rewrite lineage requires at least one iteration"),
        });
    }
    if cancellation.is_cancelled() {
        return Err(CalculationError::Cancelled);
    }
    match request.backend {
        BackendChoice::Auto => calculate_auto_lineage(request, cancellation),
        BackendChoice::Cpu => calculate_cpu_lineage(request, cancellation),
        BackendChoice::Cuda => calculate_cuda_lineage(request, cancellation),
        BackendChoice::Wgpu => calculate_wgpu_lineage(request, cancellation),
        BackendChoice::Named(_) => Err(CalculationError::BackendUnsupported(request.backend)),
    }
}

fn calculate_auto_lineage(
    request: CalculationRequest,
    cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    {
        let candidate = crate::cuda_backend::CudaBackend::new()
            .float_width(request.semantics.float_width)
            .ambiguous_rules(request.semantics.ambiguous_rules)
            .limits(request.limits);
        match candidate.compile_ir(request.grammar.derivation_ir()) {
            Ok(_) => match candidate.probe() {
                Ok(()) => {
                    return calculate_cuda_lineage(
                        CalculationRequest {
                            backend: BackendChoice::Cuda,
                            ..request
                        },
                        cancellation,
                    );
                }
                Err(error) if error.is_cuda_fallback_error() => {}
                Err(error) => return Err(error),
            },
            Err(error) if error.is_cuda_fallback_error() => {}
            Err(error) => return Err(error),
        }
    }
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    {
        match native_wgpu_backend() {
            Ok(backend) => match backend
                .float_width(request.semantics.float_width)
                .ambiguous_rules(request.semantics.ambiguous_rules)
                .limits(request.limits)
                .compile_ir(request.grammar.derivation_ir())
            {
                Ok(program) => {
                    return calculate_wgpu_lineage_with_program(
                        CalculationRequest {
                            backend: BackendChoice::Wgpu,
                            ..request
                        },
                        program,
                        cancellation,
                    );
                }
                Err(error) if is_wgpu_preflight_fallback(&error) => {}
                Err(error) => return Err(map_wgpu_error(error)),
            },
            Err(error) if is_wgpu_preflight_fallback(&error) => {}
            Err(error) => return Err(map_wgpu_error(error)),
        }
    }
    calculate_cpu_lineage(
        CalculationRequest {
            backend: BackendChoice::Cpu,
            ..request
        },
        cancellation,
    )
}

fn calculate_cpu_lineage(
    request: CalculationRequest,
    cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    let program = CpuBackend::new()
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits.into())
        .compile_ir(request.grammar.derivation_ir())?;
    let mut state = program.start_with_seed(request.seed);
    if request.iterations > 1 {
        for _ in 0..request.iterations - 1 {
            state.step_with_control(|| cancellation.is_cancelled(), |_| {})?;
        }
    }
    let (_, lineage) =
        state.step_with_lineage_and_control(|| cancellation.is_cancelled(), |_| {})?;
    Ok((lineage, BackendChoice::Cpu))
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn calculate_wgpu_lineage(
    request: CalculationRequest,
    cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    let backend = native_wgpu_backend()
        .map_err(map_wgpu_error)?
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits);
    let program = backend
        .compile_ir(request.grammar.derivation_ir())
        .map_err(map_wgpu_error)?;
    calculate_wgpu_lineage_with_program(request, program, cancellation)
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn calculate_wgpu_lineage_with_program(
    request: CalculationRequest,
    program: crate::wgpu_backend::WgpuProgram,
    cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    let result = pollster::block_on(async {
        let mut state = program.start_with_seed(request.seed)?;
        for _ in 0..request.iterations - 1 {
            state
                .step_with_cancel(&mut || cancellation.is_cancelled())
                .await?;
        }
        state
            .step_with_lineage_and_cancel(&mut || cancellation.is_cancelled())
            .await
    });
    match result {
        Ok(lineage) => Ok((lineage, BackendChoice::Wgpu)),
        Err(error) => {
            if matches!(
                error,
                crate::wgpu_backend::WgpuError::DeviceLost(_)
                    | crate::wgpu_backend::WgpuError::OutOfMemory(_)
                    | crate::wgpu_backend::WgpuError::Validation(_)
                    | crate::wgpu_backend::WgpuError::Internal(_)
                    | crate::wgpu_backend::WgpuError::MapFailed(_)
            ) {
                invalidate_native_wgpu_backend();
            }
            Err(map_wgpu_error(error))
        }
    }
}

#[cfg(any(not(feature = "wgpu"), target_arch = "wasm32"))]
fn calculate_wgpu_lineage(
    request: CalculationRequest,
    _cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    Err(CalculationError::BackendUnavailable {
        backend: request.backend,
        reason: String::from("WGPU rewrite lineage is unavailable for this synchronous target"),
    })
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
fn calculate_cuda_lineage(
    request: CalculationRequest,
    cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    let backend = crate::cuda_backend::CudaBackend::new()
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits);
    let program = backend.compile_ir(request.grammar.derivation_ir())?;
    let mut state = program.start_with_seed(request.seed);
    if request.iterations > 1 {
        state.advance_with_control(request.iterations - 1, cancellation)?;
    }
    let trace_program = CpuBackend::new()
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits.into())
        .compile_ir(request.grammar.derivation_ir())?;
    let (expected, _, lineage) = trace_program.trace_rewrite_with_control(
        state.generation(),
        state.generation_index(),
        request.seed,
        || cancellation.is_cancelled(),
        |_| {},
    )?;
    state.step_with_control(cancellation)?;
    if state.generation() != &expected {
        return Err(CalculationError::BackendFailed {
            backend: BackendChoice::Cuda,
            reason: String::from("CUDA rewrite disagreed with CPU lineage trace"),
        });
    }
    Ok((lineage, BackendChoice::Cuda))
}

#[cfg(any(not(feature = "cuda"), target_arch = "wasm32"))]
fn calculate_cuda_lineage(
    request: CalculationRequest,
    _cancellation: &CancellationToken,
) -> Result<(crate::grammar::RewriteLineage, BackendChoice), CalculationError> {
    Err(CalculationError::BackendUnavailable {
        backend: request.backend,
        reason: String::from("CUDA rewrite lineage is unavailable in this build"),
    })
}

fn calculate_auto(
    request: CalculationRequest,
    cancellation: &CancellationToken,
    on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    {
        let cuda_request = CalculationRequest {
            backend: BackendChoice::Cuda,
            ..request.clone()
        };
        let candidate = crate::cuda_backend::CudaBackend::new()
            .float_width(request.semantics.float_width)
            .ambiguous_rules(request.semantics.ambiguous_rules)
            .limits(request.limits);
        match candidate.compile_ir(request.grammar.derivation_ir()) {
            Ok(_) => match candidate.probe() {
                Ok(()) => return calculate_cuda(cuda_request, cancellation, on_progress),
                Err(error) if error.is_cuda_fallback_error() => {}
                Err(error) => return Err(error),
            },
            Err(error) if error.is_cuda_fallback_error() => {}
            Err(error) => return Err(error),
        }
    }

    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    {
        match native_wgpu_backend() {
            Ok(backend) => match backend
                .float_width(request.semantics.float_width)
                .ambiguous_rules(request.semantics.ambiguous_rules)
                .limits(request.limits)
                .compile_ir(request.grammar.derivation_ir())
            {
                Ok(program) => {
                    let wgpu_request = CalculationRequest {
                        backend: BackendChoice::Wgpu,
                        ..request.clone()
                    };
                    return calculate_wgpu_with_program(
                        wgpu_request,
                        program,
                        cancellation,
                        on_progress,
                    );
                }
                Err(error) if is_wgpu_preflight_fallback(&error) => {}
                Err(error) => return Err(map_wgpu_error(error)),
            },
            Err(error) if is_wgpu_preflight_fallback(&error) => {}
            Err(error) => return Err(map_wgpu_error(error)),
        }
    }

    calculate_cpu(
        CalculationRequest {
            backend: BackendChoice::Cpu,
            ..request
        },
        cancellation,
        on_progress,
    )
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn calculate_wgpu(
    request: CalculationRequest,
    cancellation: &CancellationToken,
    on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    let backend = native_wgpu_backend()
        .map_err(map_wgpu_error)?
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits);
    let program = backend
        .compile_ir(request.grammar.derivation_ir())
        .map_err(map_wgpu_error)?;
    calculate_wgpu_with_program(request, program, cancellation, on_progress)
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn calculate_wgpu_with_program(
    request: CalculationRequest,
    program: crate::wgpu_backend::WgpuProgram,
    cancellation: &CancellationToken,
    on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    let start = Instant::now();
    let axiom = program.axiom().map_err(map_wgpu_error)?;
    let mut current_modules = axiom.module_count();
    let mut current_items = axiom.item_count();
    if cancellation.is_cancelled()
        || !on_progress(CalculationProgress {
            phase: CalculationPhase::Preparing,
            phase_completed: 0,
            phase_total: None,
            completed_iterations: 0,
            total_iterations: request.iterations,
            modules: current_modules,
            items: current_items,
            elapsed: start.elapsed(),
        })
    {
        return Err(CalculationError::Cancelled);
    }

    let result = pollster::block_on(async {
        let mut state = program.start_with_seed(request.seed)?;
        for completed_iterations in 1..=request.iterations {
            if cancellation.is_cancelled()
                || !on_progress(CalculationProgress {
                    phase: CalculationPhase::SelectingProductions,
                    phase_completed: completed_iterations - 1,
                    phase_total: Some(request.iterations),
                    completed_iterations: completed_iterations - 1,
                    total_iterations: request.iterations,
                    modules: current_modules,
                    items: current_items,
                    elapsed: start.elapsed(),
                })
            {
                return Err(crate::wgpu_backend::WgpuError::Cancelled);
            }
            state
                .step_with_cancel(&mut || cancellation.is_cancelled())
                .await?;
            // Exact module/item counts require decoding. Keep the generation
            // device-resident and report its flat token count during compute.
            current_modules = state.token_count();
            current_items = state.token_count();
            if cancellation.is_cancelled()
                || !on_progress(CalculationProgress {
                    phase: CalculationPhase::Rewriting,
                    phase_completed: completed_iterations,
                    phase_total: Some(request.iterations),
                    completed_iterations,
                    total_iterations: request.iterations,
                    modules: current_modules,
                    items: current_items,
                    elapsed: start.elapsed(),
                })
            {
                return Err(crate::wgpu_backend::WgpuError::Cancelled);
            }
        }
        state
            .into_generation_with_cancel(&mut || cancellation.is_cancelled())
            .await
    });
    let generation = match result {
        Ok(generation) => generation,
        Err(error) => {
            if matches!(
                error,
                crate::wgpu_backend::WgpuError::DeviceLost(_)
                    | crate::wgpu_backend::WgpuError::OutOfMemory(_)
                    | crate::wgpu_backend::WgpuError::Validation(_)
                    | crate::wgpu_backend::WgpuError::Internal(_)
                    | crate::wgpu_backend::WgpuError::MapFailed(_)
            ) {
                invalidate_native_wgpu_backend();
            }
            return Err(map_wgpu_error(error));
        }
    };
    current_modules = generation.module_count();
    current_items = generation.item_count();
    if cancellation.is_cancelled()
        || !on_progress(CalculationProgress {
            phase: CalculationPhase::Complete,
            phase_completed: request.iterations,
            phase_total: Some(request.iterations),
            completed_iterations: request.iterations,
            total_iterations: request.iterations,
            modules: current_modules,
            items: current_items,
            elapsed: start.elapsed(),
        })
    {
        return Err(CalculationError::Cancelled);
    }

    Ok(CalculationResult {
        backend_used: BackendChoice::Wgpu,
        stats: ExecutionStats {
            iterations: request.iterations,
            modules: current_modules,
            items: current_items,
            elapsed: start.elapsed(),
        },
        generation,
    })
}

#[cfg(any(not(feature = "wgpu"), target_arch = "wasm32"))]
fn calculate_wgpu(
    request: CalculationRequest,
    _cancellation: &CancellationToken,
    _on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    Err(CalculationError::BackendUnavailable {
        backend: request.backend,
        reason: if cfg!(target_arch = "wasm32") {
            "the synchronous calculation API cannot await WebGPU in a browser; use WgpuBackend's async API from a Web Worker"
                .to_string()
        } else {
            "the braken/wgpu feature is disabled".to_string()
        },
    })
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn native_wgpu_backend() -> Result<crate::wgpu_backend::WgpuBackend, crate::wgpu_backend::WgpuError>
{
    let cache = NATIVE_WGPU_BACKEND.get_or_init(|| Mutex::new(None));
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(backend) = cache.as_ref() {
        return Ok(backend.clone());
    }
    let backend = pollster::block_on(crate::wgpu_backend::WgpuBackend::request())?;
    *cache = Some(backend.clone());
    Ok(backend)
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn invalidate_native_wgpu_backend() {
    if let Some(cache) = NATIVE_WGPU_BACKEND.get() {
        *cache.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn is_wgpu_preflight_fallback(error: &crate::wgpu_backend::WgpuError) -> bool {
    matches!(
        error,
        crate::wgpu_backend::WgpuError::AdapterUnavailable(_)
            | crate::wgpu_backend::WgpuError::SoftwareAdapter { .. }
            | crate::wgpu_backend::WgpuError::RequestDevice(_)
            | crate::wgpu_backend::WgpuError::UnsupportedGrammar(_)
            | crate::wgpu_backend::WgpuError::ResourceExhausted { .. }
            | crate::wgpu_backend::WgpuError::DeviceLost(_)
            | crate::wgpu_backend::WgpuError::OutOfMemory(_)
            | crate::wgpu_backend::WgpuError::Validation(_)
            | crate::wgpu_backend::WgpuError::Internal(_)
            | crate::wgpu_backend::WgpuError::MapFailed(_)
    )
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn map_wgpu_error(error: crate::wgpu_backend::WgpuError) -> CalculationError {
    use crate::wgpu_backend::WgpuError;
    match error {
        WgpuError::AdapterUnavailable(reason) => CalculationError::BackendUnavailable {
            backend: BackendChoice::Wgpu,
            reason,
        },
        WgpuError::SoftwareAdapter { name } => CalculationError::BackendUnavailable {
            backend: BackendChoice::Wgpu,
            reason: format!("adapter `{name}` is a CPU/software adapter"),
        },
        WgpuError::RequestDevice(reason) => CalculationError::BackendUnavailable {
            backend: BackendChoice::Wgpu,
            reason,
        },
        WgpuError::UnsupportedGrammar(reason) => CalculationError::UnsupportedGrammar {
            backend: BackendChoice::Wgpu,
            reason,
        },
        WgpuError::AmbiguousRules { symbol, count } => CalculationError::BackendFailed {
            backend: BackendChoice::Wgpu,
            reason: format!("module `{symbol}` has {count} applicable unweighted productions"),
        },
        WgpuError::ResourceExhausted {
            resource,
            requested,
            limit,
        } => CalculationError::ResourceExhausted {
            backend: BackendChoice::Wgpu,
            resource,
            reason: format!("requested {requested}, device limit {limit}"),
        },
        WgpuError::LimitExceeded {
            resource,
            actual,
            limit,
        } => CalculationError::ResourceExhausted {
            backend: BackendChoice::Wgpu,
            resource,
            reason: format!("generated {actual}, configured limit {limit}"),
        },
        WgpuError::OutOfMemory(reason) => CalculationError::ResourceExhausted {
            backend: BackendChoice::Wgpu,
            resource: "device memory",
            reason,
        },
        WgpuError::DeviceLost(reason)
        | WgpuError::Validation(reason)
        | WgpuError::Internal(reason)
        | WgpuError::MapFailed(reason) => CalculationError::BackendFailed {
            backend: BackendChoice::Wgpu,
            reason,
        },
        WgpuError::Cancelled => CalculationError::Cancelled,
    }
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
fn calculate_cuda(
    request: CalculationRequest,
    cancellation: &CancellationToken,
    on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    let start = Instant::now();
    if cancellation.is_cancelled()
        || !on_progress(CalculationProgress {
            phase: CalculationPhase::Preparing,
            phase_completed: 0,
            phase_total: None,
            completed_iterations: 0,
            total_iterations: request.iterations,
            modules: request.grammar.grammar().axiom.len(),
            items: request.grammar.grammar().axiom.len(),
            elapsed: start.elapsed(),
        })
    {
        return Err(CalculationError::Cancelled);
    }
    let backend = crate::cuda_backend::CudaBackend::new()
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits);
    let program = backend.compile_ir(request.grammar.derivation_ir())?;
    let generation = program.run_with_control(
        request.iterations,
        request.seed,
        Some(cancellation),
        |completed_iterations, modules, items| {
            on_progress(CalculationProgress {
                phase: CalculationPhase::Transferring,
                phase_completed: completed_iterations,
                phase_total: Some(request.iterations),
                completed_iterations,
                total_iterations: request.iterations,
                modules,
                items,
                elapsed: start.elapsed(),
            })
        },
    )?;
    if cancellation.is_cancelled()
        || !on_progress(CalculationProgress {
            phase: CalculationPhase::Complete,
            phase_completed: request.iterations,
            phase_total: Some(request.iterations),
            completed_iterations: request.iterations,
            total_iterations: request.iterations,
            modules: generation.module_count(),
            items: generation.item_count(),
            elapsed: start.elapsed(),
        })
    {
        return Err(CalculationError::Cancelled);
    }

    Ok(CalculationResult {
        backend_used: BackendChoice::Cuda,
        stats: ExecutionStats {
            iterations: request.iterations,
            modules: generation.module_count(),
            items: generation.item_count(),
            elapsed: start.elapsed(),
        },
        generation,
    })
}

#[cfg(any(not(feature = "cuda"), target_arch = "wasm32"))]
fn calculate_cuda(
    request: CalculationRequest,
    _cancellation: &CancellationToken,
    _on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    Err(CalculationError::BackendUnavailable {
        backend: request.backend,
        reason: if cfg!(target_arch = "wasm32") {
            "CUDA is not available for wasm32 builds".to_string()
        } else {
            "the braken/cuda feature is disabled".to_string()
        },
    })
}

impl CalculationError {
    /// Constructs the standard error for grammar semantics unsupported by CUDA.
    pub fn unsupported_cuda_grammar(reason: impl Into<String>) -> Self {
        Self::UnsupportedGrammar {
            backend: BackendChoice::Cuda,
            reason: reason.into(),
        }
    }

    /// Reports whether this error can make CUDA ineligible during automatic
    /// preflight selection.
    ///
    /// Callers must use this only before CUDA execution starts. A runtime error
    /// from a selected request is returned to the caller and must not trigger a
    /// silent replay on another backend.
    pub fn is_cuda_fallback_error(&self) -> bool {
        matches!(
            self,
            Self::BackendUnavailable {
                backend: BackendChoice::Cuda,
                ..
            } | Self::BackendUnsupported(BackendChoice::Cuda)
                | Self::BackendFailed {
                    backend: BackendChoice::Cuda,
                    ..
                }
                | Self::ResourceExhausted {
                    backend: BackendChoice::Cuda,
                    ..
                }
                | Self::UnsupportedGrammar {
                    backend: BackendChoice::Cuda,
                    ..
                }
        )
    }

    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    pub(crate) fn cuda_runtime_failure(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        let lowercase = reason.to_ascii_lowercase();
        if lowercase.contains("out of memory")
            || lowercase.contains("memory allocation")
            || lowercase.contains("cuda_error_out_of_memory")
        {
            Self::ResourceExhausted {
                backend: BackendChoice::Cuda,
                resource: "device memory",
                reason,
            }
        } else {
            Self::BackendFailed {
                backend: BackendChoice::Cuda,
                reason,
            }
        }
    }
}

fn calculate_cpu(
    request: CalculationRequest,
    cancellation: &CancellationToken,
    on_progress: &mut impl FnMut(CalculationProgress) -> bool,
) -> Result<CalculationResult, CalculationError> {
    let start = Instant::now();
    let program = CpuBackend::new()
        .float_width(request.semantics.float_width)
        .ambiguous_rules(request.semantics.ambiguous_rules)
        .limits(request.limits.into())
        .compile_ir(request.grammar.derivation_ir())?;
    let mut state = program.start_with_seed(request.seed);
    let mut current_modules = state.generation().module_count();
    let mut current_items = state.generation().item_count();
    if cancellation.is_cancelled()
        || !on_progress(CalculationProgress {
            phase: CalculationPhase::Preparing,
            phase_completed: 0,
            phase_total: None,
            completed_iterations: 0,
            total_iterations: request.iterations,
            modules: current_modules,
            items: current_items,
            elapsed: start.elapsed(),
        })
    {
        return Err(CalculationError::Cancelled);
    }
    for completed_iterations in 1..=request.iterations {
        let input_modules = current_modules;
        let input_items = current_items;
        let stats = state.step_with_control(
            || cancellation.is_cancelled(),
            |step| {
                let phase = match step.phase {
                    crate::grammar::CpuStepPhase::InspectingInput => {
                        CalculationPhase::InspectingInput
                    }
                    crate::grammar::CpuStepPhase::Indexing => CalculationPhase::Indexing,
                    crate::grammar::CpuStepPhase::SelectingProductions => {
                        CalculationPhase::SelectingProductions
                    }
                    crate::grammar::CpuStepPhase::Rewriting => CalculationPhase::Rewriting,
                    crate::grammar::CpuStepPhase::ValidatingOutput => {
                        CalculationPhase::ValidatingOutput
                    }
                };
                if !on_progress(CalculationProgress {
                    phase,
                    phase_completed: step.completed_items,
                    phase_total: step.total_items,
                    completed_iterations: completed_iterations - 1,
                    total_iterations: request.iterations,
                    modules: input_modules,
                    items: input_items,
                    elapsed: start.elapsed(),
                }) {
                    cancellation.cancel();
                }
            },
        )?;
        current_modules = stats.output_modules;
        current_items = stats.output_items;
        if cancellation.is_cancelled()
            || !on_progress(CalculationProgress {
                phase: if completed_iterations == request.iterations {
                    CalculationPhase::Complete
                } else {
                    CalculationPhase::Preparing
                },
                phase_completed: completed_iterations,
                phase_total: Some(request.iterations),
                completed_iterations,
                total_iterations: request.iterations,
                modules: stats.output_modules,
                items: stats.output_items,
                elapsed: start.elapsed(),
            })
        {
            return Err(CalculationError::Cancelled);
        }
    }
    let generation = state.into_generation();

    Ok(CalculationResult {
        backend_used: BackendChoice::Cpu,
        stats: ExecutionStats {
            iterations: request.iterations,
            modules: generation.module_count(),
            items: generation.item_count(),
            elapsed: start.elapsed(),
        },
        generation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doubling_request(iterations: usize) -> CalculationRequest {
        CalculationRequest {
            grammar: CompiledGrammar::parse("axiom A; match A then A A;").unwrap(),
            iterations,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        }
    }

    #[test]
    fn defaults_have_no_policy_caps() {
        assert_eq!(
            CalculationLimits::default(),
            CalculationLimits {
                max_modules: usize::MAX,
                max_items: usize::MAX,
                max_branch_depth: usize::MAX,
            }
        );
    }

    #[test]
    fn externally_cancelled_request_returns_typed_error() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error =
            calculate_with_control(doubling_request(10), &cancellation, |_| true).unwrap_err();
        assert!(matches!(error, CalculationError::Cancelled));
    }

    #[test]
    fn cancellation_token_is_shared_and_one_way() {
        let first = CancellationToken::new();
        let second = first.clone();
        assert!(!first.is_cancelled());
        assert!(!second.is_cancelled());

        second.cancel();

        assert!(first.is_cancelled());
        assert!(second.is_cancelled());
    }

    #[test]
    fn reports_structured_intra_iteration_phases() {
        let cancellation = CancellationToken::new();
        let mut phases = Vec::new();
        calculate_with_control(doubling_request(2), &cancellation, |progress| {
            phases.push(progress.phase);
            true
        })
        .unwrap();

        assert!(phases.contains(&CalculationPhase::Indexing));
        assert!(phases.contains(&CalculationPhase::Rewriting));
        assert_eq!(phases.last(), Some(&CalculationPhase::Complete));
    }

    #[test]
    fn cpu_lineage_request_reports_the_exact_final_rewrite() {
        let request = CalculationRequest {
            grammar: CompiledGrammar::parse(
                "axiom A Keep Delete; match A then B [ C ]; match Delete then nothing;",
            )
            .unwrap(),
            iterations: 1,
            backend: BackendChoice::Cpu,
            seed: 9,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        };
        let (lineage, backend) =
            calculate_rewrite_lineage_with_control(request, &CancellationToken::new()).unwrap();

        assert_eq!(backend, BackendChoice::Cpu);
        assert_eq!(lineage.successor_modules_per_input(), [2, 1, 0]);
        assert_eq!(lineage.output_modules(), Some(3));
    }

    #[test]
    fn lineage_request_rejects_an_axiom_only_request() {
        let error =
            calculate_rewrite_lineage_with_control(doubling_request(0), &CancellationToken::new())
                .unwrap_err();

        assert!(matches!(error, CalculationError::BackendFailed { .. }));
    }

    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    #[test]
    fn cuda_lineage_matches_the_exact_stochastic_generation_when_available() {
        let request = CalculationRequest {
            grammar: CompiledGrammar::parse(
                "axiom A; match A weight 1 then F [ F ]; match A weight 1 then F F F;",
            )
            .unwrap(),
            iterations: 1,
            backend: BackendChoice::Cuda,
            seed: 27,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        };
        let cancellation = CancellationToken::new();
        let (lineage, backend) =
            match calculate_rewrite_lineage_with_control(request.clone(), &cancellation) {
                Ok(result) => result,
                Err(CalculationError::BackendUnavailable { .. }) => return,
                Err(error) => panic!("CUDA lineage calculation failed: {error}"),
            };
        let generation = calculate(request).expect("the same available CUDA request should run");

        assert_eq!(backend, BackendChoice::Cuda);
        assert_eq!(
            lineage.output_modules(),
            Some(generation.generation.module_count())
        );
    }

    #[test]
    fn cuda_preflight_failures_are_safe_auto_fallbacks() {
        let errors = [
            CalculationError::BackendUnavailable {
                backend: BackendChoice::Cuda,
                reason: String::from("driver unavailable"),
            },
            CalculationError::UnsupportedGrammar {
                backend: BackendChoice::Cuda,
                reason: String::from("unsupported production"),
            },
            CalculationError::BackendFailed {
                backend: BackendChoice::Cuda,
                reason: String::from("module load failed"),
            },
            CalculationError::ResourceExhausted {
                backend: BackendChoice::Cuda,
                resource: "rule table",
                reason: String::from("too large for one binding"),
            },
        ];

        assert!(errors.iter().all(CalculationError::is_cuda_fallback_error));
        assert!(!CalculationError::Cancelled.is_cuda_fallback_error());
    }
}
