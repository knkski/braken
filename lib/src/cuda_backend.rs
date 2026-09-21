//! Native CUDA f64 derivation for the flat, context-free grammar subset.
//!
//! This module derives backend-neutral [`Generation`] values; it does not
//! select a visualizer or rasterize a scene. Compilation rejects grammar
//! features whose CPU semantics cannot be preserved. Successful programs keep
//! reusable CUDA state and process logical generations in ordered, bounded
//! device groups. Those groups bound temporary memory and cancellation latency
//! but do not impose a whole-generation policy limit.
//!
//! Cancellation is cooperative between submissions and synchronization points.
//! Already launched kernels finish normally. Public incremental state is
//! transactional: a cancelled or failed operation leaves the last successful
//! host generation and generation index unchanged.
//! Seeded selection uses the same SplitMix64 stream and flattened-token keys as
//! the CPU f64 reference. First, Error, and Uniform ambiguity policies are
//! supported; Error validates the retained host token stream before launch.

use std::collections::{BTreeMap, btree_map::Entry};
use std::mem::MaybeUninit;
use std::ops::Range;

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig, LaunchConfig1D, sys,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::execution::{CalculationError, CalculationLimits, CancellationToken};
use crate::grammar::{
    AmbiguousRulePolicy, FloatWidth, Generation, GenerationItem, Grammar, Identifier, Module,
    StepStats, Value, Word, WordItem, evaluate_constant_expression_with_width,
    resolve_global_bindings_with_width,
};
use crate::ir::DerivationIr;

const THREADS: u32 = 256;
const CUDA_INDEX_LIMIT: usize = u32::MAX as usize;
// Large enough to keep launch/PCIe overhead amortized while bounding cancellation latency and
// temporary device allocations. This is a scheduling group, not a generation-size limit.
#[cfg(not(test))]
const CUDA_SUBMISSION_TOKENS: usize = 8 * 1024 * 1024;
#[cfg(test)]
const CUDA_SUBMISSION_TOKENS: usize = 2;
#[cfg(not(test))]
const CUDA_REWRITE_OUTPUT_TOKENS: usize = 32 * 1024 * 1024;
#[cfg(test)]
const CUDA_REWRITE_OUTPUT_TOKENS: usize = 4;
const BRANCH_OPEN: u32 = u32::MAX;
const BRANCH_CLOSE: u32 = u32::MAX - 1;
const WILDCARD_RULE: u32 = u32::MAX - 2;
const NO_RULE: u32 = u32::MAX;

#[cuda_module]
mod kernels {
    use super::*;

    static mut DECISION_SCAN_SCRATCH: SharedArray<u32, 256> = SharedArray::UNINIT;
    static mut U32_SCAN_SCRATCH: SharedArray<u32, 256> = SharedArray::UNINIT;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn select_rules_and_lengths(
        input: &[u32],
        input_offset: u32,
        input_len: u32,
        rule_lhs: &[u32],
        rule_weights: &[f64],
        rule_lengths: &[u32],
        seed: u64,
        iteration: u64,
        position_offset: u64,
        ambiguous_policy: u32,
        mut decisions: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if let Some(slot) = decisions.get_mut(idx) {
            if raw >= input_len as usize {
                return;
            }
            let token = input[input_offset as usize + raw];
            let mut selected = NO_RULE;
            let mut length = 1;

            if token != BRANCH_OPEN && token != BRANCH_CLOSE {
                let mut total_weight = 0.0;
                let mut match_count = 0_u32;
                let mut first = NO_RULE;
                let mut last = NO_RULE;
                let mut rule = 0;
                while rule < rule_lhs.len() {
                    if rule_lhs[rule] == token || rule_lhs[rule] == WILDCARD_RULE {
                        if first == NO_RULE {
                            first = rule as u32;
                        }
                        last = rule as u32;
                        total_weight += rule_weights[rule];
                        match_count += 1;
                    }
                    rule += 1;
                }

                selected = first;
                if first != NO_RULE && total_weight > 0.0 {
                    selected = last;
                    let random =
                        random_unit(seed, iteration, position_offset.wrapping_add(raw as u64));
                    let mut sample = random * total_weight;
                    rule = 0;
                    while rule < rule_lhs.len() {
                        if rule_lhs[rule] == token || rule_lhs[rule] == WILDCARD_RULE {
                            sample -= rule_weights[rule];
                            if sample < 0.0 {
                                selected = rule as u32;
                                break;
                            }
                        }
                        rule += 1;
                    }
                } else if first != NO_RULE && match_count > 1 && ambiguous_policy == 0 {
                    let target =
                        (random_unit(seed, iteration, position_offset.wrapping_add(raw as u64))
                            * match_count as f64) as u32;
                    let mut seen = 0_u32;
                    rule = 0;
                    while rule < rule_lhs.len() {
                        if rule_lhs[rule] == token || rule_lhs[rule] == WILDCARD_RULE {
                            if seen == target {
                                selected = rule as u32;
                                break;
                            }
                            seen += 1;
                        }
                        rule += 1;
                    }
                }

                if selected != NO_RULE {
                    length = rule_lengths[selected as usize];
                }
            }

            *slot = ((selected as u64) << 32) | length as u64;
        }
    }

    /// Scans the successor lengths packed into the low half of `decisions`.
    ///
    /// The host bounds every submission by its worst-case successor sum, so every block and
    /// group sum fits in `u32`. Raw output pointers are used only because one lane writes one
    /// compact block sum while all lanes write disjoint offsets.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub unsafe fn scan_decision_blocks(
        decisions: &[u64],
        input_len: u32,
        offsets: *mut u32,
        block_sums: *mut u32,
    ) {
        let raw = thread::index_1d().get();
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DECISION_SCAN_SCRATCH) };
        let value = if raw < input_len as usize {
            decisions[raw] as u32
        } else {
            0
        };
        unsafe { scratch.add(lane).write(value) };
        thread::sync_threads();

        let mut stride = 1usize;
        while stride < THREADS as usize {
            let scratch_index = (lane + 1) * stride * 2 - 1;
            if scratch_index < THREADS as usize {
                let left = unsafe { scratch.add(scratch_index - stride).read() };
                let right = unsafe { scratch.add(scratch_index).read() };
                unsafe { scratch.add(scratch_index).write(right.wrapping_add(left)) };
            }
            thread::sync_threads();
            stride *= 2;
        }

        if lane == 0 {
            unsafe {
                block_sums
                    .add(block)
                    .write(scratch.add(THREADS as usize - 1).read());
                scratch.add(THREADS as usize - 1).write(0);
            }
        }
        thread::sync_threads();

        stride = THREADS as usize / 2;
        loop {
            let scratch_index = (lane + 1) * stride * 2 - 1;
            if scratch_index < THREADS as usize {
                let left = unsafe { scratch.add(scratch_index - stride).read() };
                let right = unsafe { scratch.add(scratch_index).read() };
                unsafe {
                    scratch.add(scratch_index - stride).write(right);
                    scratch.add(scratch_index).write(right.wrapping_add(left));
                }
            }
            thread::sync_threads();
            if stride == 1 {
                break;
            }
            stride /= 2;
        }

        if raw < input_len as usize {
            unsafe { offsets.add(raw).write(scratch.add(lane).read()) };
        }
    }

    /// Scans one compact block-sum level. See `scan_decision_blocks` for the safety invariant.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub unsafe fn scan_u32_blocks(
        input: &[u32],
        input_len: u32,
        offsets: *mut u32,
        block_sums: *mut u32,
    ) {
        let raw = thread::index_1d().get();
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut U32_SCAN_SCRATCH) };
        let value = if raw < input_len as usize {
            input[raw]
        } else {
            0
        };
        unsafe { scratch.add(lane).write(value) };
        thread::sync_threads();

        let mut stride = 1usize;
        while stride < THREADS as usize {
            let scratch_index = (lane + 1) * stride * 2 - 1;
            if scratch_index < THREADS as usize {
                let left = unsafe { scratch.add(scratch_index - stride).read() };
                let right = unsafe { scratch.add(scratch_index).read() };
                unsafe { scratch.add(scratch_index).write(right.wrapping_add(left)) };
            }
            thread::sync_threads();
            stride *= 2;
        }

        if lane == 0 {
            unsafe {
                block_sums
                    .add(block)
                    .write(scratch.add(THREADS as usize - 1).read());
                scratch.add(THREADS as usize - 1).write(0);
            }
        }
        thread::sync_threads();

        stride = THREADS as usize / 2;
        loop {
            let scratch_index = (lane + 1) * stride * 2 - 1;
            if scratch_index < THREADS as usize {
                let left = unsafe { scratch.add(scratch_index - stride).read() };
                let right = unsafe { scratch.add(scratch_index).read() };
                unsafe {
                    scratch.add(scratch_index - stride).write(right);
                    scratch.add(scratch_index).write(right.wrapping_add(left));
                }
            }
            thread::sync_threads();
            if stride == 1 {
                break;
            }
            stride /= 2;
        }

        if raw < input_len as usize {
            unsafe { offsets.add(raw).write(scratch.add(lane).read()) };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add_block_offsets(mut offsets: DisjointSlice<u32>, block_offsets: &[u32]) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if let Some(slot) = offsets.get_mut(idx) {
            *slot = slot.wrapping_add(block_offsets[raw / THREADS as usize]);
        }
    }

    fn random_unit(seed: u64, iteration: u64, position: u64) -> f64 {
        let mut value = seed
            ^ iteration.wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((value >> 11) as f64) * SCALE
    }

    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn rewrite_tokens(
        input: &[u32],
        input_offset: u32,
        offsets: &[u32],
        decisions: &[u64],
        rule_offsets: &[u32],
        rule_lengths: &[u32],
        rule_rhs: &[u32],
        output: &mut [u32],
    ) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if raw >= offsets.len() {
            return;
        }

        let token = input[input_offset as usize + raw];
        let output_offset = offsets[raw] as usize;
        let selected = (decisions[raw] >> 32) as u32;

        if selected == NO_RULE {
            if output_offset < output.len() {
                output[output_offset] = token;
            }
            return;
        }

        let selected = selected as usize;
        let rhs_offset = rule_offsets[selected] as usize;
        let rhs_len = rule_lengths[selected] as usize;
        let mut rhs_index = 0;
        while rhs_index < rhs_len {
            let destination = output_offset + rhs_index;
            if destination < output.len() {
                output[destination] = rule_rhs[rhs_offset + rhs_index];
            }
            rhs_index += 1;
        }
    }
}

/// Configuration for compiling and running a grammar on one native CUDA device.
///
/// CUDA supports the flat, context-free subset accepted by [`Self::compile`].
/// The default has no semantic generation limits; actual device, allocation,
/// and address-space constraints are reported as typed [`CalculationError`]s.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    device_id: usize,
    limits: Option<CalculationLimits>,
    memory_headroom_percent: usize,
    float_width: FloatWidth,
    ambiguous_rules: AmbiguousRulePolicy,
}

impl CudaBackend {
    /// Creates a backend targeting CUDA device zero with unbounded semantics.
    pub fn new() -> Self {
        Self {
            device_id: 0,
            limits: None,
            memory_headroom_percent: 60,
            float_width: FloatWidth::F64,
            ambiguous_rules: AmbiguousRulePolicy::Uniform,
        }
    }

    /// Selects the required floating-point semantics.
    pub fn float_width(mut self, float_width: FloatWidth) -> Self {
        self.float_width = float_width;
        self
    }

    /// Selects how ambiguous unweighted productions are resolved.
    pub fn ambiguous_rules(mut self, ambiguous_rules: AmbiguousRulePolicy) -> Self {
        self.ambiguous_rules = ambiguous_rules;
        self
    }

    /// Selects the CUDA device ordinal used when a context is opened.
    pub fn device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Applies caller-selected semantic generation limits.
    pub fn limits(mut self, limits: CalculationLimits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Sets the percentage of currently free device memory considered usable
    /// by capacity estimates.
    ///
    /// This is estimation headroom, not a generation-size limit.
    pub fn memory_headroom_percent(mut self, percent: usize) -> Self {
        self.memory_headroom_percent = percent.clamp(1, 100);
        self
    }

    /// Reports device capacity information for this grammar without converting
    /// those estimates into semantic generation limits.
    pub fn estimated_limits_for(
        &self,
        grammar: &Grammar,
    ) -> Result<CudaLimitEstimate, CalculationError> {
        self.validate_semantics()?;
        let compiled = FlatCudaProgram::compile(grammar)?;
        let ctx = self.open_context()?;
        CudaLimitEstimate::from_context(
            &ctx,
            self.memory_headroom_percent,
            compiled.max_successor_tokens(),
        )
    }

    /// Opens the configured device and loads this crate's embedded kernels.
    pub fn probe(&self) -> Result<(), CalculationError> {
        let ctx = self.open_context()?;

        // SAFETY: this crate owns the embedded device bundle produced for the kernels module.
        unsafe { kernels::load(&ctx).map_err(cuda_error)? };

        Ok(())
    }

    /// Compiles a grammar for repeated CUDA execution.
    ///
    /// Unsupported grammar semantics are rejected before a CUDA context is
    /// needed, allowing automatic selection to try another derivation backend.
    pub fn compile(&self, grammar: &Grammar) -> Result<CudaProgram, CalculationError> {
        self.validate_semantics()?;
        let compiled = FlatCudaProgram::compile(grammar)?;
        let axiom = compiled.decode_with_limits(
            &compiled.axiom,
            self.limits.unwrap_or_default(),
            "CUDA axiom",
        )?;
        Ok(CudaProgram {
            backend: self.clone(),
            flat: compiled,
            axiom,
        })
    }

    fn validate_semantics(&self) -> Result<(), CalculationError> {
        if self.float_width != FloatWidth::F64 {
            return Err(CalculationError::unsupported_cuda_grammar(format!(
                "the current CUDA kernels implement f64 derivation, not {}",
                self.float_width
            )));
        }
        Ok(())
    }

    /// Compiles validated backend-neutral IR through the current flat CUDA lowering.
    pub fn compile_ir(&self, ir: &DerivationIr) -> Result<CudaProgram, CalculationError> {
        let grammar = ir.to_grammar().map_err(|error| {
            CalculationError::unsupported_cuda_grammar(format!("invalid derivation IR: {error}"))
        })?;
        self.compile(&grammar)
    }

    /// Compiles and derives `iterations` generations with seed zero.
    pub fn run(
        &self,
        grammar: &Grammar,
        iterations: usize,
    ) -> Result<Generation, CalculationError> {
        self.run_with_seed(grammar, iterations, 0)
    }

    /// Compiles and derives `iterations` generations with a stable stochastic seed.
    pub fn run_with_seed(
        &self,
        grammar: &Grammar,
        iterations: usize,
        seed: u64,
    ) -> Result<Generation, CalculationError> {
        self.validate_semantics()?;
        let compiled = FlatCudaProgram::compile(grammar)?;
        let tokens = self.run_flat_with_control(
            &compiled,
            compiled.axiom.clone(),
            0,
            iterations,
            seed,
            None,
            |_, _, _| true,
        )?;
        compiled.decode_with_limits(&tokens, self.limits.unwrap_or_default(), "CUDA result")
    }

    #[allow(clippy::too_many_arguments)]
    fn run_flat_with_control(
        &self,
        compiled: &FlatCudaProgram,
        tokens: Vec<u32>,
        start_iteration: u64,
        iterations: usize,
        seed: u64,
        cancellation: Option<&CancellationToken>,
        on_iteration: impl FnMut(usize, usize, usize) -> bool,
    ) -> Result<Vec<u32>, CalculationError> {
        let mut session = CudaSession::new(self, compiled)?;
        let mut device_generation = None;
        session.run_flat_with_control(
            compiled,
            tokens,
            &mut device_generation,
            start_iteration,
            iterations,
            seed,
            self.limits.unwrap_or_default(),
            cancellation,
            on_iteration,
        )
    }

    fn open_context(&self) -> Result<std::sync::Arc<CudaContext>, CalculationError> {
        CudaContext::new(self.device_id).map_err(|error| CalculationError::BackendUnavailable {
            backend: crate::execution::BackendChoice::Cuda,
            reason: error.to_string(),
        })
    }
}

struct DeviceGeneration {
    chunks: Vec<DeviceBuffer<u32>>,
    len: usize,
}

impl DeviceGeneration {
    fn release(self, stream: &CudaStream) -> Result<(), CalculationError> {
        for chunk in self.chunks {
            release_buffer(chunk, stream)?;
        }
        Ok(())
    }
}

struct ScanLevel {
    offsets: DeviceBuffer<u32>,
    block_sums: DeviceBuffer<u32>,
}

struct DeviceScan {
    levels: Vec<ScanLevel>,
    total: usize,
}

impl DeviceScan {
    fn offsets(&self) -> &DeviceBuffer<u32> {
        &self.levels[0].offsets
    }

    fn release(self, stream: &CudaStream) -> Result<(), CalculationError> {
        for level in self.levels {
            release_buffer(level.offsets, stream)?;
            release_buffer(level.block_sums, stream)?;
        }
        Ok(())
    }
}

struct IterationOutput {
    tokens: Vec<u32>,
    chunks: Vec<DeviceBuffer<u32>>,
    total: usize,
    retained_bytes: usize,
    residency_budget: usize,
    retaining: bool,
}

impl IterationOutput {
    fn new(residency_budget: usize) -> Self {
        Self {
            tokens: Vec::new(),
            chunks: Vec::new(),
            total: 0,
            retained_bytes: 0,
            residency_budget,
            retaining: true,
        }
    }

    fn prepare_group(
        &mut self,
        output_len: usize,
        limits: CalculationLimits,
        stream: &CudaStream,
    ) -> Result<bool, CalculationError> {
        self.total = checked_cumulative_output_total(self.total, output_len, limits)?;
        self.tokens
            .try_reserve(output_len)
            .map_err(|error| cuda_host_resource("expanded CUDA generation", error.to_string()))?;

        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| cuda_host_resource("CUDA resident generation", "byte-size overflow"))?;
        let proposed = self.retained_bytes.checked_add(output_bytes);
        if self.retaining && proposed.is_none_or(|bytes| bytes > self.residency_budget) {
            self.retaining = false;
            self.retained_bytes = 0;
            for chunk in self.chunks.drain(..) {
                release_buffer(chunk, stream)?;
            }
        }
        if self.retaining {
            self.retained_bytes = proposed.unwrap();
        }
        Ok(self.retaining)
    }

    fn finish(self) -> (Vec<u32>, Option<DeviceGeneration>) {
        let devices = self.retaining.then_some(DeviceGeneration {
            chunks: self.chunks,
            len: self.total,
        });
        (self.tokens, devices)
    }

    fn release_devices(&mut self, stream: &CudaStream) -> Result<(), CalculationError> {
        self.retaining = false;
        self.retained_bytes = 0;
        for chunk in self.chunks.drain(..) {
            release_buffer(chunk, stream)?;
        }
        Ok(())
    }
}

fn checked_cumulative_output_total(
    current: usize,
    group_output: usize,
    limits: CalculationLimits,
) -> Result<usize, CalculationError> {
    let total = current.checked_add(group_output).ok_or_else(|| {
        cuda_host_resource(
            "host address space",
            "expanded generation length overflowed usize",
        )
    })?;
    let max_flat_tokens = limits.max_items.saturating_mul(2);
    if total > max_flat_tokens {
        return Err(CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource: "configured generation items",
            reason: format!(
                "expanded generation requires {total} flat tokens; configured item limit permits at most {max_flat_tokens}"
            ),
        });
    }
    Ok(total)
}

struct CudaSession {
    context: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    module: kernels::LoadedModule,
    rule_lhs: DeviceBuffer<u32>,
    rule_weights: DeviceBuffer<f64>,
    rule_offsets: DeviceBuffer<u32>,
    rule_lengths: DeviceBuffer<u32>,
    rule_rhs: DeviceBuffer<u32>,
    memory_headroom_percent: usize,
    ambiguous_rules: AmbiguousRulePolicy,
}

impl CudaSession {
    fn new(backend: &CudaBackend, compiled: &FlatCudaProgram) -> Result<Self, CalculationError> {
        let context = backend.open_context()?;
        let stream = context.default_stream();
        let rule_lhs =
            DeviceBuffer::from_host(&stream, &compiled.rule_lhs).map_err(cuda_runtime_error)?;
        let rule_weights =
            DeviceBuffer::from_host(&stream, &compiled.rule_weights).map_err(cuda_runtime_error)?;
        let rule_offsets =
            DeviceBuffer::from_host(&stream, &compiled.rule_offsets).map_err(cuda_runtime_error)?;
        let rule_lengths =
            DeviceBuffer::from_host(&stream, &compiled.rule_lengths).map_err(cuda_runtime_error)?;
        let rule_rhs =
            DeviceBuffer::from_host(&stream, &compiled.rule_rhs).map_err(cuda_runtime_error)?;
        // SAFETY: this crate owns the embedded device bundle produced for the kernels module.
        let module = unsafe { kernels::load(&context).map_err(cuda_runtime_error)? };
        Ok(Self {
            context,
            stream,
            module,
            rule_lhs,
            rule_weights,
            rule_offsets,
            rule_lengths,
            rule_rhs,
            memory_headroom_percent: backend.memory_headroom_percent,
            ambiguous_rules: backend.ambiguous_rules,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_flat_with_control(
        &mut self,
        compiled: &FlatCudaProgram,
        mut tokens: Vec<u32>,
        device_generation: &mut Option<DeviceGeneration>,
        start_iteration: u64,
        iterations: usize,
        seed: u64,
        limits: CalculationLimits,
        cancellation: Option<&CancellationToken>,
        mut on_iteration: impl FnMut(usize, usize, usize) -> bool,
    ) -> Result<Vec<u32>, CalculationError> {
        validate_token_limits_with_control(&tokens, limits, "axiom", cancellation)?;
        for local_iteration in 0..iterations {
            check_cancelled(cancellation)?;
            if tokens.is_empty() {
                *device_generation = None;
                if !on_iteration(local_iteration + 1, 0, 0) {
                    return Err(CalculationError::Cancelled);
                }
                continue;
            }
            let iteration = start_iteration.saturating_add(local_iteration as u64);
            let (next_tokens, next_devices) = self.rewrite_iteration(
                compiled,
                &tokens,
                device_generation.as_ref(),
                iteration,
                seed,
                limits,
                cancellation,
            )?;
            let metrics = validate_token_limits_with_control(
                &next_tokens,
                limits,
                "expanded generation",
                cancellation,
            )?;
            if let Some(previous) = device_generation.take() {
                previous.release(&self.stream)?;
            }
            tokens = next_tokens;
            *device_generation = next_devices;
            if !on_iteration(local_iteration + 1, metrics.modules, metrics.items) {
                return Err(CalculationError::Cancelled);
            }
        }
        Ok(tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_iteration(
        &self,
        compiled: &FlatCudaProgram,
        tokens: &[u32],
        device_generation: Option<&DeviceGeneration>,
        iteration: u64,
        seed: u64,
        limits: CalculationLimits,
        cancellation: Option<&CancellationToken>,
    ) -> Result<(Vec<u32>, Option<DeviceGeneration>), CalculationError> {
        if self.ambiguous_rules == AmbiguousRulePolicy::Error {
            for chunk in tokens.chunks(CUDA_SUBMISSION_TOKENS) {
                check_cancelled(cancellation)?;
                if let Some((symbol, count)) = compiled.first_unweighted_ambiguity(chunk) {
                    return Err(CalculationError::BackendFailed {
                        backend: crate::execution::BackendChoice::Cuda,
                        reason: format!(
                            "module `{symbol}` has {count} applicable unweighted productions"
                        ),
                    });
                }
            }
        }
        self.context.bind_to_thread().map_err(cuda_runtime_error)?;
        let (free_memory, _) = memory_info()?;
        let residency_budget = free_memory.saturating_mul(self.memory_headroom_percent) / 100;
        let mut output = IterationOutput::new(residency_budget);

        let groups_result = (|| {
            if let Some(device_generation) = device_generation {
                debug_assert_eq!(device_generation.len, tokens.len());
                let mut global_start = 0usize;
                for input in &device_generation.chunks {
                    let chunk_end = global_start.checked_add(input.len()).ok_or_else(|| {
                        cuda_host_resource("CUDA resident generation", "chunk range overflow")
                    })?;
                    let host_chunk = tokens.get(global_start..chunk_end).ok_or_else(|| {
                        cuda_host_resource(
                            "CUDA resident generation",
                            "device chunks did not match the host generation",
                        )
                    })?;
                    for range in SelectionRanges::new(
                        host_chunk,
                        &compiled.max_symbol_lengths,
                        CUDA_SUBMISSION_TOKENS,
                        CUDA_REWRITE_OUTPUT_TOKENS,
                    ) {
                        self.rewrite_group(
                            input,
                            range.start,
                            range.len(),
                            global_start + range.start,
                            iteration,
                            seed,
                            limits,
                            cancellation,
                            &mut output,
                        )?;
                    }
                    global_start = chunk_end;
                }
                if global_start != tokens.len() {
                    return Err(cuda_host_resource(
                        "CUDA resident generation",
                        "device chunks did not cover the host generation",
                    ));
                }
            } else {
                for range in SelectionRanges::new(
                    tokens,
                    &compiled.max_symbol_lengths,
                    CUDA_SUBMISSION_TOKENS,
                    CUDA_REWRITE_OUTPUT_TOKENS,
                ) {
                    check_cancelled(cancellation)?;
                    let input = DeviceBuffer::from_host(&self.stream, &tokens[range.clone()])
                        .map_err(cuda_runtime_error)?;
                    let result = self.rewrite_group(
                        &input,
                        0,
                        range.len(),
                        range.start,
                        iteration,
                        seed,
                        limits,
                        cancellation,
                        &mut output,
                    );
                    if result.is_ok() {
                        release_buffer(input, &self.stream)?;
                    }
                    result?;
                }
            }
            Ok(())
        })();
        if let Err(error) = groups_result {
            let _ = output.release_devices(&self.stream);
            return Err(error);
        }
        Ok(output.finish())
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_group(
        &self,
        input: &DeviceBuffer<u32>,
        input_offset: usize,
        input_len: usize,
        global_start: usize,
        iteration: u64,
        seed: u64,
        limits: CalculationLimits,
        cancellation: Option<&CancellationToken>,
        output: &mut IterationOutput,
    ) -> Result<(), CalculationError> {
        check_cancelled(cancellation)?;
        let input_offset = u32::try_from(input_offset).map_err(|_| {
            cuda_host_resource("CUDA input offset", "input chunk offset exceeds u32")
        })?;
        let input_len_u32 = u32::try_from(input_len)
            .map_err(|_| cuda_host_resource("CUDA submission length", "input group exceeds u32"))?;
        let mut decisions =
            DeviceBuffer::<u64>::zeroed(&self.stream, input_len).map_err(cuda_runtime_error)?;
        let prepared = self
            .module
            .prepare_select_rules_and_lengths(launch_config(input_len))
            .map_err(cuda_runtime_error)?;
        self.module
            .select_rules_and_lengths(
                &self.stream,
                &prepared,
                input,
                input_offset,
                input_len_u32,
                &self.rule_lhs,
                &self.rule_weights,
                &self.rule_lengths,
                seed,
                iteration,
                global_start as u64,
                match self.ambiguous_rules {
                    AmbiguousRulePolicy::Uniform => 0,
                    AmbiguousRulePolicy::First | AmbiguousRulePolicy::Error => 1,
                },
                &mut decisions,
            )
            .map_err(cuda_runtime_error)?;
        let scan = self.prefix_scan_decisions(&decisions, cancellation)?;
        let output_len = scan.total;
        let retain_device = output.prepare_group(output_len, limits, &self.stream)?;
        if output_len == 0 {
            scan.release(&self.stream)?;
            release_buffer(decisions, &self.stream)?;
            check_cancelled(cancellation)?;
            return Ok(());
        }

        let mut rewritten =
            DeviceBuffer::<u32>::zeroed(&self.stream, output_len).map_err(cuda_runtime_error)?;
        // SAFETY: every input owns the disjoint output interval established by the scanned
        // successor lengths, and this is a one-dimensional launch over exactly `input_len`.
        unsafe {
            self.module
                .rewrite_tokens(
                    &self.stream,
                    LaunchConfig::for_num_elems(input_len_u32),
                    input,
                    input_offset,
                    scan.offsets(),
                    &decisions,
                    &self.rule_offsets,
                    &self.rule_lengths,
                    &self.rule_rhs,
                    &mut rewritten,
                )
                .map_err(cuda_runtime_error)?;
        }
        let mut host = rewritten
            .to_host_vec(&self.stream)
            .map_err(cuda_runtime_error)?;
        output.tokens.append(&mut host);
        if retain_device {
            output.chunks.push(rewritten);
        } else {
            release_buffer(rewritten, &self.stream)?;
        }
        scan.release(&self.stream)?;
        release_buffer(decisions, &self.stream)?;
        check_cancelled(cancellation)
    }

    fn prefix_scan_decisions(
        &self,
        decisions: &DeviceBuffer<u64>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<DeviceScan, CalculationError> {
        debug_assert!(!decisions.is_empty());
        let mut levels = Vec::new();
        levels
            .try_reserve(2)
            .map_err(|error| cuda_host_resource("CUDA scan levels", error.to_string()))?;

        let offsets = DeviceBuffer::<u32>::zeroed(&self.stream, decisions.len())
            .map_err(cuda_runtime_error)?;
        let block_count = decisions.len().div_ceil(THREADS as usize);
        let block_sums =
            DeviceBuffer::<u32>::zeroed(&self.stream, block_count).map_err(cuda_runtime_error)?;
        let prepared = self
            .module
            .prepare_scan_decision_blocks(launch_config(decisions.len()))
            .map_err(cuda_runtime_error)?;
        // SAFETY: every launch lane owns one offset and lane zero owns one compact block sum.
        unsafe {
            self.module
                .scan_decision_blocks(
                    &self.stream,
                    &prepared,
                    decisions,
                    decisions.len() as u32,
                    offsets.cu_deviceptr() as *mut u32,
                    block_sums.cu_deviceptr() as *mut u32,
                )
                .map_err(cuda_runtime_error)?;
        }
        levels.push(ScanLevel {
            offsets,
            block_sums,
        });

        while levels.last().unwrap().block_sums.len() > 1 {
            check_cancelled(cancellation)?;
            let input = &levels.last().unwrap().block_sums;
            let input_len = input.len();
            let offsets =
                DeviceBuffer::<u32>::zeroed(&self.stream, input_len).map_err(cuda_runtime_error)?;
            let block_count = input_len.div_ceil(THREADS as usize);
            let block_sums = DeviceBuffer::<u32>::zeroed(&self.stream, block_count)
                .map_err(cuda_runtime_error)?;
            let prepared = self
                .module
                .prepare_scan_u32_blocks(launch_config(input_len))
                .map_err(cuda_runtime_error)?;
            // SAFETY: every launch lane owns one offset and lane zero owns one compact block sum.
            unsafe {
                self.module
                    .scan_u32_blocks(
                        &self.stream,
                        &prepared,
                        input,
                        input_len as u32,
                        offsets.cu_deviceptr() as *mut u32,
                        block_sums.cu_deviceptr() as *mut u32,
                    )
                    .map_err(cuda_runtime_error)?;
            }
            levels.push(ScanLevel {
                offsets,
                block_sums,
            });
        }

        let total = levels
            .last()
            .unwrap()
            .block_sums
            .to_host_vec(&self.stream)
            .map_err(cuda_runtime_error)?[0] as usize;
        for lower_index in (0..levels.len().saturating_sub(1)).rev() {
            check_cancelled(cancellation)?;
            let (lower_levels, upper_levels) = levels.split_at_mut(lower_index + 1);
            let lower = &mut lower_levels[lower_index].offsets;
            let upper = &upper_levels[0].offsets;
            let prepared = self
                .module
                .prepare_add_block_offsets(launch_config(lower.len()))
                .map_err(cuda_runtime_error)?;
            self.module
                .add_block_offsets(&self.stream, &prepared, lower, upper)
                .map_err(cuda_runtime_error)?;
        }
        Ok(DeviceScan { levels, total })
    }
}

fn release_buffer<T: cuda_core::DeviceCopy>(
    buffer: DeviceBuffer<T>,
    stream: &CudaStream,
) -> Result<(), CalculationError> {
    // SAFETY: all allocations and work in a CUDA session use this one stream. Enqueueing the
    // free after their last use preserves stream order without DeviceBuffer's context-wide
    // synchronization on every bounded group.
    unsafe { buffer.drop_async(stream).map_err(cuda_runtime_error) }
}

/// Immutable compiled CUDA program reusable across executions and seeds.
#[derive(Debug, Clone)]
pub struct CudaProgram {
    backend: CudaBackend,
    flat: FlatCudaProgram,
    axiom: Generation,
}

impl CudaProgram {
    /// Returns the evaluated, backend-neutral axiom.
    pub fn axiom(&self) -> &Generation {
        &self.axiom
    }

    /// Derives `iterations` generations with seed zero.
    pub fn run(&self, iterations: usize) -> Result<Generation, CalculationError> {
        self.run_with_seed(iterations, 0)
    }

    /// Derives `iterations` generations with a stable stochastic seed.
    pub fn run_with_seed(
        &self,
        iterations: usize,
        seed: u64,
    ) -> Result<Generation, CalculationError> {
        self.run_with_control(iterations, seed, None, |_, _, _| true)
    }

    pub(crate) fn run_with_control(
        &self,
        iterations: usize,
        seed: u64,
        cancellation: Option<&CancellationToken>,
        on_iteration: impl FnMut(usize, usize, usize) -> bool,
    ) -> Result<Generation, CalculationError> {
        let tokens = self.backend.run_flat_with_control(
            &self.flat,
            self.flat.axiom.clone(),
            0,
            iterations,
            seed,
            cancellation,
            on_iteration,
        )?;
        self.flat.decode_with_limits(
            &tokens,
            self.backend.limits.unwrap_or_default(),
            "CUDA result",
        )
    }

    /// Starts an incremental execution with seed zero.
    pub fn start(&self) -> CudaState {
        self.start_with_seed(0)
    }

    /// Starts an incremental execution with a stable stochastic seed.
    pub fn start_with_seed(&self, seed: u64) -> CudaState {
        CudaState {
            program: std::sync::Arc::new(self.clone()),
            generation: self.axiom.clone(),
            tokens: self.flat.axiom.clone(),
            session: None,
            device_generation: None,
            generation_index: 0,
            seed,
        }
    }
}

/// Incremental CUDA derivation state with a reusable device session.
///
/// The state exposes only the last successfully decoded host generation.
/// Failed or cancelled operations leave that host generation and its generation
/// index unchanged; internal device scratch is not part of the public state.
pub struct CudaState {
    program: std::sync::Arc<CudaProgram>,
    generation: Generation,
    tokens: Vec<u32>,
    session: Option<CudaSession>,
    device_generation: Option<DeviceGeneration>,
    generation_index: u64,
    seed: u64,
}

impl std::fmt::Debug for CudaState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaState")
            .field("generation", &self.generation)
            .field("token_count", &self.tokens.len())
            .field("session_initialized", &self.session.is_some())
            .field(
                "device_resident",
                &self
                    .device_generation
                    .as_ref()
                    .map(|generation| generation.len),
            )
            .field("generation_index", &self.generation_index)
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

impl CudaState {
    /// Returns the last successfully completed host generation.
    pub fn generation(&self) -> &Generation {
        &self.generation
    }

    /// Number of successful rewrites after the axiom.
    pub fn generation_index(&self) -> u64 {
        self.generation_index
    }

    /// Consumes the state and returns its last successful generation.
    pub fn into_generation(self) -> Generation {
        self.generation
    }

    /// Rewrites one generation without external cancellation.
    pub fn step(&mut self) -> Result<StepStats, CalculationError> {
        self.step_with_control(&CancellationToken::new())
    }

    /// Rewrites one generation with cooperative cancellation between bounded
    /// CUDA submissions.
    pub fn step_with_control(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<StepStats, CalculationError> {
        let input_modules = self.generation.module_count();
        let input_items = self.generation.item_count();
        let target = self.generation_index.saturating_add(1);
        if self.session.is_none() {
            self.session = Some(CudaSession::new(&self.program.backend, &self.program.flat)?);
        }
        let next_tokens = self.session.as_mut().unwrap().run_flat_with_control(
            &self.program.flat,
            self.tokens.clone(),
            &mut self.device_generation,
            self.generation_index,
            1,
            self.seed,
            self.program.backend.limits.unwrap_or_default(),
            Some(cancellation),
            |_, _, _| true,
        )?;
        let next = self.program.flat.decode_with_limits(
            &next_tokens,
            self.program.backend.limits.unwrap_or_default(),
            "CUDA result",
        )?;
        let stats = StepStats {
            generation: target,
            input_modules,
            output_modules: next.module_count(),
            input_items,
            output_items: next.item_count(),
        };
        self.generation = next;
        self.tokens = next_tokens;
        self.generation_index = target;
        Ok(stats)
    }

    /// Rewrites `iterations` generations without external cancellation.
    pub fn advance(&mut self, iterations: usize) -> Result<(), CalculationError> {
        self.advance_with_control(iterations, &CancellationToken::new())
    }

    /// Rewrites `iterations` generations with cooperative cancellation.
    ///
    /// The public state advances only if the complete operation succeeds.
    pub fn advance_with_control(
        &mut self,
        iterations: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), CalculationError> {
        if iterations == 0 {
            return Ok(());
        }
        let target = self.generation_index.saturating_add(iterations as u64);
        if self.session.is_none() {
            self.session = Some(CudaSession::new(&self.program.backend, &self.program.flat)?);
        }
        let result = self.session.as_mut().unwrap().run_flat_with_control(
            &self.program.flat,
            self.tokens.clone(),
            &mut self.device_generation,
            self.generation_index,
            iterations,
            self.seed,
            self.program.backend.limits.unwrap_or_default(),
            Some(cancellation),
            |_, _, _| true,
        );
        let next_tokens = match result {
            Ok(tokens) => tokens,
            Err(error) => {
                // A multi-step attempt may have produced a resident intermediate generation.
                // The public state remains transactional on error, so discard that cache and
                // re-upload the unchanged host generation if the caller retries.
                if let Some(devices) = self.device_generation.take()
                    && let Some(session) = &self.session
                {
                    let _ = devices.release(&session.stream);
                }
                return Err(error);
            }
        };
        let next = self.program.flat.decode_with_limits(
            &next_tokens,
            self.program.backend.limits.unwrap_or_default(),
            "CUDA result",
        )?;
        self.generation = next;
        self.tokens = next_tokens;
        self.generation_index = target;
        Ok(())
    }
}

/// Device-derived capacity information for diagnostics and scheduling.
///
/// These fields describe launch or resident-working-set capacity. They are not
/// authoritative whole-generation limits because execution shards large input.
#[derive(Debug, Clone)]
pub struct CudaLimitEstimate {
    /// Semantic limits suitable for passing directly to execution.
    ///
    /// CUDA capacity figures below describe one resident working set or dispatch. Generations
    /// are sharded and streamed, so those figures must not become whole-generation limits.
    pub limits: CalculationLimits,
    /// CUDA device name reported by the driver.
    pub device_name: String,
    /// Device compute capability as `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Free device memory observed when the estimate was created.
    pub free_memory_bytes: usize,
    /// Total device memory reported by the driver.
    pub total_memory_bytes: usize,
    /// Portion of free memory admitted by [`CudaBackend::memory_headroom_percent`].
    pub usable_memory_bytes: usize,
    /// Percentage of free memory admitted to the estimate.
    pub memory_headroom_percent: usize,
    /// Maximum grid dimension along the one-dimensional launch axis.
    pub max_grid_x: u32,
    /// Threads used by one backend block.
    pub block_threads: u32,
    /// Maximum tokens addressable by one launch, before sharding.
    pub max_launch_threads: usize,
    /// Per-buffer flat token address capacity of the backend's `u32` indexes.
    pub index_limited_modules: usize,
    /// Conservative resident scratch and output bytes per input token.
    pub bytes_per_input_token_worst_case: usize,
    /// Largest compiled successor, measured in flat tokens.
    pub max_successor_tokens: usize,
    /// Estimated input tokens that fit in one resident working set.
    pub memory_limited_tokens: usize,
}

impl CudaLimitEstimate {
    fn from_context(
        ctx: &std::sync::Arc<CudaContext>,
        memory_headroom_percent: usize,
        max_successor_tokens: usize,
    ) -> Result<Self, CalculationError> {
        ctx.bind_to_thread().map_err(cuda_error)?;

        let device_name = ctx.device_name().map_err(cuda_error)?;
        let compute_capability = ctx.compute_capability().map_err(cuda_error)?;
        let launch_limits = ctx.launch_limits().map_err(cuda_error)?;
        let (free_memory_bytes, total_memory_bytes) = memory_info()?;
        let usable_memory_bytes = free_memory_bytes.saturating_mul(memory_headroom_percent) / 100;
        let max_successor_tokens = max_successor_tokens.max(1);
        let bytes_per_input_token_worst_case =
            std::mem::size_of::<u32>() * (4 + max_successor_tokens);
        let memory_limited_tokens = usable_memory_bytes / bytes_per_input_token_worst_case;
        let max_grid_dim = launch_limits.max_grid_dim();
        let max_launch_threads = (max_grid_dim.0 as usize).saturating_mul(THREADS as usize);
        let index_limited_modules = CUDA_INDEX_LIMIT;

        Ok(Self {
            limits: CalculationLimits::default(),
            device_name,
            compute_capability,
            free_memory_bytes,
            total_memory_bytes,
            usable_memory_bytes,
            memory_headroom_percent,
            max_grid_x: max_grid_dim.0,
            block_threads: THREADS,
            max_launch_threads,
            index_limited_modules,
            bytes_per_input_token_worst_case,
            max_successor_tokens,
            memory_limited_tokens,
        })
    }

    /// Formats the estimate as concise diagnostic lines.
    pub fn debug_lines(&self) -> Vec<String> {
        vec![
            format!(
                "CUDA device: {} (sm_{}{})",
                self.device_name, self.compute_capability.0, self.compute_capability.1
            ),
            format!(
                "CUDA memory: {} free / {} total",
                format_bytes(self.free_memory_bytes),
                format_bytes(self.total_memory_bytes)
            ),
            format!(
                "CUDA usable memory: {} ({}% of free memory)",
                format_bytes(self.usable_memory_bytes),
                self.memory_headroom_percent
            ),
            format!(
                "CUDA resident-working-set estimate: {} bytes/input token = sizeof(u32) * (4 temporary input-sized buffers + {} output growth)",
                self.bytes_per_input_token_worst_case, self.max_successor_tokens
            ),
            format!(
                "CUDA estimated resident tokens: {} (larger generations are streamed)",
                self.memory_limited_tokens
            ),
            format!(
                "CUDA maximum threads per launch: {} (max_grid_x {} * block_threads {})",
                self.max_launch_threads, self.max_grid_x, self.block_threads
            ),
            format!(
                "CUDA per-buffer address capacity: {} tokens (u32 device indices)",
                self.index_limited_modules
            ),
            "CUDA semantic generation limits: unbounded (capacity figures are advisory scheduling diagnostics)".to_string(),
        ]
    }
}

fn memory_info() -> Result<(usize, usize), CalculationError> {
    let mut free = MaybeUninit::<usize>::uninit();
    let mut total = MaybeUninit::<usize>::uninit();
    unsafe {
        sys::cuMemGetInfo_v2(free.as_mut_ptr(), total.as_mut_ptr())
            .result()
            .map_err(cuda_error)?;
        Ok((free.assume_init(), total.assume_init()))
    }
}

fn format_bytes(bytes: usize) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;

    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.2} MiB", bytes as f64 / MIB)
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct FlatCudaProgram {
    axiom: Vec<u32>,
    id_to_name: Vec<Identifier>,
    max_symbol_lengths: Vec<u32>,
    rule_lhs: Vec<u32>,
    rule_weights: Vec<f64>,
    rule_offsets: Vec<u32>,
    rule_lengths: Vec<u32>,
    rule_rhs: Vec<u32>,
}

impl FlatCudaProgram {
    fn compile(grammar: &Grammar) -> Result<Self, CalculationError> {
        if grammar.context_filter.is_some() {
            return Err(CalculationError::unsupported_cuda_grammar(
                "`ignore` and `only` context filters are not supported by the CUDA backend yet",
            ));
        }

        let mut symbols = SymbolTable::default();
        let axiom = encode_word(&grammar.axiom, &mut symbols, "axiom")?;
        let globals = resolve_global_bindings_with_width(&grammar.bindings, FloatWidth::F64)
            .map_err(|error| {
                CalculationError::unsupported_cuda_grammar(format!(
                    "global bindings could not be evaluated: {error}"
                ))
            })?;
        let mut weight_modes = BTreeMap::<u32, bool>::new();
        let rule_count = grammar.productions.len();
        let mut rule_lhs = try_vec_capacity(rule_count, "CUDA rule table")?;
        let mut rule_weights = try_vec_capacity(rule_count, "CUDA rule weights")?;
        let mut rule_offsets = try_vec_capacity(rule_count, "CUDA rule offsets")?;
        let mut rule_lengths = try_vec_capacity(rule_count, "CUDA rule lengths")?;
        let mut rule_rhs = Vec::new();

        for production in &grammar.productions {
            if !production.center.arguments.is_empty() {
                return Err(CalculationError::unsupported_cuda_grammar(
                    "parametric productions are not supported by the CUDA backend yet",
                ));
            }
            if production.left.is_some() || production.right.is_some() {
                return Err(CalculationError::unsupported_cuda_grammar(
                    "context-sensitive productions are not supported by the CUDA backend yet",
                ));
            }
            if let Some(condition) = &production.condition {
                let applies =
                    evaluate_constant_expression_with_width(condition, &globals, FloatWidth::F64)
                        .and_then(Value::as_bool)
                        .map_err(|error| {
                            CalculationError::unsupported_cuda_grammar(format!(
                                "condition for module `{}` is not constant: {error}",
                                production.center.name
                            ))
                        })?;
                if !applies {
                    continue;
                }
            }
            let lhs = if production.center.name.as_str() == "_" {
                WILDCARD_RULE
            } else {
                symbols.intern(&production.center.name)?
            };
            let weighted = production.weight.is_some();
            match weight_modes.entry(lhs) {
                Entry::Vacant(entry) => {
                    entry.insert(weighted);
                }
                Entry::Occupied(entry) if *entry.get() != weighted => {
                    return Err(CalculationError::unsupported_cuda_grammar(format!(
                        "module `{}` has both weighted and unweighted productions",
                        production.center.name
                    )));
                }
                Entry::Occupied(_) => {}
            }

            let weight = production
                .weight
                .as_ref()
                .map(|expression| {
                    evaluate_constant_expression_with_width(expression, &globals, FloatWidth::F64)
                })
                .transpose()
                .map_err(|error| {
                    CalculationError::unsupported_cuda_grammar(format!(
                        "weight for module `{}` is not a constant expression: {error}",
                        production.center.name
                    ))
                })?
                .map(Value::as_number)
                .transpose()
                .map_err(|error| {
                    CalculationError::unsupported_cuda_grammar(format!(
                        "weight for module `{}` is not numeric: {error}",
                        production.center.name
                    ))
                })?
                .unwrap_or(0.0);
            if weighted && (!weight.is_finite() || weight <= 0.0) {
                return Err(CalculationError::unsupported_cuda_grammar(format!(
                    "production weight for module `{}` evaluated to invalid value {weight}",
                    production.center.name
                )));
            }

            let rhs = encode_word(&production.successor, &mut symbols, "successor")?;
            let rhs_offset =
                u32::try_from(rule_rhs.len()).map_err(|_| CalculationError::ResourceExhausted {
                    backend: crate::execution::BackendChoice::Cuda,
                    resource: "CUDA rule address space",
                    reason: "combined successors exceed the backend u32 index space".to_string(),
                })?;
            let rhs_len =
                u32::try_from(rhs.len()).map_err(|_| CalculationError::ResourceExhausted {
                    backend: crate::execution::BackendChoice::Cuda,
                    resource: "CUDA successor address space",
                    reason: format!(
                        "successor for module `{}` exceeds the backend u32 length space",
                        production.center.name
                    ),
                })?;
            rule_rhs.try_reserve(rhs.len()).map_err(|error| {
                CalculationError::ResourceExhausted {
                    backend: crate::execution::BackendChoice::Cuda,
                    resource: "CUDA rule successors",
                    reason: error.to_string(),
                }
            })?;
            rule_lhs.push(lhs);
            rule_weights.push(weight);
            rule_offsets.push(rhs_offset);
            rule_lengths.push(rhs_len);
            rule_rhs.extend(rhs);
        }

        if let Some(&wildcard_weighted) = weight_modes.get(&WILDCARD_RULE) {
            for (&lhs, &weighted) in &weight_modes {
                if lhs != WILDCARD_RULE && weighted != wildcard_weighted {
                    return Err(CalculationError::unsupported_cuda_grammar(
                        "wildcard and exact productions mix weighted and unweighted applicable rules",
                    ));
                }
            }
        }

        for lhs in 0..symbols.names.len() as u32 {
            let total = rule_lhs
                .iter()
                .zip(&rule_weights)
                .filter(|(candidate, _)| **candidate == lhs || **candidate == WILDCARD_RULE)
                .map(|(_, weight)| *weight)
                .sum::<f64>();
            if !total.is_finite() || total <= 0.0 {
                let weighted = weight_modes.get(&lhs).copied().unwrap_or(false)
                    || weight_modes.get(&WILDCARD_RULE).copied().unwrap_or(false);
                if weighted {
                    return Err(CalculationError::unsupported_cuda_grammar(format!(
                        "stochastic production weights for module `{}` do not have a positive finite sum",
                        symbols.names[lhs as usize]
                    )));
                }
            }
        }

        let mut max_symbol_lengths =
            try_vec_capacity(symbols.names.len(), "CUDA successor bounds")?;
        for symbol in 0..symbols.names.len() as u32 {
            let mut matched = false;
            let mut maximum = 0u32;
            for (&lhs, &length) in rule_lhs.iter().zip(&rule_lengths) {
                if lhs == symbol || lhs == WILDCARD_RULE {
                    matched = true;
                    maximum = maximum.max(length);
                }
            }
            max_symbol_lengths.push(if matched { maximum } else { 1 });
        }

        Ok(Self {
            axiom,
            id_to_name: symbols.names,
            max_symbol_lengths,
            rule_lhs,
            rule_weights,
            rule_offsets,
            rule_lengths,
            rule_rhs,
        })
    }

    fn first_unweighted_ambiguity(&self, tokens: &[u32]) -> Option<(String, usize)> {
        tokens.iter().find_map(|token| {
            if *token == BRANCH_OPEN || *token == BRANCH_CLOSE {
                return None;
            }
            let count = self
                .rule_lhs
                .iter()
                .zip(&self.rule_weights)
                .filter(|(lhs, weight)| {
                    (**lhs == *token || **lhs == WILDCARD_RULE) && **weight == 0.0
                })
                .count();
            (count > 1).then(|| {
                (
                    self.id_to_name
                        .get(*token as usize)
                        .map_or_else(|| format!("#{token}"), ToString::to_string),
                    count,
                )
            })
        })
    }

    #[cfg(test)]
    fn decode(&self, tokens: &[u32]) -> Result<Generation, CalculationError> {
        self.decode_with_limits(tokens, CalculationLimits::default(), "CUDA generation")
    }

    fn decode_with_limits(
        &self,
        tokens: &[u32],
        limits: CalculationLimits,
        label: &str,
    ) -> Result<Generation, CalculationError> {
        validate_token_limits(tokens, limits, label)?;
        let mut stack = try_vec_capacity(1, "CUDA decode traversal stack")?;
        stack.push(Vec::new());
        for &token in tokens {
            match token {
                BRANCH_OPEN => try_push(&mut stack, Vec::new(), "CUDA decode traversal stack")?,
                BRANCH_CLOSE => {
                    if stack.len() == 1 {
                        return Err(CalculationError::unsupported_cuda_grammar(
                            "CUDA output contained an unmatched branch close",
                        ));
                    }
                    let branch = Generation(stack.pop().unwrap());
                    try_push(
                        stack.last_mut().unwrap(),
                        GenerationItem::Branch(branch),
                        "decoded generation items",
                    )?;
                }
                symbol => {
                    let Some(name) = self.id_to_name.get(symbol as usize) else {
                        return Err(CalculationError::unsupported_cuda_grammar(
                            "CUDA output contained an unknown module symbol",
                        ));
                    };
                    try_push(
                        stack.last_mut().unwrap(),
                        GenerationItem::Module(Module::new(name.clone(), Vec::new())),
                        "decoded generation items",
                    )?;
                }
            }
        }
        if stack.len() != 1 {
            return Err(CalculationError::unsupported_cuda_grammar(
                "CUDA output contained an unclosed branch",
            ));
        }
        Ok(Generation(stack.pop().unwrap()))
    }

    fn max_successor_tokens(&self) -> usize {
        self.rule_lengths.iter().copied().max().unwrap_or(1).max(1) as usize
    }
}

#[derive(Default)]
struct SymbolTable {
    names: Vec<Identifier>,
    ids: BTreeMap<Identifier, u32>,
}

impl SymbolTable {
    fn intern(&mut self, name: &Identifier) -> Result<u32, CalculationError> {
        match self.ids.entry(name.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                if self.names.len() >= WILDCARD_RULE as usize {
                    return Err(CalculationError::ResourceExhausted {
                        backend: crate::execution::BackendChoice::Cuda,
                        resource: "CUDA symbol address space",
                        reason: "symbol table exhausted its reserved u32 token space".to_string(),
                    });
                }
                let id = self.names.len() as u32;
                self.names
                    .try_reserve(1)
                    .map_err(|error| CalculationError::ResourceExhausted {
                        backend: crate::execution::BackendChoice::Cuda,
                        resource: "CUDA symbol table",
                        reason: error.to_string(),
                    })?;
                self.names.push(name.clone());
                entry.insert(id);
                Ok(id)
            }
        }
    }
}

fn encode_word(
    word: &Word,
    symbols: &mut SymbolTable,
    role: &str,
) -> Result<Vec<u32>, CalculationError> {
    let mut encoded = try_vec_capacity(word.len(), "CUDA flat word")?;
    let mut stack = try_vec_capacity(1, "CUDA encoding traversal stack")?;
    stack.push((word, 0usize));
    while let Some((current, next)) = stack.last_mut() {
        if *next == current.0.len() {
            stack.pop();
            if !stack.is_empty() {
                try_push_flat_token(&mut encoded, BRANCH_CLOSE, role)?;
            }
            continue;
        }
        let item = &current.0[*next];
        *next += 1;
        match item {
            WordItem::Module(module) => {
                if !module.arguments.is_empty() {
                    return Err(CalculationError::unsupported_cuda_grammar(format!(
                        "{role} module `{}` has parameters",
                        module.name
                    )));
                }
                let symbol = symbols.intern(&module.name)?;
                try_push_flat_token(&mut encoded, symbol, role)?;
            }
            WordItem::Branch(branch) => {
                try_push_flat_token(&mut encoded, BRANCH_OPEN, role)?;
                try_push(&mut stack, (branch, 0), "CUDA encoding traversal stack")?;
            }
        }
    }
    Ok(encoded)
}

#[derive(Debug, Clone, Copy)]
struct TokenMetrics {
    modules: usize,
    items: usize,
}

fn validate_token_limits(
    tokens: &[u32],
    limits: CalculationLimits,
    label: &str,
) -> Result<TokenMetrics, CalculationError> {
    validate_token_limits_with_control(tokens, limits, label, None)
}

fn validate_token_limits_with_control(
    tokens: &[u32],
    limits: CalculationLimits,
    label: &str,
    cancellation: Option<&CancellationToken>,
) -> Result<TokenMetrics, CalculationError> {
    let mut modules = 0usize;
    let mut items = 0usize;
    let mut depth = 0usize;
    let mut max_depth = 0usize;
    for (index, &token) in tokens.iter().enumerate() {
        if index % 65_536 == 0 {
            check_cancelled(cancellation)?;
        }
        match token {
            BRANCH_OPEN => {
                items = items
                    .checked_add(1)
                    .ok_or_else(|| cuda_host_resource("generation item count", "usize overflow"))?;
                depth = depth.checked_add(1).ok_or_else(|| {
                    cuda_host_resource("generation branch depth", "usize overflow")
                })?;
                max_depth = max_depth.max(depth);
            }
            BRANCH_CLOSE => {
                if depth == 0 {
                    return Err(CalculationError::unsupported_cuda_grammar(format!(
                        "{label} contains an unmatched branch close"
                    )));
                }
                depth -= 1;
            }
            _ => {
                modules = modules.checked_add(1).ok_or_else(|| {
                    cuda_host_resource("generation module count", "usize overflow")
                })?;
                items = items
                    .checked_add(1)
                    .ok_or_else(|| cuda_host_resource("generation item count", "usize overflow"))?;
            }
        }
    }
    if depth != 0 {
        return Err(CalculationError::unsupported_cuda_grammar(format!(
            "{label} contains an unclosed branch"
        )));
    }
    if modules > limits.max_modules {
        return Err(CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource: "configured generation modules",
            reason: format!(
                "{label} has {modules} modules, exceeding the configured limit of {}",
                limits.max_modules
            ),
        });
    }
    if items > limits.max_items {
        return Err(CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource: "configured generation items",
            reason: format!(
                "{label} has {items} items, exceeding the configured limit of {}",
                limits.max_items
            ),
        });
    }
    if max_depth > limits.max_branch_depth {
        return Err(CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource: "configured generation branch depth",
            reason: format!(
                "{label} has branch depth {max_depth}, exceeding the configured limit of {}",
                limits.max_branch_depth
            ),
        });
    }
    Ok(TokenMetrics { modules, items })
}

fn launch_config(len: usize) -> LaunchConfig1D {
    LaunchConfig1D::new((len as u32).div_ceil(THREADS), THREADS, 0)
}

#[derive(Debug, Clone)]
struct SelectionRanges<'a> {
    tokens: &'a [u32],
    max_symbol_lengths: &'a [u32],
    next: usize,
    max_inputs: usize,
    max_outputs: usize,
}

impl<'a> SelectionRanges<'a> {
    fn new(
        tokens: &'a [u32],
        max_symbol_lengths: &'a [u32],
        max_inputs: usize,
        max_outputs: usize,
    ) -> Self {
        assert!(max_inputs > 0, "CUDA submission groups must accept input");
        assert!(max_outputs > 0, "CUDA submission groups must accept output");
        Self {
            tokens,
            max_symbol_lengths,
            next: 0,
            max_inputs,
            max_outputs,
        }
    }

    fn successor_bound(&self, token: u32) -> usize {
        match token {
            BRANCH_OPEN | BRANCH_CLOSE => 1,
            symbol => self
                .max_symbol_lengths
                .get(symbol as usize)
                .copied()
                .unwrap_or(1) as usize,
        }
    }
}

impl Iterator for SelectionRanges<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.tokens.len() {
            return None;
        }

        let start = self.next;
        let input_end = start.saturating_add(self.max_inputs).min(self.tokens.len());
        let mut output_tokens = 0usize;
        while self.next < input_end {
            let successor_tokens = self.successor_bound(self.tokens[self.next]);
            let Some(candidate) = output_tokens.checked_add(successor_tokens) else {
                break;
            };
            if self.next != start && candidate > self.max_outputs {
                break;
            }
            output_tokens = candidate;
            self.next += 1;
        }

        // A single successor may be larger than the scheduling target. Include it alone so the
        // caller receives a typed allocation/device error instead of stalling the planner.
        if self.next == start {
            self.next += 1;
        }
        Some(start..self.next)
    }
}

fn try_vec_capacity<T>(
    capacity: usize,
    resource: &'static str,
) -> Result<Vec<T>, CalculationError> {
    let mut output = Vec::new();
    output
        .try_reserve(capacity)
        .map_err(|error| CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource,
            reason: error.to_string(),
        })?;
    Ok(output)
}

fn cuda_host_resource(resource: &'static str, reason: impl Into<String>) -> CalculationError {
    CalculationError::ResourceExhausted {
        backend: crate::execution::BackendChoice::Cuda,
        resource,
        reason: reason.into(),
    }
}

fn try_push<T>(
    output: &mut Vec<T>,
    value: T,
    resource: &'static str,
) -> Result<(), CalculationError> {
    output
        .try_reserve(1)
        .map_err(|error| CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource,
            reason: error.to_string(),
        })?;
    output.push(value);
    Ok(())
}

fn try_push_flat_token(
    output: &mut Vec<u32>,
    token: u32,
    role: &str,
) -> Result<(), CalculationError> {
    if output.len() == CUDA_INDEX_LIMIT {
        return Err(CalculationError::ResourceExhausted {
            backend: crate::execution::BackendChoice::Cuda,
            resource: "CUDA token address space",
            reason: format!("{role} exceeds the backend u32 token index space"),
        });
    }
    try_push(output, token, "CUDA flat word")
}

fn check_cancelled(cancellation: Option<&CancellationToken>) -> Result<(), CalculationError> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        Err(CalculationError::Cancelled)
    } else {
        Ok(())
    }
}

fn cuda_error(error: impl std::fmt::Display) -> CalculationError {
    CalculationError::cuda_runtime_failure(error.to_string())
}

fn cuda_runtime_error(error: impl std::fmt::Display) -> CalculationError {
    CalculationError::cuda_runtime_failure(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_context_free_module_rewrites() {
        let grammar = Grammar::parse(
            r#"
                axiom b;
                match a then a b;
                match b then a;
            "#,
        )
        .unwrap();

        let program = FlatCudaProgram::compile(&grammar).unwrap();

        assert_eq!(program.axiom.len(), 1);
        assert_eq!(program.rule_lhs.len(), 2);
        assert_eq!(program.rule_rhs.len(), 3);

        let public_program = CudaBackend::new().compile(&grammar).unwrap();
        assert_eq!(public_program.axiom().to_string(), "b");
        let state = public_program.start_with_seed(7);
        assert_eq!(state.generation_index(), 0);
        assert_eq!(state.generation().to_string(), "b");
    }

    #[test]
    fn rejects_parametric_grammars_before_opening_cuda() {
        let grammar = Grammar::parse(
            r#"
                axiom F(1);
                match F(x) then F(x + 1);
            "#,
        )
        .unwrap();

        let error = FlatCudaProgram::compile(&grammar).unwrap_err();

        assert!(matches!(
            error,
            CalculationError::UnsupportedGrammar {
                backend: crate::execution::BackendChoice::Cuda,
                ..
            }
        ));
    }

    #[test]
    fn compiles_weighted_branches_and_global_weights() {
        let grammar = Grammar::parse(
            r#"
                let likely = 3 * 10;
                let unlikely = 10;
                axiom F [ X ];
                match F weight likely then F [ L ] F;
                match F weight unlikely then F [ R ];
            "#,
        )
        .unwrap();

        let program = FlatCudaProgram::compile(&grammar).unwrap();

        assert_eq!(program.rule_lhs.len(), 2);
        assert_eq!(program.rule_weights, vec![30.0, 10.0]);
        assert!(program.axiom.contains(&BRANCH_OPEN));
        assert!(program.axiom.contains(&BRANCH_CLOSE));
        assert_eq!(
            program.decode(&program.axiom).unwrap().to_string(),
            "F [ X ]"
        );
    }

    #[test]
    fn rejects_mixed_weighted_and_unweighted_alternatives() {
        let grammar = Grammar::parse(
            r#"
                axiom F;
                match F then A;
                match F weight 1 then B;
            "#,
        )
        .unwrap();

        let error = FlatCudaProgram::compile(&grammar).unwrap_err();
        assert!(error.to_string().contains("both weighted and unweighted"));
    }

    #[test]
    fn compiles_wildcards_and_constant_conditions_in_source_order() {
        let grammar = Grammar::parse(
            r#"
                let enabled = true;
                axiom A B;
                match A when false then X;
                match _ when enabled then W;
                match A then Y;
            "#,
        )
        .unwrap();
        let program = FlatCudaProgram::compile(&grammar).unwrap();

        assert_eq!(program.rule_lhs.len(), 2);
        assert_eq!(program.rule_lhs[0], WILDCARD_RULE);
        assert_ne!(program.rule_lhs[1], WILDCARD_RULE);
    }

    #[test]
    fn branch_tokens_obey_item_and_depth_limits() {
        let grammar = Grammar::parse("axiom F [ G [ H ] ];").unwrap();
        let program = FlatCudaProgram::compile(&grammar).unwrap();

        validate_token_limits(
            &program.axiom,
            CalculationLimits {
                max_modules: 3,
                max_items: 5,
                max_branch_depth: 2,
            },
            "test",
        )
        .unwrap();
        assert!(
            validate_token_limits(
                &program.axiom,
                CalculationLimits {
                    max_modules: 3,
                    max_items: 5,
                    max_branch_depth: 1,
                },
                "test",
            )
            .is_err()
        );
    }

    #[test]
    fn flat_materialization_handles_deep_nesting_iteratively() {
        const DEPTH: usize = 12_000;
        let mut word = Word(vec![WordItem::Module(crate::grammar::ModuleExpr {
            name: Identifier::from("F"),
            arguments: Vec::new(),
        })]);
        for _ in 0..DEPTH {
            word = Word(vec![WordItem::Branch(word)]);
        }

        let mut symbols = SymbolTable::default();
        let encoded = encode_word(&word, &mut symbols, "deep test word").unwrap();
        assert_eq!(encoded.len(), DEPTH * 2 + 1);

        let program = FlatCudaProgram {
            axiom: encoded.clone(),
            id_to_name: symbols.names.clone(),
            max_symbol_lengths: vec![1],
            rule_lhs: Vec::new(),
            rule_weights: Vec::new(),
            rule_offsets: Vec::new(),
            rule_lengths: Vec::new(),
            rule_rhs: Vec::new(),
        };
        let decoded = program.decode(&encoded).unwrap();
        assert_eq!(decoded.max_branch_depth(), DEPTH);
        assert_eq!(decoded.module_count(), 1);
        assert_eq!(decoded.item_count(), DEPTH + 1);

        // The grammar AST itself predates Generation's iterative destructor. This test is about
        // CUDA's flat traversal and avoids recursively dropping that unrelated input structure.
        std::mem::forget(word);
        drop(decoded);
    }

    #[test]
    fn decode_applies_limits_before_materializing() {
        let program = FlatCudaProgram {
            axiom: vec![0, 0, 0],
            id_to_name: vec![Identifier::from("F")],
            max_symbol_lengths: vec![1],
            rule_lhs: Vec::new(),
            rule_weights: Vec::new(),
            rule_offsets: Vec::new(),
            rule_lengths: Vec::new(),
            rule_rhs: Vec::new(),
        };
        let error = program
            .decode_with_limits(
                &program.axiom,
                CalculationLimits {
                    max_modules: 2,
                    ..CalculationLimits::default()
                },
                "test generation",
            )
            .unwrap_err();
        assert!(matches!(
            error,
            CalculationError::ResourceExhausted {
                resource: "configured generation modules",
                ..
            }
        ));
    }

    #[test]
    fn fallible_vector_helper_classifies_capacity_overflow() {
        let error = try_vec_capacity::<u8>(usize::MAX, "test allocation").unwrap_err();
        assert!(matches!(
            error,
            CalculationError::ResourceExhausted {
                resource: "test allocation",
                ..
            }
        ));
    }

    #[test]
    fn bounded_submission_ranges_cover_input_without_overlap() {
        let tokens = [0; 10];
        let ranges = SelectionRanges::new(&tokens, &[1], 4, 100).collect::<Vec<_>>();
        assert_eq!(ranges, vec![0..4, 4..8, 8..10]);
        assert!(SelectionRanges::new(&[], &[1], 4, 100).next().is_none());
    }

    #[test]
    fn selection_ranges_bound_worst_case_input_and_output_work() {
        let tokens = [0_u32, 1, 2, 3, 4, 5, 6];
        let maximum_lengths = [3_u32, 3, 1, 0, 12, 2, 2];
        let ranges = SelectionRanges::new(&tokens, &maximum_lengths, 3, 5).collect::<Vec<_>>();
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, tokens.len());
        for adjacent in ranges.windows(2) {
            assert_eq!(adjacent[0].end, adjacent[1].start);
        }
        for range in ranges {
            assert!(range.len() <= 3);
            let output = tokens[range.clone()]
                .iter()
                .map(|token| maximum_lengths[*token as usize] as usize)
                .sum::<usize>();
            assert!(output <= 5 || range.len() == 1);
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn cumulative_output_planner_is_not_limited_by_u32_address_space() {
        let current = CUDA_INDEX_LIMIT - 8;
        let planned =
            checked_cumulative_output_total(current, 32, CalculationLimits::default()).unwrap();

        assert_eq!(planned, current + 32);
        assert!(planned > CUDA_INDEX_LIMIT);
    }

    #[test]
    fn cancellation_can_stop_between_submission_ranges() {
        let cancellation = CancellationToken::new();
        let tokens = [0; 100];
        for (completed, range) in SelectionRanges::new(&tokens, &[1], 10, 100).enumerate() {
            let result = check_cancelled(Some(&cancellation));
            if completed == 1 {
                assert!(matches!(result, Err(CalculationError::Cancelled)));
                assert!(range.start < 100);
                return;
            }
            result.unwrap();
            cancellation.cancel();
        }
        panic!("planner did not expose a cancellation boundary");
    }

    #[test]
    fn hierarchical_device_scan_matches_host_prefix_when_available() {
        let backend = CudaBackend::new();
        match backend.probe() {
            Ok(()) => {}
            Err(CalculationError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA probe failure: {error}"),
        }
        let grammar = Grammar::parse("axiom F; match F then F;").unwrap();
        let compiled = FlatCudaProgram::compile(&grammar).unwrap();
        let session = CudaSession::new(&backend, &compiled).unwrap();
        let lengths = (0..777).map(|index| index % 7).collect::<Vec<u64>>();
        let decisions = DeviceBuffer::from_host(&session.stream, &lengths).unwrap();

        let scan = session.prefix_scan_decisions(&decisions, None).unwrap();
        let actual = scan.offsets().to_host_vec(&session.stream).unwrap();
        let mut expected = Vec::with_capacity(lengths.len());
        let mut total = 0usize;
        for length in lengths {
            expected.push(total as u32);
            total += length as usize;
        }
        assert_eq!(actual, expected);
        assert_eq!(scan.total, total);

        scan.release(&session.stream).unwrap();
        release_buffer(decisions, &session.stream).unwrap();
    }

    #[test]
    fn state_reuses_one_cuda_session_when_available() {
        let backend = CudaBackend::new();
        match backend.probe() {
            Ok(()) => {}
            Err(CalculationError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA probe failure: {error}"),
        }
        let grammar = Grammar::parse("axiom F; match F then F F;").unwrap();
        let program = backend.compile(&grammar).unwrap();
        let mut state = program.start();
        state.step().unwrap();
        let first_context = std::sync::Arc::as_ptr(&state.session.as_ref().unwrap().context);
        assert!(state.device_generation.is_some());
        state.step().unwrap();
        let second_context = std::sync::Arc::as_ptr(&state.session.as_ref().unwrap().context);
        assert_eq!(first_context, second_context);
        assert_eq!(state.generation().module_count(), 4);
    }

    #[test]
    fn capacity_estimate_does_not_impose_generation_limits_when_available() {
        let backend = CudaBackend::new();
        let grammar = Grammar::parse("axiom F; match F then F F;").unwrap();
        let estimate = match backend.estimated_limits_for(&grammar) {
            Ok(estimate) => estimate,
            Err(CalculationError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA estimate failure: {error}"),
        };

        assert_eq!(estimate.limits, CalculationLimits::default());
        assert!(
            estimate
                .debug_lines()
                .iter()
                .any(|line| line.contains("semantic generation limits: unbounded"))
        );
    }

    #[test]
    fn chunked_cuda_preserves_global_position_rng_when_available() {
        let backend = CudaBackend::new();
        match backend.probe() {
            Ok(()) => {}
            Err(CalculationError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA probe failure: {error}"),
        }

        let grammar = Grammar::parse(
            "axiom F F F F F F F F; match F weight 1 then B; match F weight 1 then C;",
        )
        .unwrap();
        let seed = 0xCAFE_BABE;
        let actual = backend
            .run_with_seed(&grammar, 1, seed)
            .unwrap()
            .to_string();
        let expected = (0..8)
            .map(|position| {
                if reference_random_unit(seed, 0, position) < 0.5 {
                    "B"
                } else {
                    "C"
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(actual, expected);

        let deleting = Grammar::parse("axiom A A A A A; match A then nothing;").unwrap();
        assert!(backend.run(&deleting, 1).unwrap().items().is_empty());

        let branched =
            Grammar::parse("axiom A [ A ] A; match A weight 1 then B; match A weight 1 then C D;")
                .unwrap();
        let cpu = crate::grammar::CpuBackend::new()
            .float_width(FloatWidth::F64)
            .compile(&branched)
            .unwrap();
        for seed in 0..16 {
            assert_eq!(
                backend.run_with_seed(&branched, 2, seed).unwrap(),
                cpu.run_with_seed(2, seed).unwrap(),
                "seed {seed} must select the same f64 successors"
            );
        }
    }

    #[test]
    fn cuda_ambiguity_policies_match_the_cpu_reference_when_available() {
        let base = CudaBackend::new();
        match base.probe() {
            Ok(()) => {}
            Err(CalculationError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA probe failure: {error}"),
        }
        let grammar = Grammar::parse("axiom A; match A then B; match A then C;").unwrap();
        let first = base
            .clone()
            .ambiguous_rules(AmbiguousRulePolicy::First)
            .run(&grammar, 1)
            .unwrap();
        let cpu = crate::grammar::CpuBackend::new()
            .ambiguous_rules(AmbiguousRulePolicy::First)
            .compile(&grammar)
            .unwrap()
            .run(1)
            .unwrap();
        assert_eq!(first, cpu);
        assert!(matches!(
            base.ambiguous_rules(AmbiguousRulePolicy::Error)
                .run(&grammar, 1),
            Err(CalculationError::BackendFailed { .. })
        ));
    }

    #[test]
    fn cuda_rejects_f32_semantics_before_opening_a_device() {
        let grammar = Grammar::parse("axiom A;").unwrap();
        let backend = CudaBackend::new().float_width(FloatWidth::F32);
        assert!(matches!(
            backend.compile(&grammar),
            Err(CalculationError::UnsupportedGrammar { .. })
        ));
        assert!(matches!(
            backend.run(&grammar, 0),
            Err(CalculationError::UnsupportedGrammar { .. })
        ));
    }

    fn reference_random_unit(seed: u64, iteration: u64, position: u64) -> f64 {
        let mut value = seed
            ^ iteration.wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((value >> 11) as f64) * SCALE
    }

    #[test]
    fn stochastic_fixtures_run_reproducibly_when_cuda_is_available() {
        const STOCHASTIC_FIXTURES: [&str; 3] = [
            include_str!(
                "../tests/fixtures/abop/pass/abop-034-stochastic-branching-structures.lsys"
            ),
            include_str!(
                "../tests/fixtures/abop/pass/abop-035-stochastic-flower-field-rule-table-patch.lsys"
            ),
            include_str!(
                "../tests/fixtures/abop/pass/abop-059-stochastic-developmental-switch.lsys"
            ),
        ];
        for source in STOCHASTIC_FIXTURES {
            FlatCudaProgram::compile(&Grammar::parse(source).unwrap()).unwrap();
        }

        let backend = CudaBackend::new();
        match backend.probe() {
            Ok(()) => {}
            Err(CalculationError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA probe failure: {error}"),
        }

        for source in STOCHASTIC_FIXTURES {
            let grammar = Grammar::parse(source).unwrap();
            let first = backend.run_with_seed(&grammar, 2, 42).unwrap();
            let second = backend.run_with_seed(&grammar, 2, 42).unwrap();
            assert_eq!(first, second);
            assert!(first.module_count() > 0);
        }

        let grammar =
            Grammar::parse("axiom F; match F weight 1 then A; match F weight 1 then B;").unwrap();
        let choices = (0..32)
            .map(|seed| {
                backend
                    .run_with_seed(&grammar, 1, seed)
                    .unwrap()
                    .to_string()
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(choices.len(), 2);

        let program = backend.compile(&grammar).unwrap();
        let mut state = program.start_with_seed(9);
        state.step().unwrap();
        assert_eq!(state.generation(), &program.run_with_seed(1, 9).unwrap());
    }
}
