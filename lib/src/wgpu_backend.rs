//! Hardware WGPU compute backend for the f32 flat, context-free grammar subset.
//!
//! This module derives backend-neutral [`Generation`] values; it is independent
//! of visualization and of the GUI's WGPU display shader. The current generation
//! remains on the compute device while rule selection, hierarchical exclusive
//! prefix scan, and rewriting run in ordered, bounded shards. The host reads only
//! an exact output length per shard between generations and downloads the full
//! token stream only when the caller requests a [`Generation`]. The opt-in
//! lineage step additionally downloads its selected decisions and input-token
//! tags; ordinary steps retain the output-length-only behavior.
//!
//! Shard and dispatch sizes follow the selected adapter's reported limits. They
//! bound temporary resources and cancellation latency, not total generation
//! size. Cancellation is cooperative at submission, mapping, and download
//! boundaries; a submitted GPU operation completes normally.
//! Seeded selection uses the same f32 random stream and flattened-token keys as
//! the CPU f32 reference. First, Error, and Uniform ambiguity policies are
//! supported; Error checks downloaded input shards before device selection.

use std::collections::{BTreeMap, btree_map::Entry};
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::execution::CalculationLimits;
use crate::grammar::{
    AmbiguousRulePolicy, FloatWidth, Generation, GenerationItem, Grammar, Identifier, Module,
    RewriteLineage, Value, Word, WordItem, evaluate_constant_expression_with_width,
    resolve_global_bindings_with_width,
};
use crate::ir::DerivationIr;

const WORKGROUP_SIZE: usize = 256;
/// A scheduling quantum, not a result-size limit. This is enough work to
/// saturate current GPUs while bounding time between cancellation checks.
#[cfg(not(test))]
const MAX_WORKGROUPS_PER_SUBMISSION: usize = 4_096;
// Keep headless tests small while exercising exactly the same sharding path.
#[cfg(test)]
const MAX_WORKGROUPS_PER_SUBMISSION: usize = 2;
const BRANCH_OPEN: u32 = u32::MAX;
const BRANCH_CLOSE: u32 = u32::MAX - 1;
const WILDCARD_RULE: u32 = u32::MAX - 2;

const SELECT_SHADER: &str = include_str!("wgpu_select.wgsl");
const SCAN_SHADER: &str = include_str!("wgpu_scan.wgsl");
const SCAN_ADD_SHADER: &str = include_str!("wgpu_scan_add.wgsl");
const REWRITE_SHADER: &str = include_str!("wgpu_rewrite.wgsl");

#[cfg(not(target_arch = "wasm32"))]
type Shared<T> = Arc<T>;
#[cfg(target_arch = "wasm32")]
type Shared<T> = Rc<T>;

/// A typed WGPU failure. Backend selection can distinguish unsupported input
/// from unavailable hardware and a device that failed after work began.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// No adapter satisfying the request was available.
    AdapterUnavailable(String),
    /// Automatic acceleration found only a CPU/software adapter.
    SoftwareAdapter { name: String },
    /// An adapter was found but device creation failed.
    RequestDevice(String),
    /// The grammar uses semantics outside the flat context-free subset.
    UnsupportedGrammar(String),
    /// The selected Error policy encountered multiple applicable unweighted productions.
    AmbiguousRules { symbol: String, count: usize },
    /// A real adapter binding, buffer, or address constraint was exceeded.
    ResourceExhausted {
        resource: &'static str,
        requested: u64,
        limit: u64,
    },
    /// A caller-selected semantic generation limit was exceeded.
    LimitExceeded {
        resource: &'static str,
        actual: usize,
        limit: usize,
    },
    /// The selected device was lost after creation.
    DeviceLost(String),
    /// The device reported an out-of-memory failure.
    OutOfMemory(String),
    /// WGPU rejected an operation as invalid.
    Validation(String),
    /// WGPU reported an internal runtime failure.
    Internal(String),
    /// A device-to-host mapping failed.
    MapFailed(String),
    /// Cooperative cancellation was observed at a safe boundary.
    Cancelled,
}

impl fmt::Display for WgpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterUnavailable(reason) => {
                write!(formatter, "no WGPU compute adapter is available: {reason}")
            }
            Self::SoftwareAdapter { name } => write!(
                formatter,
                "WGPU adapter `{name}` is a CPU/software adapter and is not suitable for automatic hardware acceleration"
            ),
            Self::RequestDevice(reason) => {
                write!(
                    formatter,
                    "could not create the WGPU compute device: {reason}"
                )
            }
            Self::UnsupportedGrammar(reason) => {
                write!(formatter, "WGPU does not support this grammar: {reason}")
            }
            Self::AmbiguousRules { symbol, count } => write!(
                formatter,
                "module `{symbol}` has {count} applicable unweighted productions"
            ),
            Self::ResourceExhausted {
                resource,
                requested,
                limit,
            } => write!(
                formatter,
                "WGPU {resource} requires {requested} bytes/items, exceeding the device limit of {limit}"
            ),
            Self::LimitExceeded {
                resource,
                actual,
                limit,
            } => write!(
                formatter,
                "generated {resource} count {actual} exceeds the configured limit of {limit}"
            ),
            Self::DeviceLost(reason) => write!(formatter, "WGPU device was lost: {reason}"),
            Self::OutOfMemory(reason) => write!(formatter, "WGPU ran out of memory: {reason}"),
            Self::Validation(reason) => write!(formatter, "WGPU validation failed: {reason}"),
            Self::Internal(reason) => write!(formatter, "WGPU internal error: {reason}"),
            Self::MapFailed(reason) => write!(formatter, "could not read WGPU output: {reason}"),
            Self::Cancelled => formatter.write_str("WGPU calculation cancelled"),
        }
    }
}

impl Error for WgpuError {}

/// Adapter-selection policy for an explicit WGPU request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WgpuBackendOptions {
    /// Explicit WGPU requests may opt into a software adapter. Automatic
    /// selection should leave this false.
    pub allow_software_adapter: bool,
}

/// A reusable adapter, device, queue, and compute-pipeline set.
///
/// [`Self::request`] asks for a high-performance adapter and rejects CPU
/// adapters so automatic selection does not replace the CPU backend with a
/// software GPU stack. Explicit callers can override that policy with
/// [`WgpuBackendOptions`]. On wasm, adapter acquisition is asynchronous and
/// should run away from the browser UI thread, normally in a Web Worker.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    runtime: Shared<Runtime>,
    limits: Option<CalculationLimits>,
    float_width: FloatWidth,
    ambiguous_rules: AmbiguousRulePolicy,
}

impl WgpuBackend {
    /// Requests a high-performance, non-fallback adapter and its actual
    /// supported limits. On wasm this is asynchronous by necessity and is
    /// intended to run inside a Web Worker.
    pub async fn request() -> Result<Self, WgpuError> {
        Self::request_with_options(WgpuBackendOptions::default()).await
    }

    /// Requests an adapter using explicit selection options and its actual limits.
    pub async fn request_with_options(options: WgpuBackendOptions) -> Result<Self, WgpuError> {
        #[cfg(target_arch = "wasm32")]
        let backends = wgpu::Backends::BROWSER_WEBGPU;
        #[cfg(not(target_arch = "wasm32"))]
        let backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|error| WgpuError::AdapterUnavailable(error.to_string()))?;
        let adapter_info = adapter.get_info();
        if adapter_info.device_type == wgpu::DeviceType::Cpu && !options.allow_software_adapter {
            return Err(WgpuError::SoftwareAdapter {
                name: adapter_info.name,
            });
        }
        if !adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            return Err(WgpuError::AdapterUnavailable(format!(
                "adapter `{}` does not support compute shaders",
                adapter_info.name
            )));
        }

        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("braken.compute"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits.clone(),
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| WgpuError::RequestDevice(error.to_string()))?;

        let lost = Arc::new(Mutex::new(None));
        let lost_for_callback = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            let detail = if message.is_empty() {
                format!("{reason:?}")
            } else {
                format!("{reason:?}: {message}")
            };
            *lost_for_callback
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(detail);
        });

        let issue = Arc::new(Mutex::new(None));
        let issue_for_callback = Arc::clone(&issue);
        device.on_uncaptured_error(Arc::new(move |error| {
            let issue = match error {
                wgpu::Error::OutOfMemory { source } => {
                    RuntimeIssue::OutOfMemory(source.to_string())
                }
                wgpu::Error::Validation { description, .. } => {
                    RuntimeIssue::Validation(description)
                }
                wgpu::Error::Internal { description, .. } => RuntimeIssue::Internal(description),
            };
            let mut pending = issue_for_callback
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if pending.is_none() {
                *pending = Some(issue);
            }
        }));

        let pipelines = Pipelines::new(&device);
        let runtime = Runtime {
            device,
            queue,
            adapter_info,
            limits: adapter_limits,
            lost,
            issue,
            pipelines,
        };
        runtime.poll_once()?;
        runtime.check_health()?;

        Ok(Self {
            runtime: Shared::new(runtime),
            limits: None,
            float_width: FloatWidth::F32,
            ambiguous_rules: AmbiguousRulePolicy::Uniform,
        })
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

    /// Adds caller-selected semantic limits. With no explicit limits, only
    /// actual device/index/allocation limits constrain this backend.
    pub fn limits(mut self, limits: CalculationLimits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Returns the adapter identity and device class selected for this backend.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.runtime.adapter_info
    }

    /// Returns the actual limits requested from the selected device.
    pub fn device_limits(&self) -> &wgpu::Limits {
        &self.runtime.limits
    }

    /// Reports whether the adapter is not classified as a CPU device.
    pub fn is_hardware_adapter(&self) -> bool {
        self.runtime.adapter_info.device_type != wgpu::DeviceType::Cpu
    }

    /// Compiles and uploads a supported grammar for repeated execution.
    ///
    /// Unsupported grammar semantics are rejected explicitly rather than
    /// approximated with different GPU behavior.
    pub fn compile(&self, grammar: &Grammar) -> Result<WgpuProgram, WgpuError> {
        if self.float_width != FloatWidth::F32 {
            return Err(unsupported(format!(
                "the selected adapter path implements f32 derivation, not {}",
                self.float_width
            )));
        }
        let compiled = FlatWgpuProgram::compile(grammar, self.float_width)?;
        let rules = upload_storage(&self.runtime, "braken.wgpu.rules", &pack_rules(&compiled)?)?;
        let rhs = upload_storage(
            &self.runtime,
            "braken.wgpu.rhs",
            &pack_u32(&compiled.rule_rhs)?,
        )?;

        ensure_binding_size(&self.runtime.limits, rules.size(), "rule table binding")?;
        ensure_binding_size(&self.runtime.limits, rhs.size(), "successor table binding")?;

        Ok(WgpuProgram {
            backend: self.clone(),
            flat: Shared::new(compiled),
            rules,
            rhs,
        })
    }

    /// Compiles validated backend-neutral IR through the current flat WGPU lowering.
    pub fn compile_ir(&self, ir: &DerivationIr) -> Result<WgpuProgram, WgpuError> {
        let grammar = ir
            .to_grammar()
            .map_err(|error| unsupported(format!("invalid derivation IR: {error}")))?;
        self.compile(&grammar)
    }

    /// Compiles and asynchronously derives `iterations` generations with seed zero.
    pub async fn run(&self, grammar: &Grammar, iterations: usize) -> Result<Generation, WgpuError> {
        self.run_with_seed(grammar, iterations, 0).await
    }

    /// Compiles and asynchronously derives with a stable stochastic seed.
    pub async fn run_with_seed(
        &self,
        grammar: &Grammar,
        iterations: usize,
        seed: u64,
    ) -> Result<Generation, WgpuError> {
        self.run_with_seed_and_cancel(grammar, iterations, seed, || false)
            .await
    }

    /// Compiles and derives with cooperative cancellation between bounded GPU operations.
    pub async fn run_with_seed_and_cancel(
        &self,
        grammar: &Grammar,
        iterations: usize,
        seed: u64,
        mut is_cancelled: impl FnMut() -> bool,
    ) -> Result<Generation, WgpuError> {
        let program = self.compile(grammar)?;
        let mut state = program.start_with_seed(seed)?;
        state
            .advance_with_cancel(iterations, &mut is_cancelled)
            .await?;
        state.into_generation_with_cancel(&mut is_cancelled).await
    }
}

/// Immutable compiled WGPU program reusable across executions and seeds.
#[derive(Debug, Clone)]
pub struct WgpuProgram {
    backend: WgpuBackend,
    flat: Shared<FlatWgpuProgram>,
    rules: wgpu::Buffer,
    rhs: wgpu::Buffer,
}

impl WgpuProgram {
    /// Decodes and returns the evaluated axiom without opening an execution state.
    pub fn axiom(&self) -> Result<Generation, WgpuError> {
        self.flat.decode(&self.flat.axiom, self.backend.limits)
    }

    /// Uploads the axiom and starts an incremental execution with seed zero.
    pub fn start(&self) -> Result<WgpuState, WgpuError> {
        self.start_with_seed(0)
    }

    /// Uploads the axiom and starts with a stable stochastic seed.
    pub fn start_with_seed(&self, seed: u64) -> Result<WgpuState, WgpuError> {
        let generation =
            upload_generation(&self.backend.runtime, "braken.wgpu.axiom", &self.flat.axiom)?;
        Ok(WgpuState {
            program: Shared::new(self.clone()),
            generation,
            generation_index: 0,
            seed,
        })
    }
}

/// One logical generation split into ordered, independently bindable buffers.
/// Every shard fits both `max_buffer_size` and the storage-binding limit.
#[derive(Debug)]
struct DeviceGeneration {
    chunks: Vec<DeviceChunk>,
    len: usize,
}

#[derive(Debug)]
struct DeviceChunk {
    buffer: wgpu::Buffer,
    len: usize,
}

#[derive(Clone, Copy)]
struct SelectionUnit<'a> {
    input: &'a wgpu::Buffer,
    input_start: usize,
    global_base: usize,
    len: usize,
}

struct RewriteUnit<'a> {
    input: &'a wgpu::Buffer,
    input_start: usize,
    input_len: usize,
    decisions: &'a wgpu::Buffer,
    offsets: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    output_len: usize,
}

/// Incremental WGPU derivation state whose logical generation remains sharded
/// and device-resident until explicitly downloaded.
///
/// Each step is transactional: cancellation or failure before completion leaves
/// the prior device generation and generation index available for retry.
#[derive(Debug)]
pub struct WgpuState {
    program: Shared<WgpuProgram>,
    generation: DeviceGeneration,
    generation_index: u64,
    seed: u64,
}

impl WgpuState {
    /// Number of successful rewrites after the axiom.
    pub fn generation_index(&self) -> u64 {
        self.generation_index
    }

    /// Flat token count (modules plus branch-open and branch-close tokens).
    pub fn token_count(&self) -> usize {
        self.generation.len
    }

    /// Rewrites one generation without external cancellation.
    pub async fn step(&mut self) -> Result<(), WgpuError> {
        self.step_with_cancel(&mut || false).await
    }

    /// Rewrites one generation with cooperative cancellation at bounded GPU boundaries.
    pub async fn step_with_cancel(
        &mut self,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        self.step_internal(is_cancelled, false).await.map(|_| ())
    }

    /// Rewrites one generation and downloads only the compact selected-rule
    /// lineage in addition to the normal device work.
    pub async fn step_with_lineage_and_cancel(
        &mut self,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<RewriteLineage, WgpuError> {
        self.step_internal(is_cancelled, true)
            .await?
            .ok_or_else(|| WgpuError::Internal(String::from("WGPU lineage capture was omitted")))
    }

    async fn step_internal(
        &mut self,
        is_cancelled: &mut dyn FnMut() -> bool,
        capture_lineage: bool,
    ) -> Result<Option<RewriteLineage>, WgpuError> {
        if is_cancelled() {
            return Err(WgpuError::Cancelled);
        }
        if self.generation.len == 0 {
            self.generation_index = self.generation_index.saturating_add(1);
            return if capture_lineage {
                RewriteLineage::from_successor_module_counts(Vec::new())
                    .map(Some)
                    .map_err(|error| WgpuError::Internal(error.to_string()))
            } else {
                Ok(None)
            };
        }

        let runtime = &self.program.backend.runtime;
        let mut output_chunks = Vec::new();
        output_chunks
            .try_reserve(self.generation.chunks.len())
            .map_err(host_allocation_error)?;
        let mut output_len = 0usize;
        let mut chunk_global_base = 0usize;
        let mut lineage_counts = Vec::new();
        if capture_lineage {
            lineage_counts
                .try_reserve(self.generation.len)
                .map_err(host_allocation_error)?;
        }

        // A processing unit is chosen from the actual binding, dispatch, and
        // buffer limits plus worst-case successor growth. Each unit is one
        // bounded select/scan/rewrite sequence and therefore a cancellation
        // checkpoint without sacrificing a full-size GPU dispatch.
        for input_chunk in &self.generation.chunks {
            let units = plan_rewrite_chunks(
                input_chunk.len,
                self.program.flat.max_successor_tokens(),
                &runtime.limits,
            )?;
            for unit in units {
                if is_cancelled() {
                    return Err(WgpuError::Cancelled);
                }
                let unit_bytes = element_bytes(unit.len, "input processing unit")?;
                let decisions = create_storage_buffer(
                    runtime,
                    "braken.wgpu.decisions",
                    unit_bytes,
                    if capture_lineage {
                        wgpu::BufferUsages::COPY_SRC
                    } else {
                        wgpu::BufferUsages::empty()
                    },
                )?;
                let lengths = create_storage_buffer(
                    runtime,
                    "braken.wgpu.lengths",
                    unit_bytes,
                    wgpu::BufferUsages::COPY_SRC,
                )?;
                let global_base = chunk_global_base
                    .checked_add(unit.start)
                    .ok_or_else(|| index_overflow("logical generation position"))?;

                if self.program.backend.ambiguous_rules == AmbiguousRulePolicy::Error {
                    let input_tokens = download_u32_range(
                        runtime,
                        &input_chunk.buffer,
                        unit.start,
                        unit.len,
                        is_cancelled,
                    )
                    .await?;
                    if let Some((symbol, count)) =
                        self.program.flat.first_unweighted_ambiguity(&input_tokens)
                    {
                        return Err(WgpuError::AmbiguousRules { symbol, count });
                    }
                }

                self.select_rules(
                    SelectionUnit {
                        input: &input_chunk.buffer,
                        input_start: unit.start,
                        global_base,
                        len: unit.len,
                    },
                    &decisions,
                    &lengths,
                    is_cancelled,
                )
                .await?;
                if capture_lineage {
                    let input_tokens = download_u32_range(
                        runtime,
                        &input_chunk.buffer,
                        unit.start,
                        unit.len,
                        is_cancelled,
                    )
                    .await?;
                    let selected =
                        download_u32(runtime, &decisions, unit.len, is_cancelled).await?;
                    for (token, decision) in input_tokens.into_iter().zip(selected) {
                        if token == BRANCH_OPEN || token == BRANCH_CLOSE {
                            continue;
                        }
                        let count = if decision == u32::MAX {
                            1
                        } else {
                            self.program
                                .flat
                                .rule_module_counts
                                .get(decision as usize)
                                .copied()
                                .ok_or_else(|| {
                                    WgpuError::Validation(format!(
                                        "selected rule index {decision} is outside the compiled rule table"
                                    ))
                                })? as usize
                        };
                        lineage_counts.push(count);
                    }
                }
                let (offsets, overflow) =
                    self.prefix_scan(&lengths, unit.len, is_cancelled).await?;
                let unit_output_len = read_scan_total(
                    runtime,
                    &offsets,
                    &lengths,
                    &overflow,
                    unit.len,
                    is_cancelled,
                )
                .await?;
                output_len = output_len
                    .checked_add(unit_output_len)
                    .ok_or_else(|| index_overflow("expanded logical generation"))?;
                self.validate_flat_total(output_len)?;

                if unit_output_len != 0 {
                    let output = create_storage_buffer(
                        runtime,
                        "braken.wgpu.generation.shard",
                        element_bytes(unit_output_len, "expanded generation shard")?,
                        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    )?;
                    ensure_binding_size(
                        &runtime.limits,
                        output.size(),
                        "expanded generation shard binding",
                    )?;
                    self.rewrite_unit(
                        RewriteUnit {
                            input: &input_chunk.buffer,
                            input_start: unit.start,
                            input_len: unit.len,
                            decisions: &decisions,
                            offsets: &offsets,
                            output: &output,
                            output_len: unit_output_len,
                        },
                        is_cancelled,
                    )
                    .await?;
                    output_chunks
                        .try_reserve(1)
                        .map_err(host_allocation_error)?;
                    output_chunks.push(DeviceChunk {
                        buffer: output,
                        len: unit_output_len,
                    });
                }
            }
            chunk_global_base = chunk_global_base
                .checked_add(input_chunk.len)
                .ok_or_else(|| index_overflow("logical generation position"))?;
        }

        debug_assert_eq!(chunk_global_base, self.generation.len);
        let lineage = if capture_lineage {
            RewriteLineage::from_successor_module_counts(lineage_counts)
                .map(Some)
                .map_err(|error| WgpuError::Internal(error.to_string()))
        } else {
            Ok(None)
        }?;
        self.generation = DeviceGeneration {
            chunks: output_chunks,
            len: output_len,
        };
        self.generation_index = self.generation_index.saturating_add(1);
        Ok(lineage)
    }

    /// Rewrites `iterations` generations without external cancellation.
    pub async fn advance(&mut self, iterations: usize) -> Result<(), WgpuError> {
        self.advance_with_cancel(iterations, &mut || false).await
    }

    /// Rewrites `iterations` generations with cooperative cancellation.
    ///
    /// Successful earlier steps remain committed if a later step is cancelled.
    pub async fn advance_with_cancel(
        &mut self,
        iterations: usize,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        for _ in 0..iterations {
            self.step_with_cancel(is_cancelled).await?;
        }
        Ok(())
    }

    /// Downloads and decodes a copy of the current generation.
    pub async fn generation(&self) -> Result<Generation, WgpuError> {
        self.generation_with_cancel(&mut || false).await
    }

    /// Downloads and decodes a copy with cooperative cancellation.
    pub async fn generation_with_cancel(
        &self,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Generation, WgpuError> {
        download_generation(
            &self.program.backend.runtime,
            &self.generation,
            &self.program.flat,
            self.program.backend.limits,
            is_cancelled,
        )
        .await
    }

    /// Consumes the state, downloads, and decodes its generation.
    pub async fn into_generation(self) -> Result<Generation, WgpuError> {
        self.into_generation_with_cancel(&mut || false).await
    }

    /// Consumes the state and downloads with cooperative cancellation.
    pub async fn into_generation_with_cancel(
        self,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Generation, WgpuError> {
        self.generation_with_cancel(is_cancelled).await
    }

    async fn select_rules(
        &self,
        unit: SelectionUnit<'_>,
        decisions: &wgpu::Buffer,
        lengths: &wgpu::Buffer,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        let runtime = &self.program.backend.runtime;
        ensure_indexable(unit.len, "selection processing unit")?;
        let workgroups = unit.len.div_ceil(WORKGROUP_SIZE);
        if workgroups > runtime.limits.max_compute_workgroups_per_dimension as usize {
            return Err(WgpuError::ResourceExhausted {
                resource: "selection dispatch workgroups",
                requested: workgroups as u64,
                limit: runtime.limits.max_compute_workgroups_per_dimension as u64,
            });
        }
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("braken.wgpu.select.encoder"),
            });
        let layout = runtime.pipelines.select.get_bind_group_layout(0);
        let params = upload_uniform(
            runtime,
            "braken.wgpu.select.params",
            &pack_u32(&select_params_words(
                unit.len,
                unit.global_base,
                self.program.flat.rule_lhs.len(),
                self.seed,
                self.generation_index,
                self.program.backend.ambiguous_rules,
            ))?,
        )?;
        let input_offset = (unit.input_start * size_of_u32()) as u64;
        let input_size = (unit.len * size_of_u32()) as u64;
        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("braken.wgpu.select.bind_group"),
                layout: &layout,
                entries: &[
                    binding(0, &params, 0, params.size()),
                    binding(1, unit.input, input_offset, input_size),
                    binding(2, &self.program.rules, 0, self.program.rules.size()),
                    binding(3, decisions, 0, decisions.size()),
                    binding(4, lengths, 0, lengths.size()),
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("braken.wgpu.select.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.pipelines.select);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups as u32, 1, 1);
        drop(pass);
        runtime
            .submit_and_wait(encoder.finish(), is_cancelled)
            .await
    }

    async fn prefix_scan(
        &self,
        lengths: &wgpu::Buffer,
        token_count: usize,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer), WgpuError> {
        let runtime = &self.program.backend.runtime;
        let overflow = create_storage_buffer(
            runtime,
            "braken.wgpu.scan.overflow",
            size_of_u32() as u64,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        runtime.queue.write_buffer(&overflow, 0, &[0, 0, 0, 0]);

        let mut levels = Vec::new();
        let mut input = lengths.clone();
        let mut count = token_count;
        loop {
            let blocks = count.div_ceil(WORKGROUP_SIZE);
            let offsets = create_storage_buffer(
                runtime,
                "braken.wgpu.scan.offsets",
                element_bytes(count, "scan offsets")?,
                wgpu::BufferUsages::COPY_SRC,
            )?;
            let block_sums = create_storage_buffer(
                runtime,
                "braken.wgpu.scan.block_sums",
                element_bytes(blocks, "scan block sums")?,
                wgpu::BufferUsages::empty(),
            )?;
            ensure_binding_size(
                &runtime.limits,
                block_sums.size(),
                "scan block sums binding",
            )?;
            self.scan_level(
                &input,
                &offsets,
                &block_sums,
                &overflow,
                count,
                is_cancelled,
            )
            .await?;
            levels.push(ScanLevel {
                offsets,
                block_sums: block_sums.clone(),
                count,
            });
            if blocks == 1 {
                break;
            }
            input = block_sums;
            count = blocks;
        }

        for lower_index in (0..levels.len().saturating_sub(1)).rev() {
            let block_offsets = &levels[lower_index + 1].offsets;
            self.add_scan_offsets(
                &levels[lower_index].offsets,
                block_offsets,
                &overflow,
                levels[lower_index].count,
                is_cancelled,
            )
            .await?;
        }

        Ok((levels.remove(0).offsets, overflow))
    }

    async fn scan_level(
        &self,
        input: &wgpu::Buffer,
        offsets: &wgpu::Buffer,
        block_sums: &wgpu::Buffer,
        overflow: &wgpu::Buffer,
        count: usize,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        let runtime = &self.program.backend.runtime;
        ensure_indexable(count, "scan processing unit")?;
        let chunks = plan_dispatch_chunks(count, &runtime.limits)?;
        let layout = runtime.pipelines.scan.get_bind_group_layout(0);
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("braken.wgpu.scan.encoder"),
            });
        for chunk in chunks {
            debug_assert_eq!(chunk.start % WORKGROUP_SIZE, 0);
            let params = upload_uniform(
                runtime,
                "braken.wgpu.scan.params",
                &pack_u32(&[
                    chunk.len as u32,
                    (chunk.start / WORKGROUP_SIZE) as u32,
                    0,
                    0,
                ])?,
            )?;
            let offset = (chunk.start * size_of_u32()) as u64;
            let size = (chunk.len * size_of_u32()) as u64;
            let bind_group = runtime
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("braken.wgpu.scan.bind_group"),
                    layout: &layout,
                    entries: &[
                        binding(0, &params, 0, params.size()),
                        binding(1, input, offset, size),
                        binding(2, offsets, offset, size),
                        binding(3, block_sums, 0, block_sums.size()),
                        binding(4, overflow, 0, overflow.size()),
                    ],
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("braken.wgpu.scan.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&runtime.pipelines.scan);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(chunk.workgroups as u32, 1, 1);
        }
        runtime
            .submit_and_wait(encoder.finish(), is_cancelled)
            .await
    }

    async fn add_scan_offsets(
        &self,
        offsets: &wgpu::Buffer,
        block_offsets: &wgpu::Buffer,
        overflow: &wgpu::Buffer,
        count: usize,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        let runtime = &self.program.backend.runtime;
        ensure_indexable(count, "scan-offset processing unit")?;
        ensure_binding_size(
            &runtime.limits,
            block_offsets.size(),
            "scanned block offsets binding",
        )?;
        let chunks = plan_dispatch_chunks(count, &runtime.limits)?;
        let layout = runtime.pipelines.scan_add.get_bind_group_layout(0);
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("braken.wgpu.scan_add.encoder"),
            });
        for chunk in chunks {
            let params = upload_uniform(
                runtime,
                "braken.wgpu.scan_add.params",
                &pack_u32(&[
                    chunk.len as u32,
                    (chunk.start / WORKGROUP_SIZE) as u32,
                    0,
                    0,
                ])?,
            )?;
            let offset = (chunk.start * size_of_u32()) as u64;
            let size = (chunk.len * size_of_u32()) as u64;
            let bind_group = runtime
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("braken.wgpu.scan_add.bind_group"),
                    layout: &layout,
                    entries: &[
                        binding(0, &params, 0, params.size()),
                        binding(1, offsets, offset, size),
                        binding(2, block_offsets, 0, block_offsets.size()),
                        binding(3, overflow, 0, overflow.size()),
                    ],
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("braken.wgpu.scan_add.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&runtime.pipelines.scan_add);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(chunk.workgroups as u32, 1, 1);
        }
        runtime
            .submit_and_wait(encoder.finish(), is_cancelled)
            .await
    }

    async fn rewrite_unit(
        &self,
        unit: RewriteUnit<'_>,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        if unit.input_len == 0 || unit.output_len == 0 {
            return Ok(());
        }
        ensure_indexable(unit.input_len, "rewrite processing unit")?;
        ensure_indexable(unit.output_len, "rewrite output shard")?;
        let runtime = &self.program.backend.runtime;
        let workgroups = unit.input_len.div_ceil(WORKGROUP_SIZE);
        let layout = runtime.pipelines.rewrite.get_bind_group_layout(0);
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("braken.wgpu.rewrite.encoder"),
            });
        let params = upload_uniform(
            runtime,
            "braken.wgpu.rewrite.params",
            &pack_u32(&[unit.input_len as u32, 0, 0, 0])?,
        )?;
        let input_offset = (unit.input_start * size_of_u32()) as u64;
        let input_size = (unit.input_len * size_of_u32()) as u64;
        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("braken.wgpu.rewrite.bind_group"),
                layout: &layout,
                entries: &[
                    binding(0, &params, 0, params.size()),
                    binding(1, unit.input, input_offset, input_size),
                    binding(2, unit.decisions, 0, unit.decisions.size()),
                    binding(3, unit.offsets, 0, unit.offsets.size()),
                    binding(4, &self.program.rules, 0, self.program.rules.size()),
                    binding(5, &self.program.rhs, 0, self.program.rhs.size()),
                    binding(6, unit.output, 0, (unit.output_len * size_of_u32()) as u64),
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("braken.wgpu.rewrite.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.pipelines.rewrite);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups as u32, 1, 1);
        drop(pass);
        runtime
            .submit_and_wait(encoder.finish(), is_cancelled)
            .await
    }

    fn validate_flat_total(&self, total: usize) -> Result<(), WgpuError> {
        validate_flat_total(total, self.program.backend.limits)
    }
}

#[derive(Debug)]
struct Runtime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    lost: Arc<Mutex<Option<String>>>,
    issue: Arc<Mutex<Option<RuntimeIssue>>>,
    pipelines: Pipelines,
}

impl Runtime {
    fn check_health(&self) -> Result<(), WgpuError> {
        if let Some(reason) = self
            .lost
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            return Err(WgpuError::DeviceLost(reason));
        }
        if let Some(issue) = self
            .issue
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            return Err(match issue {
                RuntimeIssue::OutOfMemory(reason) => WgpuError::OutOfMemory(reason),
                RuntimeIssue::Validation(reason) => WgpuError::Validation(reason),
                RuntimeIssue::Internal(reason) => WgpuError::Internal(reason),
            });
        }
        Ok(())
    }

    fn poll_once(&self) -> Result<(), WgpuError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| WgpuError::DeviceLost(error.to_string()))?;
        Ok(())
    }

    async fn submit_and_wait(
        &self,
        command: wgpu::CommandBuffer,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), WgpuError> {
        if is_cancelled() {
            return Err(WgpuError::Cancelled);
        }
        self.check_health()?;
        self.queue.submit([command]);
        let (signal, completion) = callback_pair();
        self.queue
            .on_submitted_work_done(move || signal.complete(()));
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| WgpuError::DeviceLost(error.to_string()))?;
        completion.await;
        self.check_health()?;
        if is_cancelled() {
            return Err(WgpuError::Cancelled);
        }
        Ok(())
    }
}

#[derive(Debug)]
enum RuntimeIssue {
    OutOfMemory(String),
    Validation(String),
    Internal(String),
}

#[derive(Debug)]
struct Pipelines {
    select: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
    rewrite: wgpu::ComputePipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            select: create_pipeline(
                device,
                "braken.wgpu.select",
                SELECT_SHADER,
                "select_rules_and_lengths",
            ),
            scan: create_pipeline(device, "braken.wgpu.scan", SCAN_SHADER, "scan_blocks"),
            scan_add: create_pipeline(
                device,
                "braken.wgpu.scan_add",
                SCAN_ADD_SHADER,
                "add_block_offsets",
            ),
            rewrite: create_pipeline(
                device,
                "braken.wgpu.rewrite",
                REWRITE_SHADER,
                "rewrite_tokens",
            ),
        }
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[derive(Debug)]
struct ScanLevel {
    offsets: wgpu::Buffer,
    #[allow(dead_code)]
    block_sums: wgpu::Buffer,
    count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchChunk {
    start: usize,
    len: usize,
    workgroups: usize,
}

/// An allocation-free plan for splitting a logical sequence into physical
/// buffer/dispatch-sized chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchPlan {
    count: usize,
    regular_len: usize,
}

impl DispatchPlan {
    fn len(self) -> usize {
        self.count.div_ceil(self.regular_len)
    }

    fn iter(self) -> DispatchPlanIter {
        DispatchPlanIter {
            plan: self,
            next: 0,
        }
    }

    fn get(self, index: usize) -> Option<DispatchChunk> {
        if index >= self.len() {
            return None;
        }
        let start = index.checked_mul(self.regular_len)?;
        let len = (self.count - start).min(self.regular_len);
        Some(DispatchChunk {
            start,
            len,
            workgroups: len.div_ceil(WORKGROUP_SIZE),
        })
    }
}

impl IntoIterator for DispatchPlan {
    type Item = DispatchChunk;
    type IntoIter = DispatchPlanIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
struct DispatchPlanIter {
    plan: DispatchPlan,
    next: usize,
}

impl Iterator for DispatchPlanIter {
    type Item = DispatchChunk;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.plan.get(self.next)?;
        self.next += 1;
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.plan.len() - self.next.min(self.plan.len());
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DispatchPlanIter {}

fn plan_dispatch_chunks(count: usize, limits: &wgpu::Limits) -> Result<DispatchPlan, WgpuError> {
    if count == 0 {
        return Ok(DispatchPlan {
            count: 0,
            regular_len: 1,
        });
    }
    let alignment_elements =
        (limits.min_storage_buffer_offset_alignment as usize / size_of_u32()).max(1);
    let chunk_alignment = lcm(WORKGROUP_SIZE, alignment_elements);
    let binding_elements = limits.max_storage_buffer_binding_size as usize / size_of_u32();
    let buffer_elements =
        usize::try_from(limits.max_buffer_size / size_of_u32() as u64).unwrap_or(usize::MAX);
    let dispatch_elements = (limits.max_compute_workgroups_per_dimension as usize)
        .min(MAX_WORKGROUPS_PER_SUBMISSION)
        .saturating_mul(WORKGROUP_SIZE);
    let capacity = binding_elements.min(buffer_elements).min(dispatch_elements);
    let regular_len = capacity / chunk_alignment * chunk_alignment;
    if regular_len == 0 {
        return Err(WgpuError::ResourceExhausted {
            resource: "dispatch chunk",
            requested: (chunk_alignment * size_of_u32()) as u64,
            limit: limits.max_storage_buffer_binding_size as u64,
        });
    }

    Ok(DispatchPlan { count, regular_len })
}

fn plan_rewrite_chunks(
    count: usize,
    max_successor_tokens: usize,
    limits: &wgpu::Limits,
) -> Result<DispatchPlan, WgpuError> {
    if count == 0 {
        return Ok(DispatchPlan {
            count: 0,
            regular_len: 1,
        });
    }
    let alignment_elements =
        (limits.min_storage_buffer_offset_alignment as usize / size_of_u32()).max(1);
    let chunk_alignment = lcm(WORKGROUP_SIZE, alignment_elements);
    let binding_capacity = limits.max_storage_buffer_binding_size as usize / size_of_u32();
    let buffer_capacity =
        usize::try_from(limits.max_buffer_size / size_of_u32() as u64).unwrap_or(usize::MAX);
    let output_capacity = binding_capacity.min(buffer_capacity);
    let max_successor_tokens = max_successor_tokens.max(1);
    let by_output = output_capacity / max_successor_tokens;
    let by_input = binding_capacity.min(buffer_capacity);
    let by_dispatch = (limits.max_compute_workgroups_per_dimension as usize)
        .min(MAX_WORKGROUPS_PER_SUBMISSION)
        .saturating_mul(WORKGROUP_SIZE);
    let capacity = by_output.min(by_input).min(by_dispatch);
    let regular_len = capacity / chunk_alignment * chunk_alignment;
    if regular_len == 0 {
        return Err(WgpuError::ResourceExhausted {
            resource: "rewrite chunk for the largest successor",
            requested: (max_successor_tokens
                .saturating_mul(chunk_alignment)
                .saturating_mul(size_of_u32())) as u64,
            limit: (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size),
        });
    }

    Ok(DispatchPlan { count, regular_len })
}

async fn read_scan_total(
    runtime: &Runtime,
    offsets: &wgpu::Buffer,
    lengths: &wgpu::Buffer,
    overflow: &wgpu::Buffer,
    count: usize,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<usize, WgpuError> {
    debug_assert!(count > 0);
    let staging = create_readback_buffer(runtime, "braken.wgpu.total", 12)?;
    let source_offset = ((count - 1) * size_of_u32()) as u64;
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("braken.wgpu.total.encoder"),
        });
    encoder.copy_buffer_to_buffer(offsets, source_offset, &staging, 0, 4);
    encoder.copy_buffer_to_buffer(lengths, source_offset, &staging, 4, 4);
    encoder.copy_buffer_to_buffer(overflow, 0, &staging, 8, 4);
    runtime
        .submit_and_wait(encoder.finish(), is_cancelled)
        .await?;
    let bytes = map_read(runtime, &staging).await?;
    let offset = unpack_u32(&bytes[0..4]);
    let length = unpack_u32(&bytes[4..8]);
    let overflowed = unpack_u32(&bytes[8..12]) != 0;
    drop(bytes);
    staging.unmap();
    if overflowed {
        return Err(WgpuError::ResourceExhausted {
            resource: "32-bit prefix sum",
            requested: u32::MAX as u64 + 1,
            limit: u32::MAX as u64,
        });
    }
    let total = offset
        .checked_add(length)
        .ok_or(WgpuError::ResourceExhausted {
            resource: "32-bit token index",
            requested: u32::MAX as u64 + 1,
            limit: u32::MAX as u64,
        })?;
    Ok(total as usize)
}

async fn download_u32(
    runtime: &Runtime,
    source: &wgpu::Buffer,
    count: usize,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<u32>, WgpuError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let byte_len = element_bytes(count, "generation readback")?;
    let staging = create_readback_buffer(runtime, "braken.wgpu.readback", byte_len)?;
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("braken.wgpu.readback.encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
    runtime
        .submit_and_wait(encoder.finish(), is_cancelled)
        .await?;
    let bytes = map_read(runtime, &staging).await?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(host_allocation_error)?;
    output.extend(bytes.chunks_exact(size_of_u32()).map(unpack_u32));
    drop(bytes);
    staging.unmap();
    Ok(output)
}

async fn download_u32_range(
    runtime: &Runtime,
    source: &wgpu::Buffer,
    start: usize,
    count: usize,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<u32>, WgpuError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let byte_len = element_bytes(count, "generation range readback")?;
    let source_offset = start
        .checked_mul(size_of_u32())
        .ok_or_else(|| index_overflow("generation range readback offset"))?
        as u64;
    let staging = create_readback_buffer(runtime, "braken.wgpu.range_readback", byte_len)?;
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("braken.wgpu.range_readback.encoder"),
        });
    encoder.copy_buffer_to_buffer(source, source_offset, &staging, 0, byte_len);
    runtime
        .submit_and_wait(encoder.finish(), is_cancelled)
        .await?;
    let bytes = map_read(runtime, &staging).await?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(host_allocation_error)?;
    output.extend(bytes.chunks_exact(size_of_u32()).map(unpack_u32));
    drop(bytes);
    staging.unmap();
    Ok(output)
}

async fn download_generation(
    runtime: &Runtime,
    generation: &DeviceGeneration,
    program: &FlatWgpuProgram,
    limits: Option<CalculationLimits>,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Generation, WgpuError> {
    let mut decoder = GenerationDecoder::new(program, limits)?;
    let mut downloaded_count = 0usize;
    for chunk in &generation.chunks {
        if is_cancelled() {
            return Err(WgpuError::Cancelled);
        }
        let downloaded = download_u32(runtime, &chunk.buffer, chunk.len, is_cancelled).await?;
        downloaded_count = downloaded_count
            .checked_add(downloaded.len())
            .ok_or_else(|| index_overflow("downloaded logical generation"))?;
        decoder.extend(&downloaded)?;
    }
    debug_assert_eq!(downloaded_count, generation.len);
    decoder.finish()
}

async fn map_read(runtime: &Runtime, buffer: &wgpu::Buffer) -> Result<wgpu::BufferView, WgpuError> {
    let slice = buffer.slice(..);
    let (signal, completion) = callback_pair();
    slice.map_async(wgpu::MapMode::Read, move |result| signal.complete(result));
    #[cfg(not(target_arch = "wasm32"))]
    runtime
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| WgpuError::DeviceLost(error.to_string()))?;
    completion
        .await
        .map_err(|error| WgpuError::MapFailed(error.to_string()))?;
    runtime.check_health()?;
    Ok(slice.get_mapped_range())
}

fn create_readback_buffer(
    runtime: &Runtime,
    label: &'static str,
    size: u64,
) -> Result<wgpu::Buffer, WgpuError> {
    ensure_buffer_size(&runtime.limits, size, "readback buffer")?;
    let buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    runtime.poll_once()?;
    runtime.check_health()?;
    Ok(buffer)
}

fn upload_generation(
    runtime: &Runtime,
    label: &'static str,
    tokens: &[u32],
) -> Result<DeviceGeneration, WgpuError> {
    let plans = plan_dispatch_chunks(tokens.len(), &runtime.limits)?;
    let mut chunks = Vec::new();
    chunks
        .try_reserve(plans.len())
        .map_err(host_allocation_error)?;
    for plan in plans {
        let buffer = create_storage_buffer(
            runtime,
            label,
            element_bytes(plan.len, "generation upload shard")?,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )?;
        ensure_binding_size(
            &runtime.limits,
            buffer.size(),
            "generation upload shard binding",
        )?;
        runtime.queue.write_buffer(
            &buffer,
            0,
            &pack_u32(&tokens[plan.start..plan.start + plan.len])?,
        );
        chunks.push(DeviceChunk {
            buffer,
            len: plan.len,
        });
    }
    Ok(DeviceGeneration {
        chunks,
        len: tokens.len(),
    })
}

fn upload_storage(
    runtime: &Runtime,
    label: &'static str,
    bytes: &[u8],
) -> Result<wgpu::Buffer, WgpuError> {
    let size = bytes.len().max(4) as u64;
    let buffer = create_storage_buffer(runtime, label, size, wgpu::BufferUsages::COPY_DST)?;
    if !bytes.is_empty() {
        runtime.queue.write_buffer(&buffer, 0, bytes);
    }
    Ok(buffer)
}

fn upload_uniform(
    runtime: &Runtime,
    label: &'static str,
    bytes: &[u8],
) -> Result<wgpu::Buffer, WgpuError> {
    let size = bytes.len().max(16) as u64;
    ensure_buffer_size(&runtime.limits, size, "uniform buffer")?;
    if size > runtime.limits.max_uniform_buffer_binding_size as u64 {
        return Err(WgpuError::ResourceExhausted {
            resource: "uniform buffer",
            requested: size,
            limit: runtime.limits.max_uniform_buffer_binding_size as u64,
        });
    }
    let buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    runtime.queue.write_buffer(&buffer, 0, bytes);
    Ok(buffer)
}

fn create_storage_buffer(
    runtime: &Runtime,
    label: &'static str,
    size: u64,
    additional_usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer, WgpuError> {
    let size = size.max(4);
    ensure_buffer_size(&runtime.limits, size, label)?;
    let buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | additional_usage,
        mapped_at_creation: false,
    });
    runtime.poll_once()?;
    runtime.check_health()?;
    Ok(buffer)
}

fn binding<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    offset: u64,
    size: u64,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: wgpu::BufferSize::new(size),
        }),
    }
}

fn ensure_buffer_size(
    limits: &wgpu::Limits,
    requested: u64,
    resource: &'static str,
) -> Result<(), WgpuError> {
    let requested = requested.max(4);
    if requested > limits.max_buffer_size {
        return Err(WgpuError::ResourceExhausted {
            resource,
            requested,
            limit: limits.max_buffer_size,
        });
    }
    Ok(())
}

fn ensure_binding_size(
    limits: &wgpu::Limits,
    requested: u64,
    resource: &'static str,
) -> Result<(), WgpuError> {
    if requested > limits.max_storage_buffer_binding_size as u64 {
        return Err(WgpuError::ResourceExhausted {
            resource,
            requested,
            limit: limits.max_storage_buffer_binding_size as u64,
        });
    }
    Ok(())
}

fn element_bytes(count: usize, resource: &'static str) -> Result<u64, WgpuError> {
    count
        .checked_mul(size_of_u32())
        .map(|bytes| bytes.max(4) as u64)
        .ok_or(WgpuError::ResourceExhausted {
            resource,
            requested: u64::MAX,
            limit: u64::MAX - 1,
        })
}

fn ensure_indexable(count: usize, resource: &'static str) -> Result<(), WgpuError> {
    if count > u32::MAX as usize {
        return Err(WgpuError::ResourceExhausted {
            resource,
            requested: count as u64,
            limit: u32::MAX as u64,
        });
    }
    Ok(())
}

fn validate_flat_total(total: usize, limits: Option<CalculationLimits>) -> Result<(), WgpuError> {
    if let Some(limits) = limits {
        let flat_item_limit = limits.max_items.saturating_mul(2);
        if total > flat_item_limit {
            return Err(WgpuError::LimitExceeded {
                resource: "flat token",
                actual: total,
                limit: flat_item_limit,
            });
        }
    }
    Ok(())
}

fn select_params_words(
    unit_len: usize,
    global_base: usize,
    rule_count: usize,
    seed: u64,
    iteration: u64,
    ambiguous_rules: AmbiguousRulePolicy,
) -> [u32; 12] {
    let global_base = global_base as u64;
    [
        unit_len as u32,
        global_base as u32,
        (global_base >> 32) as u32,
        rule_count as u32,
        seed as u32,
        (seed >> 32) as u32,
        iteration as u32,
        (iteration >> 32) as u32,
        match ambiguous_rules {
            AmbiguousRulePolicy::Uniform => 0,
            AmbiguousRulePolicy::First | AmbiguousRulePolicy::Error => 1,
        },
        0,
        0,
        0,
    ]
}

fn index_overflow(resource: &'static str) -> WgpuError {
    WgpuError::OutOfMemory(format!(
        "host address space was exhausted while sizing {resource}"
    ))
}

fn host_allocation_error(error: std::collections::TryReserveError) -> WgpuError {
    WgpuError::OutOfMemory(format!("host allocation failed: {error}"))
}

const fn size_of_u32() -> usize {
    std::mem::size_of::<u32>()
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn lcm(left: usize, right: usize) -> usize {
    left / gcd(left, right) * right
}

fn pack_u32(values: &[u32]) -> Result<Vec<u8>, WgpuError> {
    let byte_len = values
        .len()
        .checked_mul(size_of_u32())
        .ok_or_else(|| index_overflow("host upload byte count"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .map_err(host_allocation_error)?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn unpack_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("u32 requires exactly four bytes"))
}

fn pack_rules(program: &FlatWgpuProgram) -> Result<Vec<u8>, WgpuError> {
    let word_count = program
        .rule_lhs
        .len()
        .checked_mul(4)
        .ok_or_else(|| index_overflow("packed rule table"))?;
    let mut words = Vec::new();
    words
        .try_reserve_exact(word_count)
        .map_err(host_allocation_error)?;
    for index in 0..program.rule_lhs.len() {
        words.push(program.rule_lhs[index]);
        words.push(program.rule_weights[index].to_bits());
        words.push(program.rule_offsets[index]);
        words.push(program.rule_lengths[index]);
    }
    // The shader declares a runtime array of 16-byte Rule records. Keep one
    // inert record for an axiom-only grammar so every backend can validate the
    // binding even though `rule_count == 0` prevents shader access.
    if words.is_empty() {
        words.try_reserve_exact(4).map_err(host_allocation_error)?;
        words.extend([0, 0, 0, 0]);
    }
    pack_u32(&words)
}

#[derive(Debug, Clone)]
struct FlatWgpuProgram {
    axiom: Vec<u32>,
    id_to_name: Vec<Identifier>,
    rule_lhs: Vec<u32>,
    rule_weights: Vec<f32>,
    rule_offsets: Vec<u32>,
    rule_lengths: Vec<u32>,
    rule_module_counts: Vec<u32>,
    rule_rhs: Vec<u32>,
}

impl FlatWgpuProgram {
    fn compile(grammar: &Grammar, float_width: FloatWidth) -> Result<Self, WgpuError> {
        if grammar.context_filter.is_some() {
            return Err(unsupported(
                "`ignore` and `only` context filters are not supported by the WGPU backend yet",
            ));
        }

        let mut symbols = SymbolTable::default();
        let axiom = encode_word(&grammar.axiom, &mut symbols, "axiom")?;
        let globals = resolve_global_bindings_with_width(&grammar.bindings, float_width).map_err(
            |error| unsupported(format!("global bindings could not be evaluated: {error}")),
        )?;
        let mut weight_modes = BTreeMap::<u32, bool>::new();
        let mut rule_lhs = Vec::new();
        let mut rule_weights = Vec::new();
        let mut rule_offsets = Vec::new();
        let mut rule_lengths = Vec::new();
        let mut rule_module_counts = Vec::new();
        let mut rule_rhs = Vec::new();
        let production_count = grammar.productions.len();
        rule_lhs
            .try_reserve_exact(production_count)
            .map_err(host_allocation_error)?;
        rule_weights
            .try_reserve_exact(production_count)
            .map_err(host_allocation_error)?;
        rule_offsets
            .try_reserve_exact(production_count)
            .map_err(host_allocation_error)?;
        rule_lengths
            .try_reserve_exact(production_count)
            .map_err(host_allocation_error)?;
        rule_module_counts
            .try_reserve_exact(production_count)
            .map_err(host_allocation_error)?;

        for production in &grammar.productions {
            if !production.center.arguments.is_empty() {
                return Err(unsupported(
                    "parametric productions are not supported by the WGPU backend yet",
                ));
            }
            if production.left.is_some() || production.right.is_some() {
                return Err(unsupported(
                    "context-sensitive productions are not supported by the WGPU backend yet",
                ));
            }
            if let Some(condition) = &production.condition {
                let applies =
                    evaluate_constant_expression_with_width(condition, &globals, float_width)
                        .and_then(Value::as_bool)
                        .map_err(|error| {
                            unsupported(format!(
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
                    return Err(unsupported(format!(
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
                    evaluate_constant_expression_with_width(expression, &globals, float_width)
                })
                .transpose()
                .map_err(|error| {
                    unsupported(format!(
                        "weight for module `{}` is not constant: {error}",
                        production.center.name
                    ))
                })?
                .map(Value::as_number)
                .transpose()
                .map_err(|error| {
                    unsupported(format!(
                        "weight for module `{}` is not numeric: {error}",
                        production.center.name
                    ))
                })?
                .unwrap_or(0.0);
            if weighted && (!weight.is_finite() || weight <= 0.0) {
                return Err(unsupported(format!(
                    "production weight for module `{}` evaluated to invalid value {weight}",
                    production.center.name
                )));
            }
            let weight = weight as f32;
            if weighted && (!weight.is_finite() || weight <= 0.0) {
                return Err(unsupported(format!(
                    "production weight for module `{}` cannot be represented by WGPU f32 arithmetic",
                    production.center.name
                )));
            }

            let rhs = encode_word(&production.successor, &mut symbols, "successor")?;
            ensure_indexable(rule_rhs.len(), "successor table offset")?;
            ensure_indexable(rhs.len(), "successor length")?;
            rule_lhs.push(lhs);
            rule_weights.push(weight);
            rule_offsets.push(rule_rhs.len() as u32);
            rule_lengths.push(rhs.len() as u32);
            rule_module_counts.push(
                u32::try_from(
                    rhs.iter()
                        .filter(|token| **token != BRANCH_OPEN && **token != BRANCH_CLOSE)
                        .count(),
                )
                .map_err(|_| index_overflow("successor module count"))?,
            );
            rule_rhs
                .try_reserve(rhs.len())
                .map_err(host_allocation_error)?;
            rule_rhs.extend(rhs);
            ensure_indexable(rule_rhs.len(), "successor table")?;
        }
        ensure_indexable(rule_lhs.len(), "rule table")?;

        if let Some(&wildcard_weighted) = weight_modes.get(&WILDCARD_RULE) {
            for (&lhs, &weighted) in &weight_modes {
                if lhs != WILDCARD_RULE && weighted != wildcard_weighted {
                    return Err(unsupported(
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
                .sum::<f32>();
            let weighted = weight_modes.get(&lhs).copied().unwrap_or(false)
                || weight_modes.get(&WILDCARD_RULE).copied().unwrap_or(false);
            if weighted && (!total.is_finite() || total <= 0.0) {
                return Err(unsupported(format!(
                    "stochastic production weights for module `{}` do not have a positive finite f32 sum",
                    symbols.names[lhs as usize]
                )));
            }
        }

        Ok(Self {
            axiom,
            id_to_name: symbols.names,
            rule_lhs,
            rule_weights,
            rule_offsets,
            rule_lengths,
            rule_module_counts,
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

    fn decode(
        &self,
        tokens: &[u32],
        limits: Option<CalculationLimits>,
    ) -> Result<Generation, WgpuError> {
        let mut decoder = GenerationDecoder::new(self, limits)?;
        decoder.extend(tokens)?;
        decoder.finish()
    }

    fn max_successor_tokens(&self) -> usize {
        self.rule_lengths.iter().copied().max().unwrap_or(1).max(1) as usize
    }
}

struct GenerationDecoder<'a> {
    program: &'a FlatWgpuProgram,
    limits: CalculationLimits,
    stack: Vec<Vec<GenerationItem>>,
    modules: usize,
    items: usize,
}

impl<'a> GenerationDecoder<'a> {
    fn new(
        program: &'a FlatWgpuProgram,
        limits: Option<CalculationLimits>,
    ) -> Result<Self, WgpuError> {
        let mut stack: Vec<Vec<GenerationItem>> = Vec::new();
        stack.try_reserve_exact(1).map_err(host_allocation_error)?;
        stack.push(Vec::new());
        Ok(Self {
            program,
            limits: limits.unwrap_or_default(),
            stack,
            modules: 0,
            items: 0,
        })
    }

    fn extend(&mut self, tokens: &[u32]) -> Result<(), WgpuError> {
        for &token in tokens {
            self.push(token)?;
        }
        Ok(())
    }

    fn push(&mut self, token: u32) -> Result<(), WgpuError> {
        match token {
            BRANCH_OPEN => {
                self.items = self
                    .items
                    .checked_add(1)
                    .ok_or_else(|| index_overflow("decoded item count"))?;
                check_limit("item", self.items, self.limits.max_items)?;
                let depth = self.stack.len();
                check_limit("branch depth", depth, self.limits.max_branch_depth)?;
                self.stack.try_reserve(1).map_err(host_allocation_error)?;
                self.stack.push(Vec::new());
            }
            BRANCH_CLOSE => {
                if self.stack.len() == 1 {
                    return Err(unsupported(
                        "WGPU output contained an unmatched branch close",
                    ));
                }
                let branch = Generation(self.stack.pop().expect("branch stack is non-empty"));
                let parent = self.stack.last_mut().expect("root remains");
                parent.try_reserve(1).map_err(host_allocation_error)?;
                parent.push(GenerationItem::Branch(branch));
            }
            symbol => {
                let name = self
                    .program
                    .id_to_name
                    .get(symbol as usize)
                    .ok_or_else(|| unsupported("WGPU output contained an unknown symbol"))?;
                self.modules = self
                    .modules
                    .checked_add(1)
                    .ok_or_else(|| index_overflow("decoded module count"))?;
                self.items = self
                    .items
                    .checked_add(1)
                    .ok_or_else(|| index_overflow("decoded item count"))?;
                check_limit("module", self.modules, self.limits.max_modules)?;
                check_limit("item", self.items, self.limits.max_items)?;
                let name = try_clone_identifier(name)?;
                let axis = self.stack.last_mut().expect("root remains");
                axis.try_reserve(1).map_err(host_allocation_error)?;
                axis.push(GenerationItem::Module(Module::new(name, Vec::new())));
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Generation, WgpuError> {
        if self.stack.len() != 1 {
            return Err(unsupported("WGPU output contained an unclosed branch"));
        }
        Ok(Generation(self.stack.pop().expect("root remains")))
    }
}

#[derive(Default)]
struct SymbolTable {
    names: Vec<Identifier>,
    ids: BTreeMap<Identifier, u32>,
}

impl SymbolTable {
    fn intern(&mut self, name: &Identifier) -> Result<u32, WgpuError> {
        match self.ids.entry(name.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                if self.names.len() >= WILDCARD_RULE as usize {
                    return Err(WgpuError::ResourceExhausted {
                        resource: "symbol table",
                        requested: self.names.len() as u64 + 1,
                        limit: WILDCARD_RULE as u64,
                    });
                }
                let id = self.names.len() as u32;
                self.names.push(name.clone());
                entry.insert(id);
                Ok(id)
            }
        }
    }
}

fn encode_word(word: &Word, symbols: &mut SymbolTable, role: &str) -> Result<Vec<u32>, WgpuError> {
    struct Frame<'a> {
        word: &'a Word,
        next: usize,
        close_after: bool,
    }

    let mut encoded = Vec::new();
    let mut stack = Vec::new();
    stack.try_reserve_exact(1).map_err(host_allocation_error)?;
    stack.push(Frame {
        word,
        next: 0,
        close_after: false,
    });

    while let Some(frame) = stack.last_mut() {
        if frame.next == frame.word.0.len() {
            let close_after = frame.close_after;
            stack.pop();
            if close_after {
                push_encoded_token(&mut encoded, BRANCH_CLOSE)?;
            }
            continue;
        }
        let item = &frame.word.0[frame.next];
        frame.next += 1;
        match item {
            WordItem::Module(module) => {
                if !module.arguments.is_empty() {
                    return Err(unsupported(format!(
                        "{role} module `{}` has parameters",
                        module.name
                    )));
                }
                let symbol = symbols.intern(&module.name)?;
                push_encoded_token(&mut encoded, symbol)?;
            }
            WordItem::Branch(branch) => {
                push_encoded_token(&mut encoded, BRANCH_OPEN)?;
                stack.try_reserve(1).map_err(host_allocation_error)?;
                stack.push(Frame {
                    word: branch,
                    next: 0,
                    close_after: true,
                });
            }
        }
    }
    Ok(encoded)
}

fn push_encoded_token(encoded: &mut Vec<u32>, token: u32) -> Result<(), WgpuError> {
    encoded.try_reserve(1).map_err(host_allocation_error)?;
    encoded.push(token);
    Ok(())
}

fn check_limit(resource: &'static str, actual: usize, limit: usize) -> Result<(), WgpuError> {
    if actual > limit {
        return Err(WgpuError::LimitExceeded {
            resource,
            actual,
            limit,
        });
    }
    Ok(())
}

fn try_clone_identifier(identifier: &Identifier) -> Result<Identifier, WgpuError> {
    let source = identifier.as_str();
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(source.len())
        .map_err(host_allocation_error)?;
    cloned.push_str(source);
    Ok(Identifier::new(cloned))
}

fn unsupported(reason: impl Into<String>) -> WgpuError {
    WgpuError::UnsupportedGrammar(reason.into())
}

struct CallbackSignal<T> {
    state: Arc<Mutex<CallbackState<T>>>,
}

impl<T> CallbackSignal<T> {
    fn complete(self, value: T) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.value = Some(value);
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }
}

struct CallbackFuture<T> {
    state: Arc<Mutex<CallbackState<T>>>,
}

struct CallbackState<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

impl<T> Future for CallbackFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(value) = state.value.take() {
            Poll::Ready(value)
        } else {
            state.waker = Some(context.waker().clone());
            Poll::Pending
        }
    }
}

fn callback_pair<T>() -> (CallbackSignal<T>, CallbackFuture<T>) {
    let state = Arc::new(Mutex::new(CallbackState {
        value: None,
        waker: None,
    }));
    (
        CallbackSignal {
            state: Arc::clone(&state),
        },
        CallbackFuture { state },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits(binding_size: u32, workgroups: u32, alignment: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: binding_size,
            max_compute_workgroups_per_dimension: workgroups,
            min_storage_buffer_offset_alignment: alignment,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn packs_rule_records_for_wgsl_layout() {
        let grammar = Grammar::parse(
            r#"
                axiom A;
                match A weight 3 then A B;
                match A weight 1 then C;
            "#,
        )
        .unwrap();
        let flat = FlatWgpuProgram::compile(&grammar, FloatWidth::F32).unwrap();
        let bytes = pack_rules(&flat).unwrap();

        assert_eq!(bytes.len(), 2 * 16);
        assert_eq!(unpack_u32(&bytes[0..4]), flat.rule_lhs[0]);
        assert_eq!(f32::from_bits(unpack_u32(&bytes[4..8])), 3.0);
        assert_eq!(unpack_u32(&bytes[8..12]), 0);
        assert_eq!(unpack_u32(&bytes[12..16]), 2);
        assert_eq!(flat.rule_module_counts, [2, 1]);
    }

    #[test]
    fn dispatch_chunks_respect_binding_alignment_and_workgroup_limit() {
        let limits = test_limits(4096, 2, 256);
        let chunks = plan_dispatch_chunks(1_500, &limits).unwrap();

        assert_eq!(chunks.iter().map(|chunk| chunk.len).sum::<usize>(), 1_500);
        assert!(chunks.iter().all(|chunk| chunk.workgroups <= 2));
        assert!(chunks.iter().all(|chunk| chunk.start % 256 == 0));
        assert!(chunks.iter().all(|chunk| chunk.len <= 512));
    }

    #[test]
    fn rewrite_plan_accounts_for_worst_case_growth() {
        let limits = test_limits(16 * 1024, 65_535, 256);
        let chunks = plan_rewrite_chunks(10_000, 4, &limits).unwrap();
        let output_capacity = limits.max_storage_buffer_binding_size as usize / 4;

        assert_eq!(chunks.iter().map(|chunk| chunk.len).sum::<usize>(), 10_000);
        assert!(chunks.iter().all(|chunk| chunk.len * 4 <= output_capacity));
    }

    #[test]
    fn dispatch_planning_shards_at_the_device_buffer_limit() {
        let mut limits = test_limits(16 * 1024, 65_535, 256);
        limits.max_buffer_size = 2 * 1024;
        let chunks = plan_dispatch_chunks(2_000, &limits).unwrap();

        assert!(chunks.len() > 1);
        assert_eq!(chunks.iter().map(|chunk| chunk.len).sum::<usize>(), 2_000);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.len * size_of_u32() <= limits.max_buffer_size as usize)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn logical_dispatch_plan_crosses_the_u32_boundary_without_allocating_chunks() {
        let limits = test_limits(16 * 1024, 65_535, 256);
        let logical_len = u32::MAX as usize + 12_345;
        let plan = plan_dispatch_chunks(logical_len, &limits).unwrap();

        let crossing_index = (u32::MAX as usize / plan.regular_len) + 1;
        let crossing = plan.get(crossing_index).unwrap();
        let last = plan.get(plan.len() - 1).unwrap();

        assert!(crossing.start > u32::MAX as usize);
        assert_eq!(last.start.checked_add(last.len), Some(logical_len));
        assert!(validate_flat_total(logical_len, None).is_ok());
    }

    #[test]
    fn decode_enforces_depth_before_building_an_oversized_tree() {
        let grammar = Grammar::parse("axiom A;").unwrap();
        let flat = FlatWgpuProgram::compile(&grammar, FloatWidth::F32).unwrap();
        let mut tokens = vec![BRANCH_OPEN; 10_000];
        tokens.push(flat.axiom[0]);
        tokens.extend(std::iter::repeat_n(BRANCH_CLOSE, 10_000));

        let error = flat
            .decode(
                &tokens,
                Some(CalculationLimits {
                    max_modules: usize::MAX,
                    max_items: usize::MAX,
                    max_branch_depth: 64,
                }),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            WgpuError::LimitExceeded {
                resource: "branch depth",
                actual: 65,
                limit: 64,
            }
        ));
    }

    #[test]
    fn decoder_preserves_branches_across_downloaded_shards() {
        let grammar = Grammar::parse("axiom A [ B [ C ] ] D;").unwrap();
        let flat = FlatWgpuProgram::compile(&grammar, FloatWidth::F32).unwrap();
        let mut decoder = GenerationDecoder::new(&flat, None).unwrap();

        for token in &flat.axiom {
            decoder.extend(std::slice::from_ref(token)).unwrap();
        }

        assert_eq!(decoder.finish().unwrap().to_string(), "A [ B [ C ] ] D");
    }

    #[test]
    fn flat_compiler_rejects_gpu_incompatible_grammar_without_an_adapter() {
        let grammar = Grammar::parse(
            r#"
                axiom F(1);
                match F(x) then F(x + 1);
            "#,
        )
        .unwrap();
        assert!(matches!(
            FlatWgpuProgram::compile(&grammar, FloatWidth::F32),
            Err(WgpuError::UnsupportedGrammar(_))
        ));
    }

    #[test]
    fn flat_program_detects_only_present_unweighted_ambiguities() {
        let grammar = Grammar::parse(
            "axiom A C; match A then B; match A then C; match B then A; match B then C;",
        )
        .unwrap();
        let flat = FlatWgpuProgram::compile(&grammar, FloatWidth::F32).unwrap();
        assert_eq!(
            flat.first_unweighted_ambiguity(&flat.axiom),
            Some((String::from("A"), 2))
        );
    }

    #[test]
    fn position_key_uses_both_halves_and_not_chunk_coordinates() {
        // Keep the host reference synchronized with wgpu_select.wgsl. Its
        // inputs deliberately contain no local/chunk index.
        fn hash_word(mut value: u32) -> u32 {
            value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
            value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
            value ^ (value >> 16)
        }
        fn random_bits(seed: u64, iteration: u64, position: u64) -> u32 {
            let mut key = seed as u32 ^ hash_word(((seed >> 32) as u32).wrapping_add(0x9e37_79b9));
            key ^= hash_word((iteration as u32).wrapping_add(0x85eb_ca6b));
            key ^= hash_word(((iteration >> 32) as u32).wrapping_add(0xc2b2_ae35));
            key ^= hash_word((position as u32).wrapping_add(0x27d4_eb2f));
            key ^= hash_word((position >> 32) as u32);
            hash_word(key) >> 8
        }

        let expected = random_bits(42, 7, 12_345);
        assert_eq!(expected, random_bits(42, 7, 5_000 + 7_345));
        assert_ne!(expected, random_bits(42, 7, 12_346));
        assert_ne!(expected, random_bits(42, 7, (1_u64 << 32) + 12_345));

        // Mirror the shader's low-half addition and carry at a unit boundary.
        let base = u32::MAX as u64 - 7;
        let local_index = 10_u32;
        let base_lo = base as u32;
        let position_lo = base_lo.wrapping_add(local_index);
        let carry = u32::from(position_lo < base_lo);
        let position_hi = ((base >> 32) as u32).wrapping_add(carry);
        let reconstructed = (u64::from(position_hi) << 32) | u64::from(position_lo);
        assert_eq!(reconstructed, base + u64::from(local_index));

        #[cfg(target_pointer_width = "64")]
        {
            let params = select_params_words(
                512,
                u32::MAX as usize + 514,
                3,
                0x0123_4567_89ab_cdef,
                0xfedc_ba98_7654_3210,
                AmbiguousRulePolicy::Uniform,
            );
            assert_eq!(
                params,
                [
                    512,
                    513,
                    1,
                    3,
                    0x89ab_cdef,
                    0x0123_4567,
                    0x7654_3210,
                    0xfedc_ba98,
                    0,
                    0,
                    0,
                    0,
                ]
            );
        }
    }

    #[test]
    fn optional_headless_compute_smoke_test() {
        if std::env::var_os("BRAKEN_WGPU_TEST").is_none() {
            return;
        }
        pollster::block_on(async {
            let backend = WgpuBackend::request_with_options(WgpuBackendOptions {
                allow_software_adapter: true,
            })
            .await
            .unwrap();
            let axiom_only = Grammar::parse("axiom A;").unwrap();
            assert_eq!(backend.run(&axiom_only, 3).await.unwrap().to_string(), "A");
            let deletion = Grammar::parse("axiom A; match A then nothing;").unwrap();
            assert!(backend.run(&deletion, 1).await.unwrap().0.is_empty());
            let deletion_program = backend.compile(&deletion).unwrap();
            let mut deletion_state = deletion_program.start().unwrap();
            let deletion_lineage = deletion_state
                .step_with_lineage_and_cancel(&mut || false)
                .await
                .unwrap();
            assert_eq!(deletion_lineage.successor_modules_per_input(), [0]);

            let branched = Grammar::parse("axiom A [ B ]; match A then C [ D ];").unwrap();
            let branched_program = backend.compile(&branched).unwrap();
            let mut branched_state = branched_program.start().unwrap();
            let branched_lineage = branched_state
                .step_with_lineage_and_cancel(&mut || false)
                .await
                .unwrap();
            assert_eq!(branched_lineage.successor_modules_per_input(), [2, 1]);

            let stochastic =
                Grammar::parse("axiom A; match A weight 1 then B; match A weight 1 then C D E;")
                    .unwrap();
            let stochastic_program = backend.compile(&stochastic).unwrap();
            for seed in 0..16 {
                let mut stochastic_state = stochastic_program.start_with_seed(seed).unwrap();
                let lineage = stochastic_state
                    .step_with_lineage_and_cancel(&mut || false)
                    .await
                    .unwrap();
                let generation = stochastic_state.generation().await.unwrap();
                assert_eq!(lineage.output_modules(), Some(generation.module_count()));
            }

            let ambiguous = Grammar::parse("axiom A; match A then B; match A then C;").unwrap();
            assert_eq!(
                backend
                    .clone()
                    .ambiguous_rules(AmbiguousRulePolicy::First)
                    .run(&ambiguous, 1)
                    .await
                    .unwrap()
                    .to_string(),
                "B"
            );
            assert!(matches!(
                backend
                    .clone()
                    .ambiguous_rules(AmbiguousRulePolicy::Error)
                    .run(&ambiguous, 1)
                    .await,
                Err(WgpuError::AmbiguousRules { count: 2, .. })
            ));

            let branched_stochastic = Grammar::parse(
                "axiom A [ A ] A; match A weight 1 then B; match A weight 1 then C D;",
            )
            .unwrap();
            let cpu = crate::grammar::CpuBackend::new()
                .float_width(FloatWidth::F32)
                .compile(&branched_stochastic)
                .unwrap();
            for seed in 0..16 {
                assert_eq!(
                    backend
                        .run_with_seed(&branched_stochastic, 2, seed)
                        .await
                        .unwrap(),
                    cpu.run_with_seed(2, seed).unwrap(),
                    "seed {seed} must select the same f32 successors"
                );
            }

            let grammar = Grammar::parse("axiom A; match A then A A;").unwrap();
            let program = backend.compile(&grammar).unwrap();
            let mut state = program.start().unwrap();
            state.advance(10).await.unwrap();
            let mut cancellation_checks = 0;
            let error = state
                .step_with_cancel(&mut || {
                    cancellation_checks += 1;
                    cancellation_checks >= 16
                })
                .await
                .unwrap_err();
            assert_eq!(error, WgpuError::Cancelled);
            assert_eq!(state.generation_index(), 10);
            assert_eq!(state.token_count(), 1_024);

            state.step().await.unwrap();
            assert!(state.generation.chunks.len() > 1);
            let result = state.into_generation().await.unwrap();
            assert_eq!(result.module_count(), 2_048);
        });
    }
}
