use std::{mem::MaybeUninit, sync::Arc};

use braken::GenerationItem;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, IntoResult, LaunchConfig1D, sys,
};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use super::turtle_2d::{
    EncodedOp, HeadingTransform, MotionTransform, OP_DRAW, OP_HALF_MOVE, OP_MOVE, OP_WIDTH_DELTA,
    OP_WIDTH_SET, WidthTransform, apply_width, checked_increment, classify_module, compose_heading,
    compose_motion, deliver_batch, heading_transform, include_bounds, new_line_batch,
    new_module_index_batch, new_module_position_batch, resource_error,
};
use crate::{
    IndexedModulePosition2d, Line2d, StyledLine2d, Turtle2dBatch, Turtle2dProgress,
    Turtle2dStreamRequest, Turtle2dStreamSummary, VisualizeError, VisualizerBackend,
    VisualizerKind,
};

unsafe extern "C" {
    fn __nv_sin(value: f64) -> f64;
    fn __nv_cos(value: f64) -> f64;
}

// This bounds temporary device residency and cancellation latency, not the size
// of a generation. Consecutive groups carry exact turtle state forward.
#[cfg(not(test))]
const CUDA_LATENCY_CEILING: usize = 256 * 1024;
#[cfg(test)]
const CUDA_LATENCY_CEILING: usize = 512;
const CUDA_THREADS: u32 = 256;
const CUDA_MEMORY_FRACTION_DIVISOR: usize = 5;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, DeviceCopy)]
struct GpuOutput {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    line_width: f64,
    angle_after: f64,
    scale_after: f64,
    width_after: f64,
    invert_after: u32,
    draw: u32,
}

#[derive(Debug, Clone, Copy)]
struct TurtleCarry {
    x: f64,
    y: f64,
    angle: f64,
    scale: f64,
    width: f64,
    invert: u32,
}

impl TurtleCarry {
    fn initial(request: &Turtle2dStreamRequest<'_>) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            angle: request.config.initial_angle,
            scale: 1.0,
            width: request.config.initial_width,
            invert: 0,
        }
    }

    fn update_from(&mut self, output: GpuOutput) {
        self.x = output.x1;
        self.y = output.y1;
        self.angle = output.angle_after;
        self.scale = output.scale_after;
        self.width = output.width_after;
        self.invert = output.invert_after;
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn encode_heading(ops: &[EncodedOp], mut output: DisjointSlice<HeadingTransform>) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = heading_transform(ops[raw]);
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scan_heading_stage(
        input: &[HeadingTransform],
        stride: u32,
        mut output: DisjointSlice<HeadingTransform>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = if raw < stride as usize {
                input[raw]
            } else {
                compose_heading(input[raw - stride as usize], input[raw])
            };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn encode_motion(
        ops: &[EncodedOp],
        heading_prefix: &[HeadingTransform],
        base_angle: f64,
        base_scale: f64,
        base_invert: u32,
        mut output: DisjointSlice<MotionTransform>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(slot) = output.get_mut(index) {
            let before = if raw == 0 {
                HeadingTransform::IDENTITY
            } else {
                heading_prefix[raw - 1]
            };
            let angle = base_angle
                + if base_invert == 0 {
                    before.normal_delta
                } else {
                    before.inverted_delta
                };
            let scale = base_scale * before.scale;
            let op = ops[raw];
            let mut transform = MotionTransform::IDENTITY;
            if op.tag == OP_DRAW || op.tag == OP_MOVE {
                let distance = op.value * scale;
                transform.dx = distance * unsafe { __nv_cos(angle) };
                transform.dy = distance * unsafe { __nv_sin(angle) };
            } else if op.tag == OP_HALF_MOVE {
                let distance = op.value * scale * 0.5;
                transform.dx = distance * unsafe { __nv_cos(angle) };
                transform.dy = distance * unsafe { __nv_sin(angle) };
            } else if op.tag == OP_WIDTH_SET {
                transform.width = WidthTransform::set(op.value);
            } else if op.tag == OP_WIDTH_DELTA {
                transform.width = WidthTransform::adjust(op.value);
            }
            *slot = transform;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn scan_motion_stage(
        input: &[MotionTransform],
        stride: u32,
        mut output: DisjointSlice<MotionTransform>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = if raw < stride as usize {
                input[raw]
            } else {
                compose_motion(input[raw - stride as usize], input[raw])
            };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn emit_geometry(
        ops: &[EncodedOp],
        heading_prefix: &[HeadingTransform],
        motion_prefix: &[MotionTransform],
        base_x: f64,
        base_y: f64,
        base_angle: f64,
        base_scale: f64,
        base_width: f64,
        base_invert: u32,
        mut output: DisjointSlice<GpuOutput>,
    ) {
        let index = thread::index_1d();
        let raw = index.get();
        if let Some(slot) = output.get_mut(index) {
            let motion_before = if raw == 0 {
                MotionTransform::IDENTITY
            } else {
                motion_prefix[raw - 1]
            };
            let motion_after = motion_prefix[raw];
            let heading_after = heading_prefix[raw];
            *slot = GpuOutput {
                x0: base_x + motion_before.dx,
                y0: base_y + motion_before.dy,
                x1: base_x + motion_after.dx,
                y1: base_y + motion_after.dy,
                line_width: apply_width(motion_before.width, base_width),
                angle_after: base_angle
                    + if base_invert == 0 {
                        heading_after.normal_delta
                    } else {
                        heading_after.inverted_delta
                    },
                scale_after: base_scale * heading_after.scale,
                width_after: apply_width(motion_after.width, base_width),
                invert_after: base_invert ^ heading_after.invert_xor,
                draw: ops[raw].draw,
            };
        }
    }
}

pub(crate) struct CudaTurtleRuntime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    scratch: Option<CudaScratch>,
}

struct CudaScratch {
    len: usize,
    input: DeviceBuffer<EncodedOp>,
    heading_a: DeviceBuffer<HeadingTransform>,
    heading_b: DeviceBuffer<HeadingTransform>,
    motion_a: DeviceBuffer<MotionTransform>,
    motion_b: DeviceBuffer<MotionTransform>,
    output: DeviceBuffer<GpuOutput>,
}

#[derive(Debug)]
pub(crate) struct CudaStreamFailure {
    pub(crate) error: VisualizeError,
    pub(crate) started: bool,
}

impl CudaStreamFailure {
    fn new(error: VisualizeError, started: bool) -> Self {
        Self { error, started }
    }
}

impl CudaTurtleRuntime {
    pub(crate) fn new() -> Result<Self, VisualizeError> {
        let context = CudaContext::new(0).map_err(|_error| VisualizeError::BackendUnavailable {
            backend: VisualizerBackend::Cuda,
        })?;
        let stream = context.default_stream();
        // SAFETY: this crate owns the embedded device bundle generated for this
        // kernels module and loads it into the matching context.
        let module = unsafe {
            kernels::load(&context)
                .map_err(|error| cuda_error("loading turtle kernels", error.to_string()))?
        };
        Ok(Self {
            context,
            stream,
            module,
            scratch: None,
        })
    }

    fn group_size(&self, remaining: usize) -> Result<usize, VisualizeError> {
        if remaining == 0 {
            return Ok(0);
        }
        self.context
            .bind_to_thread()
            .map_err(|error| cuda_error("binding the CUDA context", error.to_string()))?;
        let (free_memory, _) = memory_info()?;
        let bytes_per_item = std::mem::size_of::<EncodedOp>()
            + 2 * std::mem::size_of::<HeadingTransform>()
            + 2 * std::mem::size_of::<MotionTransform>()
            + std::mem::size_of::<GpuOutput>();
        // `free_memory` excludes retained reusable scratch. Add that exact
        // residency back before replanning so equal workloads do not
        // geometrically shrink their group on every request.
        let retained_bytes = self
            .scratch
            .as_ref()
            .and_then(|scratch| scratch.len.checked_mul(bytes_per_item))
            .unwrap_or(0);
        let effective_free = free_memory.saturating_add(retained_bytes);
        let memory_items = (effective_free / CUDA_MEMORY_FRACTION_DIVISOR) / bytes_per_item;
        let limits = self
            .context
            .launch_limits()
            .map_err(|error| cuda_error("querying CUDA launch limits", error.to_string()))?;
        let launch_items = (limits.max_grid_dim().0 as usize).saturating_mul(CUDA_THREADS as usize);
        let planned = remaining
            .min(CUDA_LATENCY_CEILING)
            .min(u32::MAX as usize)
            .min(launch_items)
            .min(memory_items);
        if planned == 0 {
            return Err(VisualizeError::ResourceExhausted {
                resource: "CUDA turtle scratch memory",
                requested: Some(bytes_per_item),
            });
        }
        Ok(planned)
    }

    fn ensure_scratch(&mut self, len: usize) -> Result<(), VisualizeError> {
        if self
            .scratch
            .as_ref()
            .is_some_and(|scratch| scratch.len == len)
        {
            return Ok(());
        }
        // A differently sized request cannot use DeviceBuffer slicing. Drop
        // the old group before allocating the replacement to avoid transient
        // double residency; equal-sized groups retain and refill all buffers.
        drop(self.scratch.take());
        let input = DeviceBuffer::<EncodedOp>::zeroed(&self.stream, len).map_err(|error| {
            cuda_allocation_error(
                "allocating turtle command input",
                "CUDA turtle command input",
                len,
                error,
            )
        })?;
        let heading_a =
            DeviceBuffer::<HeadingTransform>::zeroed(&self.stream, len).map_err(|error| {
                cuda_allocation_error(
                    "allocating heading scan input",
                    "CUDA heading scan input",
                    len,
                    error,
                )
            })?;
        let heading_b =
            DeviceBuffer::<HeadingTransform>::zeroed(&self.stream, len).map_err(|error| {
                cuda_allocation_error(
                    "allocating heading scan scratch",
                    "CUDA heading scan scratch",
                    len,
                    error,
                )
            })?;
        let motion_a =
            DeviceBuffer::<MotionTransform>::zeroed(&self.stream, len).map_err(|error| {
                cuda_allocation_error(
                    "allocating motion scan input",
                    "CUDA motion scan input",
                    len,
                    error,
                )
            })?;
        let motion_b =
            DeviceBuffer::<MotionTransform>::zeroed(&self.stream, len).map_err(|error| {
                cuda_allocation_error(
                    "allocating motion scan scratch",
                    "CUDA motion scan scratch",
                    len,
                    error,
                )
            })?;
        let output = DeviceBuffer::<GpuOutput>::zeroed(&self.stream, len).map_err(|error| {
            cuda_allocation_error(
                "allocating turtle geometry output",
                "CUDA turtle geometry output",
                len,
                error,
            )
        })?;
        self.scratch = Some(CudaScratch {
            len,
            input,
            heading_a,
            heading_b,
            motion_a,
            motion_b,
            output,
        });
        Ok(())
    }

    fn run_chunk(
        &mut self,
        ops: &[EncodedOp],
        carry: TurtleCarry,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<GpuOutput>, CudaStreamFailure> {
        debug_assert!(!ops.is_empty());
        check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, false))?;
        self.context.bind_to_thread().map_err(|error| {
            CudaStreamFailure::new(
                cuda_error("binding the CUDA context", error.to_string()),
                false,
            )
        })?;
        self.ensure_scratch(ops.len())
            .map_err(|error| CudaStreamFailure::new(error, false))?;
        let launch =
            || LaunchConfig1D::new((ops.len() as u32).div_ceil(CUDA_THREADS), CUDA_THREADS, 0);
        let encode_heading_launch =
            self.module
                .prepare_encode_heading(launch())
                .map_err(|error| {
                    CudaStreamFailure::new(
                        cuda_error("preparing heading encoding", error.to_string()),
                        false,
                    )
                })?;
        let scan_heading_launch =
            self.module
                .prepare_scan_heading_stage(launch())
                .map_err(|error| {
                    CudaStreamFailure::new(
                        cuda_error("preparing heading scan", error.to_string()),
                        false,
                    )
                })?;
        let encode_motion_launch =
            self.module
                .prepare_encode_motion(launch())
                .map_err(|error| {
                    CudaStreamFailure::new(
                        cuda_error("preparing motion encoding", error.to_string()),
                        false,
                    )
                })?;
        let scan_motion_launch =
            self.module
                .prepare_scan_motion_stage(launch())
                .map_err(|error| {
                    CudaStreamFailure::new(
                        cuda_error("preparing motion scan", error.to_string()),
                        false,
                    )
                })?;
        let emit_geometry_launch =
            self.module
                .prepare_emit_geometry(launch())
                .map_err(|error| {
                    CudaStreamFailure::new(
                        cuda_error("preparing geometry emission", error.to_string()),
                        false,
                    )
                })?;

        check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, false))?;
        let scratch = self.scratch.as_mut().expect("CUDA scratch prepared");
        scratch
            .input
            .copy_from_host(&self.stream, ops)
            .map_err(|error| {
                CudaStreamFailure::new(
                    cuda_error("uploading turtle commands", error.to_string()),
                    false,
                )
            })?;
        check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, false))?;

        self.module
            .encode_heading(
                &self.stream,
                &encode_heading_launch,
                &scratch.input,
                &mut scratch.heading_a,
            )
            .map_err(|error| {
                CudaStreamFailure::new(
                    cuda_error("encoding heading transforms", error.to_string()),
                    true,
                )
            })?;

        let mut heading_in_a = true;
        let mut stride = 1usize;
        while stride < ops.len() {
            check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, true))?;
            if heading_in_a {
                self.module
                    .scan_heading_stage(
                        &self.stream,
                        &scan_heading_launch,
                        &scratch.heading_a,
                        stride as u32,
                        &mut scratch.heading_b,
                    )
                    .map_err(|error| {
                        CudaStreamFailure::new(
                            cuda_error("scanning heading transforms", error.to_string()),
                            true,
                        )
                    })?;
            } else {
                self.module
                    .scan_heading_stage(
                        &self.stream,
                        &scan_heading_launch,
                        &scratch.heading_b,
                        stride as u32,
                        &mut scratch.heading_a,
                    )
                    .map_err(|error| {
                        CudaStreamFailure::new(
                            cuda_error("scanning heading transforms", error.to_string()),
                            true,
                        )
                    })?;
            }
            heading_in_a = !heading_in_a;
            stride = stride.checked_mul(2).ok_or_else(|| {
                CudaStreamFailure::new(resource_error("CUDA heading scan stride", usize::MAX), true)
            })?;
        }
        let heading_prefix = if heading_in_a {
            &scratch.heading_a
        } else {
            &scratch.heading_b
        };

        check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, true))?;
        self.module
            .encode_motion(
                &self.stream,
                &encode_motion_launch,
                &scratch.input,
                heading_prefix,
                carry.angle,
                carry.scale,
                carry.invert,
                &mut scratch.motion_a,
            )
            .map_err(|error| {
                CudaStreamFailure::new(
                    cuda_error("encoding turtle motion", error.to_string()),
                    true,
                )
            })?;

        let mut motion_in_a = true;
        stride = 1;
        while stride < ops.len() {
            check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, true))?;
            if motion_in_a {
                self.module
                    .scan_motion_stage(
                        &self.stream,
                        &scan_motion_launch,
                        &scratch.motion_a,
                        stride as u32,
                        &mut scratch.motion_b,
                    )
                    .map_err(|error| {
                        CudaStreamFailure::new(
                            cuda_error("scanning turtle motion", error.to_string()),
                            true,
                        )
                    })?;
            } else {
                self.module
                    .scan_motion_stage(
                        &self.stream,
                        &scan_motion_launch,
                        &scratch.motion_b,
                        stride as u32,
                        &mut scratch.motion_a,
                    )
                    .map_err(|error| {
                        CudaStreamFailure::new(
                            cuda_error("scanning turtle motion", error.to_string()),
                            true,
                        )
                    })?;
            }
            motion_in_a = !motion_in_a;
            stride = stride.checked_mul(2).ok_or_else(|| {
                CudaStreamFailure::new(resource_error("CUDA motion scan stride", usize::MAX), true)
            })?;
        }
        let motion_prefix = if motion_in_a {
            &scratch.motion_a
        } else {
            &scratch.motion_b
        };

        check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, true))?;
        self.module
            .emit_geometry(
                &self.stream,
                &emit_geometry_launch,
                &scratch.input,
                heading_prefix,
                motion_prefix,
                carry.x,
                carry.y,
                carry.angle,
                carry.scale,
                carry.width,
                carry.invert,
                &mut scratch.output,
            )
            .map_err(|error| {
                CudaStreamFailure::new(
                    cuda_error("emitting turtle geometry", error.to_string()),
                    true,
                )
            })?;
        let mut host_output = Vec::new();
        host_output.try_reserve_exact(ops.len()).map_err(|_| {
            CudaStreamFailure::new(resource_error("CUDA turtle host output", ops.len()), true)
        })?;
        host_output.resize(ops.len(), GpuOutput::default());
        scratch
            .output
            .copy_to_host(&self.stream, &mut host_output)
            .map_err(|error| {
                CudaStreamFailure::new(
                    cuda_error("downloading turtle geometry", error.to_string()),
                    true,
                )
            })?;

        check_cancelled(is_cancelled).map_err(|error| CudaStreamFailure::new(error, true))?;
        Ok(host_output)
    }
}

pub(crate) fn stream_cuda(
    runtime: &mut CudaTurtleRuntime,
    request: &Turtle2dStreamRequest<'_>,
    emit: &mut impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle2dStreamSummary, CudaStreamFailure> {
    let mut started = false;
    let mut carry = TurtleCarry::initial(request);
    let mut lines = new_line_batch(
        request
            .batch_size
            .min(request.generation.items().len().max(1)),
    )
    .map_err(|error| CudaStreamFailure::new(error, false))?;
    let mut polygons = Vec::new();
    let mut module_indices = new_module_index_batch(
        request
            .batch_size
            .min(request.generation.items().len().max(1)),
    )
    .map_err(|error| CudaStreamFailure::new(error, false))?;
    let mut module_positions = new_module_position_batch(
        request
            .batch_size
            .min(request.generation.items().len().max(1)),
    )
    .map_err(|error| CudaStreamFailure::new(error, false))?;
    let mut batch_bounds = None;
    let mut overall_bounds = None;
    let mut progress = Turtle2dProgress::default();
    let mut batch_items = 0usize;
    let group_size = runtime
        .group_size(request.generation.items().len())
        .map_err(|error| CudaStreamFailure::new(error, false))?;

    for items in request.generation.items().chunks(group_size.max(1)) {
        check_cancelled(request.is_cancelled)
            .map_err(|error| CudaStreamFailure::new(error, started))?;
        let mut ops = Vec::new();
        ops.try_reserve_exact(group_size).map_err(|_| {
            CudaStreamFailure::new(
                resource_error("CUDA turtle command group", group_size),
                started,
            )
        })?;
        for item in items {
            check_cancelled(request.is_cancelled)
                .map_err(|error| CudaStreamFailure::new(error, started))?;
            let GenerationItem::Module(module) = item else {
                return Err(CudaStreamFailure::new(
                    VisualizeError::UnsupportedInput {
                        visualizer: VisualizerKind::Turtle2d,
                        backend: VisualizerBackend::Cuda,
                        reason: "branch state save/restore requires the CPU turtle backend".into(),
                    },
                    started,
                ));
            };
            let op = classify_module(module, &request.config)
                .map_err(|error| CudaStreamFailure::new(error, started))?;
            let op = match op {
                super::turtle_2d::TurtleOp::TurnDefault(direction) => {
                    super::turtle_2d::TurtleOp::Turn(request.config.turn_angle * direction)
                }
                op => op,
            };
            ops.push(op.into());
        }
        ops.resize(group_size, EncodedOp::default());

        let output = runtime.run_chunk(&ops, carry, request.is_cancelled)?;
        started = true;
        debug_assert_eq!(output.len(), group_size);
        for item in output.iter().copied().take(items.len()) {
            check_cancelled(request.is_cancelled)
                .map_err(|error| CudaStreamFailure::new(error, true))?;
            validate_output(item).map_err(|error| CudaStreamFailure::new(error, true))?;
            progress.items_processed =
                checked_increment(progress.items_processed, "turtle processed-item counter")
                    .map_err(|error| CudaStreamFailure::new(error, true))?;
            let module_index = progress.modules_processed;
            progress.modules_processed = checked_increment(
                progress.modules_processed,
                "turtle processed-module counter",
            )
            .map_err(|error| CudaStreamFailure::new(error, true))?;
            module_positions.push(IndexedModulePosition2d {
                module_index,
                position: (item.x0, item.y0),
            });

            if item.draw != 0 {
                let line = StyledLine2d {
                    line: Line2d((item.x0, item.y0), (item.x1, item.y1)),
                    width: item.line_width,
                    color: request.config.initial_color,
                };
                include_bounds(&mut batch_bounds, line.line);
                include_bounds(&mut overall_bounds, line.line);
                lines.push(line);
                module_indices.push(module_index);
                progress.lines_emitted =
                    checked_increment(progress.lines_emitted, "turtle emitted-line counter")
                        .map_err(|error| CudaStreamFailure::new(error, true))?;
            }

            batch_items = checked_increment(batch_items, "turtle batch work counter")
                .map_err(|error| CudaStreamFailure::new(error, true))?;
            if batch_items == request.batch_size {
                deliver_batch(
                    request,
                    emit,
                    &mut lines,
                    &mut polygons,
                    &mut module_indices,
                    &mut module_positions,
                    &mut batch_bounds,
                    overall_bounds,
                    progress,
                    true,
                )
                .map_err(|error| CudaStreamFailure::new(error, true))?;
                batch_items = 0;
            }
        }
        if let Some(last) = output.get(items.len().saturating_sub(1)).copied() {
            carry.update_from(last);
        }
    }

    if batch_items != 0 {
        deliver_batch(
            request,
            emit,
            &mut lines,
            &mut polygons,
            &mut module_indices,
            &mut module_positions,
            &mut batch_bounds,
            overall_bounds,
            progress,
            false,
        )
        .map_err(|error| CudaStreamFailure::new(error, started))?;
    }
    Ok(Turtle2dStreamSummary {
        bounds: overall_bounds,
        progress,
        backend_used: VisualizerBackend::Cuda,
    })
}

fn validate_output(output: GpuOutput) -> Result<(), VisualizeError> {
    if output.x0.is_finite()
        && output.y0.is_finite()
        && output.x1.is_finite()
        && output.y1.is_finite()
        && output.line_width.is_finite()
        && output.line_width >= 0.0
        && output.angle_after.is_finite()
        && output.scale_after.is_finite()
        && output.width_after.is_finite()
        && output.width_after >= 0.0
        && output.invert_after <= 1
        && output.draw <= 1
    {
        Ok(())
    } else {
        Err(VisualizeError::InvalidConfiguration(
            "turtle_2d produced non-finite or invalid CUDA state".into(),
        ))
    }
}

fn check_cancelled(is_cancelled: &dyn Fn() -> bool) -> Result<(), VisualizeError> {
    if is_cancelled() {
        Err(VisualizeError::Cancelled)
    } else {
        Ok(())
    }
}

fn memory_info() -> Result<(usize, usize), VisualizeError> {
    let mut free = MaybeUninit::<usize>::uninit();
    let mut total = MaybeUninit::<usize>::uninit();
    unsafe {
        sys::cuMemGetInfo_v2(free.as_mut_ptr(), total.as_mut_ptr())
            .result()
            .map_err(|error| cuda_error("querying CUDA memory", error.to_string()))?;
        Ok((free.assume_init(), total.assume_init()))
    }
}

fn cuda_error(operation: &'static str, reason: String) -> VisualizeError {
    VisualizeError::BackendRuntime {
        backend: VisualizerBackend::Cuda,
        operation,
        reason,
    }
}

fn cuda_allocation_error(
    operation: &'static str,
    resource: &'static str,
    requested: usize,
    error: cuda_core::DriverError,
) -> VisualizeError {
    if error.0 == sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY {
        resource_error(resource, requested)
    } else {
        cuda_error(operation, error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use braken::{Generation, Module, Value};

    #[test]
    fn cuda_out_of_memory_is_a_typed_resource_error() {
        let error = cuda_allocation_error(
            "allocating test scratch",
            "CUDA test scratch",
            123,
            cuda_core::DriverError(sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY),
        );
        assert_eq!(
            error,
            VisualizeError::ResourceExhausted {
                resource: "CUDA test scratch",
                requested: Some(123),
            }
        );
    }

    #[test]
    fn cuda_stream_matches_cpu_when_a_device_is_available() {
        let mut runtime = match CudaTurtleRuntime::new() {
            Ok(runtime) => runtime,
            Err(VisualizeError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA initialization failure: {error}"),
        };
        let generation = Generation(
            (0..513)
                .map(|index| {
                    let module = match index % 9 {
                        0 => Module::new("F", vec![Value::Number(0.75)]),
                        1 => Module::new("Plus", vec![Value::Number(17.0)]),
                        2 => Module::new("Scale", vec![Value::Number(1.001)]),
                        3 => Module::new("InvertTurns", Vec::new()),
                        4 => Module::new("WidthIncrease", Vec::new()),
                        5 => Module::new("f", Vec::new()),
                        6 => Module::new("Minus", vec![Value::Number(8.0)]),
                        7 => Module::new("WidthDecrease", Vec::new()),
                        _ => Module::new("F", Vec::new()),
                    };
                    GenerationItem::Module(module)
                })
                .collect(),
        );
        let config = crate::Turtle2dConfig {
            initial_angle: 0.125,
            width_increment: 0.2,
            ..Default::default()
        };
        let mut expected = Vec::new();
        let mut expected_positions = Vec::new();
        super::super::turtle_2d::stream_cpu(
            Turtle2dStreamRequest {
                generation: &generation,
                config: config.clone(),
                batch_size: 37,
                is_cancelled: &|| false,
            },
            |batch| {
                expected.extend(batch.lines);
                expected_positions.extend(batch.module_positions);
                Ok(())
            },
        )
        .unwrap();
        let mut actual = Vec::new();
        let mut actual_positions = Vec::new();
        let request = Turtle2dStreamRequest {
            generation: &generation,
            config,
            batch_size: 31,
            is_cancelled: &|| false,
        };
        let mut collect = |batch: Turtle2dBatch| {
            actual.extend(batch.lines);
            actual_positions.extend(batch.module_positions);
            Ok(())
        };
        let summary = stream_cuda(&mut runtime, &request, &mut collect).unwrap();

        assert_eq!(summary.backend_used, VisualizerBackend::Cuda);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            for (actual, expected) in [
                (actual.line.0.0, expected.line.0.0),
                (actual.line.0.1, expected.line.0.1),
                (actual.line.1.0, expected.line.1.0),
                (actual.line.1.1, expected.line.1.1),
                (actual.width, expected.width),
            ] {
                let tolerance = 1.0e-9 * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance);
            }
        }
        assert_eq!(actual_positions.len(), expected_positions.len());
        for (actual, expected) in actual_positions.iter().zip(&expected_positions) {
            assert_eq!(actual.module_index, expected.module_index);
            for (actual, expected) in [
                (actual.position.0, expected.position.0),
                (actual.position.1, expected.position.1),
            ] {
                let tolerance = 1.0e-9 * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn empty_cuda_stream_completes_without_allocating_a_group_when_available() {
        let mut runtime = match CudaTurtleRuntime::new() {
            Ok(runtime) => runtime,
            Err(VisualizeError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA initialization failure: {error}"),
        };
        let generation = Generation::default();
        let request = Turtle2dStreamRequest {
            generation: &generation,
            config: crate::Turtle2dConfig::default(),
            batch_size: 8,
            is_cancelled: &|| false,
        };
        let mut batches = 0;
        let summary = stream_cuda(&mut runtime, &request, &mut |_| {
            batches += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(summary.backend_used, VisualizerBackend::Cuda);
        assert_eq!(summary.progress, Turtle2dProgress::default());
        assert_eq!(batches, 0);
        assert!(runtime.scratch.is_none());
    }

    #[test]
    fn compatible_cuda_streams_reuse_scratch_when_a_device_is_available() {
        let mut runtime = match CudaTurtleRuntime::new() {
            Ok(runtime) => runtime,
            Err(VisualizeError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA initialization failure: {error}"),
        };
        let generation = Generation(
            (0..64)
                .map(|_| GenerationItem::Module(Module::new("F", Vec::new())))
                .collect(),
        );
        let mut first_pointer = None;
        for _ in 0..2 {
            let request = Turtle2dStreamRequest {
                generation: &generation,
                config: crate::Turtle2dConfig::default(),
                batch_size: 16,
                is_cancelled: &|| false,
            };
            stream_cuda(&mut runtime, &request, &mut |_| Ok(())).unwrap();
            let pointer = runtime
                .scratch
                .as_ref()
                .expect("non-empty stream allocated scratch")
                .input
                .cu_deviceptr();
            if let Some(first_pointer) = first_pointer {
                assert_eq!(pointer, first_pointer);
            } else {
                assert_ne!(pointer, 0);
                first_pointer = Some(pointer);
            }
        }
    }

    #[test]
    fn cancellation_after_first_launch_is_not_a_fallback_safe_failure_when_available() {
        use std::cell::Cell;

        let mut runtime = match CudaTurtleRuntime::new() {
            Ok(runtime) => runtime,
            Err(VisualizeError::BackendUnavailable { .. }) => return,
            Err(error) => panic!("unexpected CUDA initialization failure: {error}"),
        };
        let generation = Generation(
            (0..4)
                .map(|_| GenerationItem::Module(Module::new("F", Vec::new())))
                .collect(),
        );
        let checks = Cell::new(0usize);
        let is_cancelled = || {
            let check = checks.get();
            checks.set(check + 1);
            check >= 8
        };
        let request = Turtle2dStreamRequest {
            generation: &generation,
            config: crate::Turtle2dConfig::default(),
            batch_size: 16,
            is_cancelled: &is_cancelled,
        };
        let error = stream_cuda(&mut runtime, &request, &mut |_| Ok(())).unwrap_err();
        assert!(error.started);
        assert_eq!(error.error, VisualizeError::Cancelled);
    }
}
