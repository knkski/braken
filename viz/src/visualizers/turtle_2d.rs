#[cfg(test)]
use braken::Generation;
use braken::{GenerationItem, Module, Value};

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
use crate::VisualizerKind;
use crate::{
    Bounds2d, IndexedModulePosition2d, Line2d, Polygon2d, StrokeColor, StyledLine2d, Turtle2dBatch,
    Turtle2dConfig, Turtle2dProgress, Turtle2dStreamRequest, Turtle2dStreamSummary, VisualizeError,
    VisualizerBackend, palette_color,
};
#[cfg(test)]
use crate::{Primitive2d, Scene2d, VisualizationContext};

#[cfg(test)]
const COMPATIBILITY_BATCH_SIZE: usize = 16 * 1024;

#[cfg(test)]
pub(crate) fn visualize_cpu(
    generation: &Generation,
    config: Turtle2dConfig,
    _: VisualizationContext,
) -> Result<Scene2d, VisualizeError> {
    let background = config.background;
    let mut primitives = Vec::new();
    stream_cpu(
        Turtle2dStreamRequest {
            generation,
            config,
            batch_size: COMPATIBILITY_BATCH_SIZE,
            is_cancelled: &|| false,
        },
        |batch| {
            let additional = batch.lines.len().saturating_add(batch.polygons.len());
            primitives
                .try_reserve(additional)
                .map_err(|_| resource_error("complete turtle scene", additional))?;
            primitives.extend(batch.lines.into_iter().map(Primitive2d::Line));
            primitives.extend(batch.polygons.into_iter().map(Primitive2d::Polygon));
            Ok(())
        },
    )?;
    Ok(Scene2d {
        primitives,
        background,
    })
}

#[derive(Default)]
pub(crate) struct StreamerState {
    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    cuda: Option<super::turtle_2d_cuda::CudaTurtleRuntime>,
}

pub(crate) fn stream(
    state: &mut StreamerState,
    backend: VisualizerBackend,
    request: Turtle2dStreamRequest<'_>,
    emit: impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle2dStreamSummary, VisualizeError> {
    validate_request(&request)?;
    match backend {
        VisualizerBackend::Cpu => stream_cpu(request, emit),
        VisualizerBackend::Cuda => stream_explicit_cuda(state, request, emit),
        VisualizerBackend::Auto => {
            #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
            {
                let mut emit = emit;
                if let Err(error) = preflight_cuda_input(&request) {
                    return match error {
                        VisualizeError::UnsupportedInput { .. } => {
                            stream_cpu_validated(&request, &mut emit)
                        }
                        error => Err(error),
                    };
                }
                if state.cuda.is_none() {
                    match super::turtle_2d_cuda::CudaTurtleRuntime::new() {
                        Ok(runtime) => state.cuda = Some(runtime),
                        Err(_) => return stream_cpu_validated(&request, &mut emit),
                    }
                }
                let cuda_result = super::turtle_2d_cuda::stream_cuda(
                    state.cuda.as_mut().expect("CUDA runtime initialized"),
                    &request,
                    &mut emit,
                );
                match cuda_result {
                    Ok(summary) => Ok(summary),
                    // Scratch allocation, launch-contract validation, and the
                    // initial command upload are preflight: no kernel or sink
                    // has observed work yet, so Auto can still run CPU. Once
                    // the first kernel is submitted, every failure propagates.
                    Err(failure)
                        if !failure.started
                            && matches!(
                                &failure.error,
                                VisualizeError::BackendUnavailable { .. }
                                    | VisualizeError::BackendRuntime { .. }
                                    | VisualizeError::ResourceExhausted { .. }
                            ) =>
                    {
                        if matches!(
                            &failure.error,
                            VisualizeError::BackendUnavailable { .. }
                                | VisualizeError::BackendRuntime { .. }
                        ) {
                            state.cuda = None;
                        }
                        stream_cpu_validated(&request, &mut emit)
                    }
                    Err(failure) => {
                        if matches!(
                            &failure.error,
                            VisualizeError::BackendUnavailable { .. }
                                | VisualizeError::BackendRuntime { .. }
                        ) {
                            state.cuda = None;
                        }
                        Err(failure.error)
                    }
                }
            }
            #[cfg(not(all(feature = "cuda", not(target_arch = "wasm32"))))]
            {
                stream_cpu(request, emit)
            }
        }
    }
}

fn stream_explicit_cuda(
    state: &mut StreamerState,
    request: Turtle2dStreamRequest<'_>,
    emit: impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle2dStreamSummary, VisualizeError> {
    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    {
        let mut emit = emit;
        preflight_cuda_input(&request)?;
        if state.cuda.is_none() {
            state.cuda = Some(super::turtle_2d_cuda::CudaTurtleRuntime::new()?);
        }
        let result = super::turtle_2d_cuda::stream_cuda(
            state.cuda.as_mut().expect("CUDA runtime initialized"),
            &request,
            &mut emit,
        );
        match result {
            Ok(summary) => Ok(summary),
            Err(failure) => {
                if matches!(
                    &failure.error,
                    VisualizeError::BackendUnavailable { .. }
                        | VisualizeError::BackendRuntime { .. }
                ) {
                    state.cuda = None;
                }
                Err(failure.error)
            }
        }
    }
    #[cfg(not(all(feature = "cuda", not(target_arch = "wasm32"))))]
    {
        let _ = (state, request, emit);
        Err(VisualizeError::BackendUnavailable {
            backend: VisualizerBackend::Cuda,
        })
    }
}

#[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
fn preflight_cuda_input(request: &Turtle2dStreamRequest<'_>) -> Result<(), VisualizeError> {
    for item in request.generation.items() {
        if (request.is_cancelled)() {
            return Err(VisualizeError::Cancelled);
        }
        match item {
            GenerationItem::Branch(_) => {
                return Err(VisualizeError::UnsupportedInput {
                    visualizer: VisualizerKind::Turtle2d,
                    backend: VisualizerBackend::Cuda,
                    reason: "branch state save/restore requires the CPU turtle backend".into(),
                });
            }
            GenerationItem::Module(module) => {
                let op = classify_module(module, &request.config)?;
                if matches!(
                    op,
                    TurtleOp::TurnAngleDelta(_)
                        | TurtleOp::ColorSet(_)
                        | TurtleOp::ColorDelta(_)
                        | TurtleOp::Dot(_)
                        | TurtleOp::PolygonBegin
                        | TurtleOp::PolygonEnd
                        | TurtleOp::Vertex
                        | TurtleOp::Cut
                ) {
                    return Err(VisualizeError::UnsupportedInput {
                        visualizer: VisualizerKind::Turtle2d,
                        backend: VisualizerBackend::Cuda,
                        reason: format!(
                            "command {:?} requires the complete CPU turtle backend",
                            module.name
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TurtleState {
    position: (f64, f64),
    angle: f64,
    step_scale: f64,
    width: f64,
    invert_turns: bool,
    turn_angle: f64,
    color: StrokeColor,
    color_index: Option<i32>,
}

#[cfg(any(test, all(feature = "cuda", not(target_arch = "wasm32"))))]
mod cuda_math {
    use super::TurtleOp;

    pub(crate) const OP_DRAW: u32 = 1;
    pub(crate) const OP_MOVE: u32 = 2;
    pub(crate) const OP_TURN: u32 = 3;
    pub(crate) const OP_TURN_AROUND: u32 = 4;
    pub(crate) const OP_INVERT_TURNS: u32 = 5;
    pub(crate) const OP_SCALE: u32 = 6;
    pub(crate) const OP_WIDTH_SET: u32 = 7;
    pub(crate) const OP_WIDTH_DELTA: u32 = 8;
    pub(crate) const OP_HALF_MOVE: u32 = 9;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    #[cfg_attr(
        all(feature = "cuda", not(target_arch = "wasm32")),
        derive(cuda_core::DeviceCopy)
    )]
    pub(crate) struct EncodedOp {
        pub(crate) value: f64,
        pub(crate) tag: u32,
        pub(crate) draw: u32,
    }

    impl From<TurtleOp> for EncodedOp {
        fn from(op: TurtleOp) -> Self {
            match op {
                TurtleOp::Draw(value) => Self {
                    value,
                    tag: OP_DRAW,
                    draw: 1,
                },
                TurtleOp::Move(value) => Self {
                    value,
                    tag: OP_MOVE,
                    draw: 0,
                },
                TurtleOp::HalfMove(value) => Self {
                    value,
                    tag: OP_HALF_MOVE,
                    draw: 0,
                },
                TurtleOp::Turn(value) => Self {
                    value,
                    tag: OP_TURN,
                    draw: 0,
                },
                TurtleOp::TurnDefault(_) => Self::default(),
                TurtleOp::TurnAround => Self {
                    value: 0.0,
                    tag: OP_TURN_AROUND,
                    draw: 0,
                },
                TurtleOp::InvertTurns => Self {
                    value: 0.0,
                    tag: OP_INVERT_TURNS,
                    draw: 0,
                },
                TurtleOp::Scale(value) => Self {
                    value,
                    tag: OP_SCALE,
                    draw: 0,
                },
                TurtleOp::WidthSet(value) => Self {
                    value,
                    tag: OP_WIDTH_SET,
                    draw: 0,
                },
                TurtleOp::WidthDelta(value) => Self {
                    value,
                    tag: OP_WIDTH_DELTA,
                    draw: 0,
                },
                TurtleOp::TurnAngleDelta(_)
                | TurtleOp::ColorSet(_)
                | TurtleOp::ColorDelta(_)
                | TurtleOp::Dot(_)
                | TurtleOp::PolygonBegin
                | TurtleOp::PolygonEnd
                | TurtleOp::Vertex
                | TurtleOp::Cut => Self::default(),
                TurtleOp::Noop => Self::default(),
            }
        }
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(
        all(feature = "cuda", not(target_arch = "wasm32")),
        derive(cuda_core::DeviceCopy)
    )]
    pub(crate) struct HeadingTransform {
        pub(crate) normal_delta: f64,
        pub(crate) inverted_delta: f64,
        pub(crate) scale: f64,
        pub(crate) invert_xor: u32,
        pub(crate) _padding: u32,
    }

    impl HeadingTransform {
        pub(crate) const IDENTITY: Self = Self {
            normal_delta: 0.0,
            inverted_delta: 0.0,
            scale: 1.0,
            invert_xor: 0,
            _padding: 0,
        };
    }

    impl Default for HeadingTransform {
        fn default() -> Self {
            Self::IDENTITY
        }
    }

    pub(crate) fn heading_transform(op: EncodedOp) -> HeadingTransform {
        match op.tag {
            OP_TURN => HeadingTransform {
                normal_delta: op.value,
                inverted_delta: -op.value,
                ..HeadingTransform::IDENTITY
            },
            OP_TURN_AROUND => HeadingTransform {
                normal_delta: core::f64::consts::PI,
                inverted_delta: core::f64::consts::PI,
                ..HeadingTransform::IDENTITY
            },
            OP_INVERT_TURNS => HeadingTransform {
                invert_xor: 1,
                ..HeadingTransform::IDENTITY
            },
            OP_SCALE => HeadingTransform {
                scale: op.value,
                ..HeadingTransform::IDENTITY
            },
            _ => HeadingTransform::IDENTITY,
        }
    }

    /// Compose two source-ordered transforms (`first`, then `second`).
    pub(crate) fn compose_heading(
        first: HeadingTransform,
        second: HeadingTransform,
    ) -> HeadingTransform {
        let (second_normal, second_inverted) = if first.invert_xor == 0 {
            (second.normal_delta, second.inverted_delta)
        } else {
            (second.inverted_delta, second.normal_delta)
        };
        HeadingTransform {
            normal_delta: first.normal_delta + second_normal,
            inverted_delta: first.inverted_delta + second_inverted,
            scale: first.scale * second.scale,
            invert_xor: first.invert_xor ^ second.invert_xor,
            _padding: 0,
        }
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(
        all(feature = "cuda", not(target_arch = "wasm32")),
        derive(cuda_core::DeviceCopy)
    )]
    pub(crate) struct WidthTransform {
        pub(crate) add: f64,
        pub(crate) minimum: f64,
        pub(crate) takes_input: u32,
        pub(crate) has_minimum: u32,
    }

    impl WidthTransform {
        pub(crate) const IDENTITY: Self = Self {
            add: 0.0,
            minimum: 0.0,
            takes_input: 1,
            has_minimum: 0,
        };

        pub(crate) fn set(value: f64) -> Self {
            Self {
                add: value,
                minimum: 0.0,
                takes_input: 0,
                has_minimum: 0,
            }
        }

        pub(crate) fn adjust(delta: f64) -> Self {
            Self {
                add: delta,
                minimum: 0.0,
                takes_input: 1,
                has_minimum: 1,
            }
        }
    }

    impl Default for WidthTransform {
        fn default() -> Self {
            Self::IDENTITY
        }
    }

    pub(crate) fn apply_width(transform: WidthTransform, input: f64) -> f64 {
        let mut output = if transform.takes_input == 0 {
            transform.add
        } else {
            input + transform.add
        };
        if transform.has_minimum != 0 && output < transform.minimum {
            output = transform.minimum;
        }
        output
    }

    pub(crate) fn compose_width(first: WidthTransform, second: WidthTransform) -> WidthTransform {
        if second.takes_input == 0 {
            return WidthTransform::set(apply_width(second, 0.0));
        }
        if first.takes_input == 0 {
            return WidthTransform::set(apply_width(second, apply_width(first, 0.0)));
        }

        let add = first.add + second.add;
        let first_minimum = first.minimum + second.add;
        let (minimum, has_minimum) = match (first.has_minimum != 0, second.has_minimum != 0) {
            (true, true) => (
                if first_minimum > second.minimum {
                    first_minimum
                } else {
                    second.minimum
                },
                1,
            ),
            (true, false) => (first_minimum, 1),
            (false, true) => (second.minimum, 1),
            (false, false) => (0.0, 0),
        };
        WidthTransform {
            add,
            minimum,
            takes_input: 1,
            has_minimum,
        }
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(
        all(feature = "cuda", not(target_arch = "wasm32")),
        derive(cuda_core::DeviceCopy)
    )]
    pub(crate) struct MotionTransform {
        pub(crate) dx: f64,
        pub(crate) dy: f64,
        pub(crate) width: WidthTransform,
    }

    impl MotionTransform {
        pub(crate) const IDENTITY: Self = Self {
            dx: 0.0,
            dy: 0.0,
            width: WidthTransform::IDENTITY,
        };
    }

    impl Default for MotionTransform {
        fn default() -> Self {
            Self::IDENTITY
        }
    }

    pub(crate) fn compose_motion(
        first: MotionTransform,
        second: MotionTransform,
    ) -> MotionTransform {
        MotionTransform {
            dx: first.dx + second.dx,
            dy: first.dy + second.dy,
            width: compose_width(first.width, second.width),
        }
    }
}

#[cfg(any(test, all(feature = "cuda", not(target_arch = "wasm32"))))]
pub(crate) use cuda_math::*;

struct TraversalFrame<'a> {
    items: &'a [GenerationItem],
    next: usize,
    restore: Option<TurtleState>,
}

pub(crate) fn stream_cpu(
    request: Turtle2dStreamRequest<'_>,
    mut emit: impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle2dStreamSummary, VisualizeError> {
    validate_request(&request)?;
    let result = stream_cpu_validated(&request, &mut emit)?;
    Ok(result)
}

fn stream_cpu_validated(
    request: &Turtle2dStreamRequest<'_>,
    emit: &mut impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle2dStreamSummary, VisualizeError> {
    let config = &request.config;
    let generation = request.generation;
    let batch_size = request.batch_size;
    if request.batch_size == 0 {
        unreachable!("validated above");
    }
    if (request.is_cancelled)() {
        return Err(VisualizeError::Cancelled);
    }

    let mut state = TurtleState {
        position: (0.0, 0.0),
        angle: config.initial_angle,
        step_scale: 1.0,
        width: config.initial_width,
        invert_turns: false,
        turn_angle: config.turn_angle,
        color: config.initial_color,
        color_index: match config.initial_color {
            StrokeColor::PaletteIndex(index) => Some(i32::from(index)),
            StrokeColor::ThemeDefault | StrokeColor::Rgb(_) => None,
        },
    };
    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|_| resource_error("turtle traversal stack", 1))?;
    stack.push(TraversalFrame {
        items: generation.items(),
        next: 0,
        restore: None,
    });

    let mut lines = new_line_batch(batch_size)?;
    let mut emitted_polygons = Vec::new();
    let mut polygon_stack = Vec::new();
    let mut module_indices = new_module_index_batch(batch_size)?;
    let mut module_positions = new_module_position_batch(batch_size)?;
    let mut batch_bounds = None;
    let mut overall_bounds = None;
    let mut progress = Turtle2dProgress::default();
    let mut batch_items = 0usize;

    while !stack.is_empty() {
        if (request.is_cancelled)() {
            return Err(VisualizeError::Cancelled);
        }

        let item = {
            let Some(frame) = stack.last_mut() else {
                break;
            };
            if frame.next == frame.items.len() {
                let restore = frame.restore;
                stack.pop();
                if let Some(saved) = restore {
                    state = saved;
                }
                progress.active_branch_depth = stack.len().saturating_sub(1);
                continue;
            }
            let item = &frame.items[frame.next];
            frame.next += 1;
            item
        };

        progress.items_processed =
            checked_increment(progress.items_processed, "turtle processed-item counter")?;
        match item {
            GenerationItem::Module(module) => {
                let module_index = progress.modules_processed;
                progress.modules_processed = checked_increment(
                    progress.modules_processed,
                    "turtle processed-module counter",
                )?;
                module_positions.push(IndexedModulePosition2d {
                    module_index,
                    position: state.position,
                });
                match visit_module(module, config, &mut state, &mut polygon_stack)? {
                    VisitOutcome::Line(line) => {
                        include_bounds(&mut batch_bounds, line.line);
                        include_bounds(&mut overall_bounds, line.line);
                        lines.push(line);
                        module_indices.push(module_index);
                        progress.lines_emitted = checked_increment(
                            progress.lines_emitted,
                            "turtle emitted-line counter",
                        )?;
                    }
                    VisitOutcome::Cut => {
                        if let Some(frame) = stack.last_mut() {
                            frame.next = frame.items.len();
                        }
                    }
                    VisitOutcome::Polygon(polygon) => {
                        for &point in &polygon.vertices {
                            include_point(&mut batch_bounds, point);
                            include_point(&mut overall_bounds, point);
                        }
                        emitted_polygons.push(polygon);
                        progress.polygons_emitted = checked_increment(
                            progress.polygons_emitted,
                            "turtle emitted-polygon counter",
                        )?;
                    }
                    VisitOutcome::None => {}
                }
            }
            GenerationItem::Branch(branch) => {
                progress.branches_entered =
                    checked_increment(progress.branches_entered, "turtle entered-branch counter")?;
                let requested = stack
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| resource_error("turtle traversal stack", usize::MAX))?;
                stack
                    .try_reserve(1)
                    .map_err(|_| resource_error("turtle traversal stack", requested))?;
                stack.push(TraversalFrame {
                    items: branch.items(),
                    next: 0,
                    restore: Some(state),
                });
                progress.active_branch_depth = stack.len() - 1;
                progress.max_branch_depth =
                    progress.max_branch_depth.max(progress.active_branch_depth);
            }
        }

        batch_items = checked_increment(batch_items, "turtle batch work counter")?;
        if batch_items == batch_size {
            deliver_batch(
                request,
                emit,
                &mut lines,
                &mut emitted_polygons,
                &mut module_indices,
                &mut module_positions,
                &mut batch_bounds,
                overall_bounds,
                progress,
                true,
            )?;
            batch_items = 0;
        }
    }

    if batch_items != 0 {
        deliver_batch(
            request,
            emit,
            &mut lines,
            &mut emitted_polygons,
            &mut module_indices,
            &mut module_positions,
            &mut batch_bounds,
            overall_bounds,
            progress,
            false,
        )?;
    }
    if !polygon_stack.is_empty() {
        return Err(VisualizeError::InvalidConfiguration(
            "turtle_2d ended with an unterminated polygon".into(),
        ));
    }
    Ok(Turtle2dStreamSummary {
        bounds: overall_bounds,
        progress,
        backend_used: VisualizerBackend::Cpu,
    })
}

pub(crate) fn validate_request(request: &Turtle2dStreamRequest<'_>) -> Result<(), VisualizeError> {
    validate_config(&request.config)?;
    if request.batch_size == 0 {
        return Err(VisualizeError::InvalidConfiguration(
            "turtle_2d batch size must be greater than zero".into(),
        ));
    }
    if (request.is_cancelled)() {
        return Err(VisualizeError::Cancelled);
    }
    Ok(())
}

fn validate_config(config: &Turtle2dConfig) -> Result<(), VisualizeError> {
    if !config.turn_angle.is_finite()
        || !config.initial_angle.is_finite()
        || !config.default_step.is_finite()
        || !config.scale_multiplier.is_finite()
        || !config.initial_width.is_finite()
        || config.initial_width < 0.0
        || !config.width_increment.is_finite()
        || config.width_increment < 0.0
        || !config.turn_angle_increment.is_finite()
    {
        return Err(VisualizeError::InvalidConfiguration(
            "turtle_2d distances and angles must be finite, and widths must be finite and non-negative"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn new_line_batch(size: usize) -> Result<Vec<StyledLine2d>, VisualizeError> {
    let mut lines = Vec::new();
    lines
        .try_reserve_exact(size)
        .map_err(|_| resource_error("turtle line batch", size))?;
    Ok(lines)
}

pub(crate) fn new_module_index_batch(size: usize) -> Result<Vec<usize>, VisualizeError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(size)
        .map_err(|_| resource_error("turtle module-index batch", size))?;
    Ok(indices)
}

pub(crate) fn new_module_position_batch(
    size: usize,
) -> Result<Vec<IndexedModulePosition2d>, VisualizeError> {
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(size)
        .map_err(|_| resource_error("turtle module-position batch", size))?;
    Ok(positions)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn deliver_batch(
    request: &Turtle2dStreamRequest<'_>,
    emit: &mut impl FnMut(Turtle2dBatch) -> Result<(), VisualizeError>,
    lines: &mut Vec<StyledLine2d>,
    polygons: &mut Vec<Polygon2d>,
    module_indices: &mut Vec<usize>,
    module_positions: &mut Vec<IndexedModulePosition2d>,
    batch_bounds: &mut Option<Bounds2d>,
    overall_bounds: Option<Bounds2d>,
    progress: Turtle2dProgress,
    allocate_next: bool,
) -> Result<(), VisualizeError> {
    if (request.is_cancelled)() {
        return Err(VisualizeError::Cancelled);
    }
    let emitted = std::mem::take(lines);
    let emitted_polygons = std::mem::take(polygons);
    let emitted_indices = std::mem::take(module_indices);
    let emitted_positions = std::mem::take(module_positions);
    debug_assert_eq!(emitted.len(), emitted_indices.len());
    emit(Turtle2dBatch {
        lines: emitted,
        polygons: emitted_polygons,
        module_indices: emitted_indices,
        module_positions: emitted_positions,
        batch_bounds: batch_bounds.take(),
        overall_bounds,
        progress,
    })?;
    if allocate_next {
        *lines = new_line_batch(request.batch_size)?;
        *module_indices = new_module_index_batch(request.batch_size)?;
        *module_positions = new_module_position_batch(request.batch_size)?;
    }
    Ok(())
}

pub(crate) fn include_bounds(bounds: &mut Option<Bounds2d>, line: Line2d) {
    if let Some(bounds) = bounds {
        bounds.include_line(line);
    } else {
        *bounds = Some(Bounds2d::from_line(line));
    }
}

fn include_point(bounds: &mut Option<Bounds2d>, point: (f64, f64)) {
    include_bounds(bounds, Line2d(point, point));
}

pub(crate) fn checked_increment(
    value: usize,
    resource: &'static str,
) -> Result<usize, VisualizeError> {
    value
        .checked_add(1)
        .ok_or_else(|| resource_error(resource, usize::MAX))
}

pub(crate) fn resource_error(resource: &'static str, requested: usize) -> VisualizeError {
    VisualizeError::ResourceExhausted {
        resource,
        requested: Some(requested),
    }
}

fn resolve_action<'a>(
    module_name: &'a str,
    config: &'a Turtle2dConfig,
) -> Result<&'a str, VisualizeError> {
    let mut name = module_name;
    let mut followed = 0usize;
    loop {
        let Some((_, action)) = config
            .module_aliases
            .iter()
            .find(|(candidate, _)| candidate == name)
        else {
            return Ok(name);
        };
        if action == name {
            return Ok(name);
        }
        followed = followed
            .checked_add(1)
            .ok_or_else(|| resource_error("turtle alias traversal", usize::MAX))?;
        if followed > config.module_aliases.len() {
            return Err(VisualizeError::InvalidConfiguration(format!(
                "turtle_2d render-map alias cycle involving {module_name:?}"
            )));
        }
        name = action;
    }
}

fn ensure_state_is_finite(state: &TurtleState) -> Result<(), VisualizeError> {
    if state.position.0.is_finite()
        && state.position.1.is_finite()
        && state.angle.is_finite()
        && state.step_scale.is_finite()
        && state.width.is_finite()
    {
        Ok(())
    } else {
        Err(VisualizeError::InvalidConfiguration(
            "turtle_2d produced non-finite state".into(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TurtleOp {
    Draw(f64),
    Move(f64),
    HalfMove(f64),
    Turn(f64),
    TurnDefault(f64),
    TurnAround,
    InvertTurns,
    Scale(f64),
    WidthSet(f64),
    WidthDelta(f64),
    TurnAngleDelta(f64),
    ColorSet(i32),
    ColorDelta(i32),
    Dot(f64),
    PolygonBegin,
    PolygonEnd,
    Vertex,
    Cut,
    Noop,
}

pub(crate) fn classify_module(
    module: &Module,
    config: &Turtle2dConfig,
) -> Result<TurtleOp, VisualizeError> {
    let name = resolve_action(module.name.as_str(), config)?;
    let special = match name {
        "HalfForward" => Some(TurtleOp::Draw(module_distance(module, config) * 0.5)),
        "HalfMove" => Some(TurtleOp::HalfMove(module_distance(module, config))),
        _ => None,
    };
    if let Some(op) = special {
        return Ok(op);
    }
    if config
        .draw_modules
        .iter()
        .any(|candidate| candidate == name)
    {
        return Ok(TurtleOp::Draw(module_distance(module, config)));
    }
    if config
        .move_modules
        .iter()
        .any(|candidate| candidate == name)
    {
        return Ok(TurtleOp::Move(module_distance(module, config)));
    }

    let op = match name {
        "+" | "Plus" | "ExplicitPlus" | "Left" | "Turn" => {
            module_degrees(module).map_or(TurtleOp::TurnDefault(1.0), TurtleOp::Turn)
        }
        "-" | "Minus" | "ExplicitMinus" | "Right" => module_degrees(module)
            .map(|angle| TurtleOp::Turn(-angle))
            .unwrap_or(TurtleOp::TurnDefault(-1.0)),
        "TurnAround" | "Around" => TurtleOp::TurnAround,
        "InvertTurns" => TurtleOp::InvertTurns,
        "Scale" | "ScaleLength" => {
            TurtleOp::Scale(module_angle(module).unwrap_or(config.scale_multiplier))
        }
        "ScaleLengthInverse" => TurtleOp::Scale(
            module_angle(module)
                .unwrap_or(config.scale_multiplier)
                .recip(),
        ),
        "Width" | "SetWidth" => TurtleOp::WidthSet(width_argument(module)?),
        "WidthIncrease" if module.arguments.is_empty() => {
            TurtleOp::WidthDelta(config.width_increment)
        }
        "WidthDecrease" if module.arguments.is_empty() => {
            TurtleOp::WidthDelta(-config.width_increment)
        }
        // The normalized corpus uses parameterized forms for the source
        // renderer's absolute-width commands; only bare forms are relative.
        "WidthIncrease" | "WidthDecrease" => TurtleOp::WidthSet(width_argument(module)?),
        "TurnAngleIncrease" => {
            TurtleOp::TurnAngleDelta(module_degrees(module).unwrap_or(config.turn_angle_increment))
        }
        "TurnAngleDecrease" => {
            TurtleOp::TurnAngleDelta(-module_degrees(module).unwrap_or(config.turn_angle_increment))
        }
        "Color" => TurtleOp::ColorSet(color_argument(module)?),
        "ColorIncrement" => {
            TurtleOp::ColorDelta(module_integer(module).unwrap_or(config.color_increment))
        }
        "ColorNext" => {
            TurtleOp::ColorDelta(module_integer(module).unwrap_or(config.color_increment))
        }
        "ColorPrevious" => {
            TurtleOp::ColorDelta(-module_integer(module).unwrap_or(config.color_increment))
        }
        "Dot" => TurtleOp::Dot(module_angle(module).unwrap_or(1.0)),
        "PolygonBegin" => TurtleOp::PolygonBegin,
        "PolygonEnd" => TurtleOp::PolygonEnd,
        "Vertex" => TurtleOp::Vertex,
        "Cut" => TurtleOp::Cut,
        _ => TurtleOp::Noop,
    };
    Ok(op)
}

enum VisitOutcome {
    Line(StyledLine2d),
    Polygon(Polygon2d),
    Cut,
    None,
}

fn visit_module(
    module: &Module,
    config: &Turtle2dConfig,
    state: &mut TurtleState,
    polygon_stack: &mut Vec<(Vec<(f64, f64)>, StrokeColor)>,
) -> Result<VisitOutcome, VisualizeError> {
    // In the ABOP polygon vocabulary, F and f trace polygon edges as they
    // advance. Other drawing modules, notably the G framework used by the
    // developmental leaf models, deliberately do not contribute vertices.
    let traces_polygon_edge = matches!(resolve_action(module.name.as_str(), config)?, "F" | "f");
    let line: Option<()> = match classify_module(module, config)? {
        TurtleOp::Draw(distance) => {
            return advance_with_polygon_capture(
                distance * state.step_scale,
                state,
                true,
                traces_polygon_edge,
                polygon_stack,
            );
        }
        TurtleOp::Move(distance) => {
            return advance_with_polygon_capture(
                distance * state.step_scale,
                state,
                false,
                traces_polygon_edge,
                polygon_stack,
            );
        }
        TurtleOp::HalfMove(distance) => {
            return advance_by(distance * state.step_scale * 0.5, state, false);
        }
        TurtleOp::Turn(delta) => {
            state.angle += if state.invert_turns { -delta } else { delta };
            None
        }
        TurtleOp::TurnDefault(direction) => {
            let delta = state.turn_angle * direction;
            state.angle += if state.invert_turns { -delta } else { delta };
            None
        }
        TurtleOp::TurnAround => {
            state.angle += std::f64::consts::PI;
            None
        }
        TurtleOp::InvertTurns => {
            state.invert_turns = !state.invert_turns;
            None
        }
        TurtleOp::Scale(scale) => {
            state.step_scale *= scale;
            None
        }
        TurtleOp::WidthSet(width) => {
            state.width = width;
            None
        }
        TurtleOp::WidthDelta(delta) => {
            state.width = (state.width + delta).max(0.0);
            None
        }
        TurtleOp::TurnAngleDelta(delta) => {
            state.turn_angle += delta;
            None
        }
        TurtleOp::ColorSet(index) => {
            state.color_index = Some(index);
            state.color = palette_color(index, &config.palette);
            None
        }
        TurtleOp::ColorDelta(delta) => {
            let index = state
                .color_index
                .unwrap_or(0)
                .checked_add(delta)
                .ok_or_else(|| {
                    VisualizeError::InvalidConfiguration("turtle color index overflow".into())
                })?;
            state.color_index = Some(index);
            state.color = palette_color(index, &config.palette);
            None
        }
        TurtleOp::Dot(diameter) => {
            if !diameter.is_finite() || diameter < 0.0 {
                return Err(VisualizeError::InvalidConfiguration(
                    "turtle dot diameter must be finite and non-negative".into(),
                ));
            }
            return Ok(VisitOutcome::Line(StyledLine2d {
                line: Line2d(state.position, state.position),
                width: state.width * diameter,
                color: state.color,
            }));
        }
        TurtleOp::PolygonBegin => {
            polygon_stack.try_reserve(1).map_err(|_| {
                resource_error(
                    "turtle polygon stack",
                    polygon_stack.len().saturating_add(1),
                )
            })?;
            polygon_stack.push((Vec::new(), state.color));
            None
        }
        TurtleOp::Vertex => {
            let Some((vertices, _)) = polygon_stack.last_mut() else {
                return Err(VisualizeError::InvalidConfiguration(
                    "Vertex requires an active PolygonBegin".into(),
                ));
            };
            vertices.try_reserve(1).map_err(|_| {
                resource_error("turtle polygon vertices", vertices.len().saturating_add(1))
            })?;
            vertices.push(state.position);
            None
        }
        TurtleOp::PolygonEnd => {
            let Some((vertices, color)) = polygon_stack.pop() else {
                return Err(VisualizeError::InvalidConfiguration(
                    "PolygonEnd requires an active PolygonBegin".into(),
                ));
            };
            if vertices.len() >= 3 {
                return Ok(VisitOutcome::Polygon(Polygon2d { vertices, color }));
            }
            None
        }
        TurtleOp::Cut => return Ok(VisitOutcome::Cut),
        TurtleOp::Noop => None,
    };
    ensure_state_is_finite(state)?;
    debug_assert!(line.is_none());
    Ok(VisitOutcome::None)
}

fn module_distance(module: &Module, config: &Turtle2dConfig) -> f64 {
    module
        .arguments
        .first()
        .and_then(value_to_number)
        .unwrap_or(config.default_step)
}

fn advance_by(
    distance: f64,
    state: &mut TurtleState,
    draw: bool,
) -> Result<VisitOutcome, VisualizeError> {
    let next = (
        state.position.0 + distance * state.angle.cos(),
        state.position.1 + distance * state.angle.sin(),
    );
    if !next.0.is_finite() || !next.1.is_finite() {
        return Err(VisualizeError::InvalidConfiguration(
            "turtle_2d produced a non-finite coordinate".into(),
        ));
    }
    let line = draw.then_some(StyledLine2d {
        line: Line2d(state.position, next),
        width: state.width,
        color: state.color,
    });
    state.position = next;
    Ok(line.map_or(VisitOutcome::None, VisitOutcome::Line))
}

fn advance_with_polygon_capture(
    distance: f64,
    state: &mut TurtleState,
    draw: bool,
    traces_polygon_edge: bool,
    polygon_stack: &mut [(Vec<(f64, f64)>, StrokeColor)],
) -> Result<VisitOutcome, VisualizeError> {
    let start = state.position;
    let outcome = advance_by(distance, state, draw)?;
    if traces_polygon_edge {
        capture_polygon_edge(polygon_stack, start, state.position)?;
    }
    Ok(outcome)
}

fn capture_polygon_edge(
    polygon_stack: &mut [(Vec<(f64, f64)>, StrokeColor)],
    start: (f64, f64),
    end: (f64, f64),
) -> Result<(), VisualizeError> {
    let Some((vertices, _)) = polygon_stack.last_mut() else {
        return Ok(());
    };
    let include_start = vertices.last().copied() != Some(start);
    let additional = 1usize + usize::from(include_start);
    let requested = vertices
        .len()
        .checked_add(additional)
        .ok_or_else(|| resource_error("turtle polygon vertices", usize::MAX))?;
    vertices
        .try_reserve(additional)
        .map_err(|_| resource_error("turtle polygon vertices", requested))?;
    if include_start {
        vertices.push(start);
    }
    vertices.push(end);
    Ok(())
}

fn width_argument(module: &Module) -> Result<f64, VisualizeError> {
    let Some(width) = module.arguments.first().and_then(value_to_number) else {
        return Err(VisualizeError::InvalidConfiguration(
            "Width requires a numeric argument".into(),
        ));
    };
    if !width.is_finite() || width < 0.0 {
        return Err(VisualizeError::InvalidConfiguration(
            "turtle width must be finite and non-negative".into(),
        ));
    }
    Ok(width)
}

fn module_angle(module: &Module) -> Option<f64> {
    module.arguments.first().and_then(value_to_number)
}
fn module_integer(module: &Module) -> Option<i32> {
    let value = module_angle(module)?;
    if value.is_finite()
        && value.fract() == 0.0
        && value >= f64::from(i32::MIN)
        && value <= f64::from(i32::MAX)
    {
        Some(value as i32)
    } else {
        None
    }
}

fn color_argument(module: &Module) -> Result<i32, VisualizeError> {
    module_integer(module).ok_or_else(|| {
        VisualizeError::InvalidConfiguration("Color requires an integer palette index".into())
    })
}
fn module_degrees(module: &Module) -> Option<f64> {
    module_angle(module).map(f64::to_radians)
}
fn value_to_number(value: &Value) -> Option<f64> {
    if let Value::Number(number) = value {
        Some(*number)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use crate::{
        Primitive2d, Turtle2dStreamRequest, stream_turtle_2d,
        targets::{Palette, svg},
    };
    use braken::{GenerationItem, Module};

    fn collect_stream(
        generation: &Generation,
        config: Turtle2dConfig,
        batch_size: usize,
    ) -> (Vec<StyledLine2d>, Turtle2dStreamSummary, Vec<usize>) {
        let mut lines = Vec::new();
        let mut batch_lengths = Vec::new();
        let summary = stream_turtle_2d(
            Turtle2dStreamRequest {
                generation,
                config,
                batch_size,
                is_cancelled: &|| false,
            },
            |batch| {
                batch_lengths.push(batch.lines.len());
                lines.extend(batch.lines);
                Ok(())
            },
        )
        .unwrap();
        (lines, summary, batch_lengths)
    }

    #[test]
    fn draws_and_restores_branches() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Branch(Generation(vec![GenerationItem::Module(Module::new(
                "F",
                Vec::new(),
            ))])),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let output = visualize_cpu(
            &generation,
            Turtle2dConfig::default(),
            VisualizationContext::default(),
        )
        .unwrap();
        let lines = output
            .primitives
            .iter()
            .filter_map(|item| match item {
                crate::Primitive2d::Line(line) => Some(line.line),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1].0, lines[2].0);
    }

    #[test]
    fn emitted_lines_retain_depth_first_module_indices_through_branches_and_aliases() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("TurnLeft", Vec::new())),
            GenerationItem::Branch(Generation(vec![
                GenerationItem::Module(Module::new("Move", Vec::new())),
                GenerationItem::Module(Module::new("F", Vec::new())),
            ])),
            GenerationItem::Module(Module::new("X", Vec::new())),
        ]);
        let mut config = Turtle2dConfig::default();
        config
            .module_aliases
            .push((String::from("X"), String::from("F")));
        let mut indices = Vec::new();
        let mut positions = Vec::new();
        stream_turtle_2d(
            Turtle2dStreamRequest {
                generation: &generation,
                config,
                batch_size: 2,
                is_cancelled: &|| false,
            },
            |batch| {
                assert_eq!(batch.lines.len(), batch.module_indices.len());
                indices.extend(batch.module_indices);
                positions.extend(batch.module_positions);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(indices, [0, 3, 4]);
        assert_eq!(
            positions
                .iter()
                .map(|position| position.module_index)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn source_metadata_configures_heading_and_module_roles() {
        let config = Turtle2dConfig::default()
            .with_source_metadata(
                "# Heading  : 0\n# Initial Width: 2.5\n# Width Increment: 0.25\n# Draw     : X Y\n# Move     : Z",
            );
        assert_eq!(config.initial_angle, 0.0);
        assert_eq!(config.initial_width, 2.5);
        assert_eq!(config.width_increment, 0.25);
        assert_eq!(config.draw_modules, ["X", "Y"]);
        assert_eq!(config.move_modules, ["Z"]);

        let generation = Generation(vec![
            GenerationItem::Module(Module::new("X", Vec::new())),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("Z", Vec::new())),
            GenerationItem::Module(Module::new("Y", Vec::new())),
        ]);
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        let lines = scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                crate::Primitive2d::Line(line) => Some(line.line),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], Line2d((0.0, 0.0), (1.0, 0.0)));
        assert!((lines[1].0.0 - 1.0).abs() < 1.0e-10);
        assert!((lines[1].0.1 - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn normalized_scaling_and_inverted_turns_affect_geometry() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("ScaleLength", vec![Value::Number(2.0)])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("InvertTurns", Vec::new())),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            initial_angle: 0.0,
            ..Turtle2dConfig::default()
        };
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        let lines = scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                crate::Primitive2d::Line(line) => Some(line.line),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lines[0], Line2d((0.0, 0.0), (2.0, 0.0)));
        assert!((lines[1].1.0 - 2.0).abs() < 1.0e-10);
        assert!((lines[1].1.1 + 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn width_commands_style_lines_and_restore_across_branches() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("Width", vec![Value::Number(2.0)])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("WidthIncrease", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Branch(Generation(vec![
                GenerationItem::Module(Module::new("WidthDecrease", Vec::new())),
                GenerationItem::Module(Module::new("F", Vec::new())),
            ])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("WidthIncrease", vec![Value::Number(5.0)])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("WidthDecrease", vec![Value::Number(0.5)])),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let scene = visualize_cpu(
            &generation,
            Turtle2dConfig::default(),
            VisualizationContext::default(),
        )
        .unwrap();
        let widths = scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                crate::Primitive2d::Line(line) => Some(line.width),
                crate::Primitive2d::Polygon(_) | crate::Primitive2d::Text(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(widths, [2.0, 3.0, 2.0, 3.0, 5.0, 0.5]);
    }

    #[test]
    fn width_decrease_clamps_at_zero_and_invalid_widths_fail() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("WidthDecrease", Vec::new())),
            GenerationItem::Module(Module::new("WidthDecrease", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let scene = visualize_cpu(
            &generation,
            Turtle2dConfig::default(),
            VisualizationContext::default(),
        )
        .unwrap();
        assert!(matches!(
            &scene.primitives[0],
            crate::Primitive2d::Line(line) if line.width == 0.0
        ));

        let invalid = Generation(vec![GenerationItem::Module(Module::new(
            "Width",
            vec![Value::Number(-1.0)],
        ))]);
        assert!(matches!(
            visualize_cpu(
                &invalid,
                Turtle2dConfig::default(),
                VisualizationContext::default()
            ),
            Err(VisualizeError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn batches_are_exactly_equivalent_to_the_scene_for_both_palettes() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("Width", vec![Value::Number(2.5)])),
            GenerationItem::Module(Module::new("F", vec![Value::Number(2.0)])),
            GenerationItem::Module(Module::new("Plus", vec![Value::Number(45.0)])),
            GenerationItem::Branch(Generation(vec![
                GenerationItem::Module(Module::new("WidthDecrease", Vec::new())),
                GenerationItem::Module(Module::new("F", Vec::new())),
                GenerationItem::Module(Module::new("Plus", Vec::new())),
                GenerationItem::Module(Module::new("F", Vec::new())),
            ])),
            GenerationItem::Module(Module::new("WidthIncrease", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            initial_angle: 0.0,
            ..Turtle2dConfig::default()
        };
        let scene =
            visualize_cpu(&generation, config.clone(), VisualizationContext::default()).unwrap();
        let (streamed, summary, batch_lengths) = collect_stream(&generation, config, 2);
        let streamed_scene = Scene2d {
            primitives: streamed.into_iter().map(Primitive2d::Line).collect(),
            background: None,
        };

        assert_eq!(streamed_scene, scene);
        assert_eq!(batch_lengths.iter().sum::<usize>(), 4);
        assert!(batch_lengths.iter().all(|&length| length <= 2));
        assert_eq!(summary.progress.lines_emitted, 4);
        assert!(summary.bounds.is_some());
        for palette in [Palette::Light, Palette::Dark] {
            assert_eq!(
                svg::encode(&streamed_scene, palette),
                svg::encode(&scene, palette)
            );
        }
    }

    #[test]
    fn batch_size_does_not_change_geometry_style_or_progress() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("WidthIncrease", Vec::new())),
            GenerationItem::Branch(Generation(vec![
                GenerationItem::Module(Module::new("Plus", Vec::new())),
                GenerationItem::Module(Module::new("F", Vec::new())),
                GenerationItem::Branch(Generation(vec![GenerationItem::Module(Module::new(
                    "F",
                    Vec::new(),
                ))])),
            ])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let config = Turtle2dConfig::default();
        let (expected, expected_summary, _) = collect_stream(&generation, config.clone(), 64);

        for batch_size in [1, 2, 3, 4, 7] {
            let (actual, summary, lengths) =
                collect_stream(&generation, config.clone(), batch_size);
            assert_eq!(actual, expected, "batch size {batch_size}");
            assert_eq!(summary, expected_summary, "batch size {batch_size}");
            assert!(lengths.iter().all(|&length| length <= batch_size));
        }
    }

    #[test]
    fn explicit_stack_handles_more_than_4096_nested_branches() {
        const DEPTH: usize = 5_000;
        let mut generation = Generation(vec![GenerationItem::Module(Module::new("F", Vec::new()))]);
        for _ in 0..DEPTH {
            generation = Generation(vec![GenerationItem::Branch(generation)]);
        }

        let (lines, summary, _) = collect_stream(&generation, Turtle2dConfig::default(), 8);
        assert_eq!(lines.len(), 1);
        assert_eq!(summary.progress.branches_entered, DEPTH);
        assert_eq!(summary.progress.max_branch_depth, DEPTH);

        // The data model itself has recursive drop semantics; leaking this small
        // synthetic fixture keeps that unrelated recursion out of this test.
        std::mem::forget(generation);
    }

    #[test]
    fn cancellation_is_observed_inside_a_large_unflushed_batch() {
        let generation = Generation(
            (0..50_000)
                .map(|_| GenerationItem::Module(Module::new("F", Vec::new())))
                .collect(),
        );
        let checks = Cell::new(0usize);
        let is_cancelled = || {
            let current = checks.get();
            checks.set(current + 1);
            current >= 512
        };
        let mut emitted_batches = 0usize;
        let result = stream_turtle_2d(
            Turtle2dStreamRequest {
                generation: &generation,
                config: Turtle2dConfig::default(),
                batch_size: 10_000,
                is_cancelled: &is_cancelled,
            },
            |_| {
                emitted_batches += 1;
                Ok(())
            },
        );
        assert_eq!(result, Err(VisualizeError::Cancelled));
        assert_eq!(emitted_batches, 0);
        assert!(checks.get() < generation.items().len());
    }

    #[test]
    fn movement_only_work_still_emits_progress_batches() {
        let generation = Generation(
            (0..5)
                .map(|_| GenerationItem::Module(Module::new("Move", Vec::new())))
                .collect(),
        );
        let mut updates = Vec::new();
        let summary = stream_turtle_2d(
            Turtle2dStreamRequest {
                generation: &generation,
                config: Turtle2dConfig::default(),
                batch_size: 2,
                is_cancelled: &|| false,
            },
            |batch| {
                assert!(batch.lines.is_empty());
                updates.push(batch.progress.items_processed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(updates, [2, 4, 5]);
        assert_eq!(summary.progress.items_processed, 5);
        assert_eq!(summary.progress.lines_emitted, 0);
    }

    #[test]
    fn impossible_batch_allocation_is_a_typed_resource_error() {
        let generation = Generation::default();
        let error = stream_turtle_2d(
            Turtle2dStreamRequest {
                generation: &generation,
                config: Turtle2dConfig::default(),
                batch_size: usize::MAX,
                is_cancelled: &|| false,
            },
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            VisualizeError::ResourceExhausted {
                resource: "turtle line batch",
                requested: Some(usize::MAX),
            }
        ));
    }

    #[test]
    fn command_classifier_is_shared_across_aliases_and_parameter_forms() {
        let config = Turtle2dConfig {
            default_step: 2.5,
            turn_angle: 0.75,
            module_aliases: vec![("X".into(), "Plus".into())],
            ..Default::default()
        };
        assert_eq!(
            classify_module(&Module::new("F", Vec::new()), &config),
            Ok(TurtleOp::Draw(2.5))
        );
        assert_eq!(
            classify_module(&Module::new("f", vec![Value::Number(4.0)]), &config),
            Ok(TurtleOp::Move(4.0))
        );
        assert_eq!(
            classify_module(&Module::new("HalfMove", Vec::new()), &config),
            Ok(TurtleOp::HalfMove(2.5))
        );
        assert_eq!(
            classify_module(&Module::new("X", vec![Value::Number(30.0)]), &config),
            Ok(TurtleOp::Turn(30_f64.to_radians()))
        );
        assert_eq!(
            classify_module(&Module::new("WidthDecrease", Vec::new()), &config),
            Ok(TurtleOp::WidthDelta(-1.0))
        );
        assert_eq!(
            classify_module(
                &Module::new("WidthDecrease", vec![Value::Number(0.25)]),
                &config
            ),
            Ok(TurtleOp::WidthSet(0.25))
        );
    }

    #[test]
    fn ordered_heading_scan_composition_matches_sequential_state() {
        let ops = [
            TurtleOp::Turn(0.3),
            TurtleOp::InvertTurns,
            TurtleOp::Turn(0.2),
            TurtleOp::Scale(2.0),
            TurtleOp::InvertTurns,
            TurtleOp::Turn(-0.1),
            TurtleOp::Scale(0.25),
        ];
        let transforms = ops
            .into_iter()
            .map(|op| heading_transform(op.into()))
            .collect::<Vec<_>>();
        let mut scanned = transforms.clone();
        let mut scratch = scanned.clone();
        let mut stride = 1;
        while stride < scanned.len() {
            for index in 0..scanned.len() {
                scratch[index] = if index < stride {
                    scanned[index]
                } else {
                    compose_heading(scanned[index - stride], scanned[index])
                };
            }
            std::mem::swap(&mut scanned, &mut scratch);
            stride *= 2;
        }

        let mut angle = 0.4;
        let mut scale = 1.5;
        let mut inverted = false;
        for (index, op) in ops.into_iter().enumerate() {
            match op {
                TurtleOp::Turn(delta) => angle += if inverted { -delta } else { delta },
                TurtleOp::TurnAround => angle += std::f64::consts::PI,
                TurtleOp::InvertTurns => inverted = !inverted,
                TurtleOp::Scale(value) => scale *= value,
                _ => {}
            }
            let prefix = scanned[index];
            let actual_angle = 0.4 + prefix.normal_delta;
            assert!((actual_angle - angle).abs() < 1.0e-12);
            assert!((1.5 * prefix.scale - scale).abs() < 1.0e-12);
            assert_eq!(prefix.invert_xor != 0, inverted);
        }
    }

    #[test]
    fn composed_width_commands_preserve_set_and_clamp_order() {
        let commands = [
            WidthTransform::adjust(-2.0),
            WidthTransform::adjust(0.5),
            WidthTransform::set(3.0),
            WidthTransform::adjust(-5.0),
            WidthTransform::adjust(1.25),
        ];
        let combined = commands
            .into_iter()
            .fold(WidthTransform::IDENTITY, compose_width);
        for initial in [0.0, 1.0, 10.0] {
            let expected = commands
                .into_iter()
                .fold(initial, |width, command| apply_width(command, width));
            assert_eq!(apply_width(combined, initial), expected);
        }
    }

    #[test]
    fn ordered_motion_composition_matches_sequential_displacement_and_width() {
        let commands = [
            MotionTransform {
                dx: 1.0,
                dy: -2.0,
                width: WidthTransform::IDENTITY,
            },
            MotionTransform {
                dx: 0.0,
                dy: 0.0,
                width: WidthTransform::adjust(-3.0),
            },
            MotionTransform {
                dx: 4.0,
                dy: 0.5,
                width: WidthTransform::set(2.0),
            },
            MotionTransform {
                dx: -0.25,
                dy: 8.0,
                width: WidthTransform::adjust(1.5),
            },
        ];
        let combined = commands
            .into_iter()
            .fold(MotionTransform::IDENTITY, compose_motion);
        let mut x = 0.0;
        let mut y = 0.0;
        let mut width = 5.0;
        for command in commands {
            x += command.dx;
            y += command.dy;
            width = apply_width(command.width, width);
        }
        assert_eq!(combined.dx, x);
        assert_eq!(combined.dy, y);
        assert_eq!(apply_width(combined.width, 5.0), width);
    }

    #[test]
    #[cfg(all(feature = "cuda", not(target_arch = "wasm32")))]
    fn explicit_cuda_rejects_branches_before_device_initialization() {
        let generation = Generation(vec![GenerationItem::Branch(Generation::default())]);
        let error = crate::Turtle2dStreamer::new()
            .stream(
                VisualizerBackend::Cuda,
                Turtle2dStreamRequest {
                    generation: &generation,
                    config: Turtle2dConfig::default(),
                    batch_size: 16,
                    is_cancelled: &|| false,
                },
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            VisualizeError::UnsupportedInput {
                visualizer: VisualizerKind::Turtle2d,
                backend: VisualizerBackend::Cuda,
                ..
            }
        ));
    }

    #[test]
    fn auto_falls_back_to_cpu_for_branches_and_reports_it() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Branch(Generation(vec![GenerationItem::Module(Module::new(
                "F",
                Vec::new(),
            ))])),
        ]);
        let summary = crate::Turtle2dStreamer::new()
            .stream(
                VisualizerBackend::Auto,
                Turtle2dStreamRequest {
                    generation: &generation,
                    config: Turtle2dConfig::default(),
                    batch_size: 16,
                    is_cancelled: &|| false,
                },
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(summary.backend_used, VisualizerBackend::Cpu);
        assert_eq!(summary.progress.branches_entered, 1);
    }

    #[test]
    fn color_commands_use_exact_palettes_and_branch_state_is_restored() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("Color", vec![Value::Number(1.0)])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Branch(Generation(vec![
                GenerationItem::Module(Module::new("ColorNext", Vec::new())),
                GenerationItem::Module(Module::new("F", Vec::new())),
            ])),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("ColorPrevious", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            palette: vec![[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            ..Turtle2dConfig::default()
        };
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        let colors = scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive2d::Line(line) => Some(line.color),
                Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            colors,
            [
                StrokeColor::Rgb([40, 50, 60]),
                StrokeColor::Rgb([70, 80, 90]),
                StrokeColor::Rgb([40, 50, 60]),
                StrokeColor::Rgb([10, 20, 30]),
            ]
        );
    }

    #[test]
    fn half_forward_default_scaling_and_mutable_turn_angle_are_interpreted() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("HalfForward", Vec::new())),
            GenerationItem::Module(Module::new("ScaleLength", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("ScaleLengthInverse", Vec::new())),
            GenerationItem::Module(Module::new("TurnAngleIncrease", vec![Value::Number(30.0)])),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            initial_angle: 0.0,
            scale_multiplier: 2.0,
            ..Turtle2dConfig::default()
        };
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        let lines = scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive2d::Line(line) => Some(line.line),
                Primitive2d::Polygon(_) | Primitive2d::Text(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lines[0], Line2d((0.0, 0.0), (0.5, 0.0)));
        assert_eq!(lines[1], Line2d((0.5, 0.0), (2.5, 0.0)));
        assert!((lines[2].1.0 - 2.0).abs() < 1.0e-10);
        assert!((lines[2].1.1 - 3_f64.sqrt() * 0.5).abs() < 1.0e-10);
    }

    #[test]
    fn polygon_capture_and_cut_emit_only_the_intended_geometry() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("PolygonBegin", Vec::new())),
            GenerationItem::Module(Module::new("Vertex", Vec::new())),
            GenerationItem::Module(Module::new("Move", vec![Value::Number(1.0)])),
            GenerationItem::Module(Module::new("Vertex", Vec::new())),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("Move", vec![Value::Number(1.0)])),
            GenerationItem::Module(Module::new("Vertex", Vec::new())),
            GenerationItem::Module(Module::new("PolygonEnd", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("Cut", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            initial_angle: 0.0,
            ..Turtle2dConfig::default()
        };
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        assert_eq!(
            scene
                .primitives
                .iter()
                .filter(|primitive| matches!(primitive, Primitive2d::Line(_)))
                .count(),
            1
        );
        let polygon = scene
            .primitives
            .iter()
            .find_map(|primitive| match primitive {
                Primitive2d::Polygon(polygon) => Some(polygon),
                Primitive2d::Line(_) | Primitive2d::Text(_) => None,
            })
            .expect("polygon emitted");
        assert_eq!(polygon.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    fn abop_forward_moves_trace_polygon_edges_but_framework_modules_do_not() {
        let contour = Generation(vec![
            GenerationItem::Module(Module::new("PolygonBegin", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("f", Vec::new())),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("F", Vec::new())),
            GenerationItem::Module(Module::new("PolygonEnd", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            initial_angle: 0.0,
            ..Turtle2dConfig::default()
        };
        let scene = visualize_cpu(&contour, config, VisualizationContext::default()).unwrap();
        let polygon = scene
            .primitives
            .iter()
            .find_map(|primitive| match primitive {
                Primitive2d::Polygon(polygon) => Some(polygon),
                Primitive2d::Line(_) | Primitive2d::Text(_) => None,
            })
            .expect("contour polygon emitted");
        for (actual, expected) in
            polygon
                .vertices
                .iter()
                .copied()
                .zip([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        {
            assert!((actual.0 - expected.0).abs() < 1.0e-10);
            assert!((actual.1 - expected.1).abs() < 1.0e-10);
        }
        assert_eq!(
            scene
                .primitives
                .iter()
                .filter(|primitive| matches!(primitive, Primitive2d::Line(_)))
                .count(),
            2
        );

        let framework = Generation(vec![
            GenerationItem::Module(Module::new("PolygonBegin", Vec::new())),
            GenerationItem::Module(Module::new("Vertex", Vec::new())),
            GenerationItem::Module(Module::new("G", Vec::new())),
            GenerationItem::Module(Module::new("Vertex", Vec::new())),
            GenerationItem::Module(Module::new("Plus", Vec::new())),
            GenerationItem::Module(Module::new("G", Vec::new())),
            GenerationItem::Module(Module::new("Vertex", Vec::new())),
            GenerationItem::Module(Module::new("PolygonEnd", Vec::new())),
        ]);
        let config = Turtle2dConfig {
            initial_angle: 0.0,
            draw_modules: vec!["G".into()],
            ..Turtle2dConfig::default()
        };
        let scene = visualize_cpu(&framework, config, VisualizationContext::default()).unwrap();
        let polygon = scene
            .primitives
            .iter()
            .find_map(|primitive| match primitive {
                Primitive2d::Polygon(polygon) => Some(polygon),
                Primitive2d::Line(_) | Primitive2d::Text(_) => None,
            })
            .expect("explicit framework polygon emitted");
        assert_eq!(polygon.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    #[cfg(not(all(feature = "cuda", not(target_arch = "wasm32"))))]
    fn explicit_cuda_without_native_support_is_typed_unavailable() {
        let generation = Generation(vec![GenerationItem::Module(Module::new("F", Vec::new()))]);
        let error = crate::Turtle2dStreamer::new()
            .stream(
                VisualizerBackend::Cuda,
                Turtle2dStreamRequest {
                    generation: &generation,
                    config: Turtle2dConfig::default(),
                    batch_size: 16,
                    is_cancelled: &|| false,
                },
                |_| Ok(()),
            )
            .unwrap_err();
        assert_eq!(
            error,
            VisualizeError::BackendUnavailable {
                backend: VisualizerBackend::Cuda
            }
        );
    }
}
