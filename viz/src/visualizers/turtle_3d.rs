//! CPU reference interpreter for three-dimensional turtle systems.
//!
//! Turtle-world XYZ coordinates remain intact in [`crate::Scene3d`]. Display
//! backends may rotate that retained geometry interactively, while static
//! targets can use [`crate::Scene3d::canonical_projection`].

use crate::{
    Bounds3d, Line3d, Point3d, Polygon3d, Primitive3d, Scene3d, StrokeColor, StyledLine3d,
    Turtle3dBatch, Turtle3dConfig, Turtle3dPrimitiveKind, Turtle3dProgress, Turtle3dStreamRequest,
    Turtle3dStreamSummary, Turtle3dWidthReferenceEstimator, VisualizationContext, VisualizeError,
    VisualizerBackend, VisualizerKind, palette_color,
};
use braken::{Generation, GenerationItem, Module, Value};

const COMPATIBILITY_BATCH_SIZE: usize = 16 * 1024;

#[derive(Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn scale(self, scale: f64) -> Self {
        Self {
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
        }
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalized(self) -> Option<Self> {
        let length = self.length();
        (length.is_finite() && length > 1.0e-12).then(|| self.scale(length.recip()))
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    fn point(self) -> Point3d {
        (self.x, self.y, self.z)
    }
}

#[derive(Clone, Copy)]
struct State {
    position: Vec3,
    heading: Vec3,
    left: Vec3,
    up: Vec3,
    step_scale: f64,
    width: f64,
    turn_angle: f64,
    invert_turns: bool,
    color: StrokeColor,
    color_index: Option<i32>,
}

struct Frame<'a> {
    items: &'a [GenerationItem],
    next: usize,
    restore: Option<State>,
}

pub(crate) fn visualize_cpu(
    generation: &Generation,
    config: Turtle3dConfig,
    _: VisualizationContext,
) -> Result<Scene3d, VisualizeError> {
    let background = config.turtle.background;
    let mut primitives = Vec::new();
    stream_cpu(
        Turtle3dStreamRequest {
            generation,
            config,
            batch_size: COMPATIBILITY_BATCH_SIZE,
            is_cancelled: &|| false,
        },
        |batch| {
            let additional = batch.lines.len().saturating_add(batch.polygons.len());
            primitives
                .try_reserve(additional)
                .map_err(|_| resource_error("complete 3D turtle scene", additional))?;
            let mut lines = batch.lines.into_iter();
            let mut polygons = batch.polygons.into_iter();
            for kind in batch.primitive_order {
                let primitive = match kind {
                    Turtle3dPrimitiveKind::Line => Primitive3d::Line(
                        lines
                            .next()
                            .expect("3D turtle line order matches its payload"),
                    ),
                    Turtle3dPrimitiveKind::Polygon => Primitive3d::Polygon(
                        polygons
                            .next()
                            .expect("3D turtle polygon order matches its payload"),
                    ),
                };
                primitives.push(primitive);
            }
            debug_assert!(lines.next().is_none());
            debug_assert!(polygons.next().is_none());
            Ok(())
        },
    )?;
    Ok(Scene3d {
        primitives,
        background,
    })
}

pub(crate) fn stream_cpu(
    request: Turtle3dStreamRequest<'_>,
    mut emit: impl FnMut(Turtle3dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle3dStreamSummary, VisualizeError> {
    validate_request(&request)?;
    stream_cpu_validated(&request, &mut emit)
}

fn stream_cpu_validated(
    request: &Turtle3dStreamRequest<'_>,
    emit: &mut impl FnMut(Turtle3dBatch) -> Result<(), VisualizeError>,
) -> Result<Turtle3dStreamSummary, VisualizeError> {
    let config = &request.config.turtle;
    let generation = request.generation;
    let batch_size = request.batch_size;
    if (request.is_cancelled)() {
        return Err(VisualizeError::Cancelled);
    }
    let initial = config.initial_angle;
    let mut state = State {
        position: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        heading: Vec3 {
            x: initial.cos(),
            y: initial.sin(),
            z: 0.0,
        },
        left: Vec3 {
            x: -initial.sin(),
            y: initial.cos(),
            z: 0.0,
        },
        up: Vec3::Z,
        step_scale: 1.0,
        width: config.initial_width,
        turn_angle: config.turn_angle,
        invert_turns: false,
        color: config.initial_color,
        color_index: match config.initial_color {
            StrokeColor::PaletteIndex(index) => Some(i32::from(index)),
            StrokeColor::ThemeDefault | StrokeColor::Rgb(_) => None,
        },
    };
    let mut frames = Vec::new();
    frames
        .try_reserve(1)
        .map_err(|_| resource_error("turtle_3d traversal stack", 1))?;
    frames.push(Frame {
        items: generation.items(),
        next: 0,
        restore: None,
    });
    let mut polygon_stack = Vec::<(Vec<Point3d>, StrokeColor)>::new();
    let mut lines = new_line_batch(batch_size)?;
    let mut polygons = Vec::new();
    let mut primitive_order = new_primitive_order_batch(batch_size)?;
    let mut batch_bounds = None;
    let mut overall_bounds = None;
    let mut progress = Turtle3dProgress::default();
    let mut width_reference = Turtle3dWidthReferenceEstimator::default();
    let mut batch_items = 0usize;

    while !frames.is_empty() {
        if (request.is_cancelled)() {
            return Err(VisualizeError::Cancelled);
        }
        let item = {
            let frame = frames.last_mut().expect("checked non-empty");
            if frame.next == frame.items.len() {
                let restore = frame.restore;
                frames.pop();
                if let Some(saved) = restore {
                    state = saved;
                }
                progress.active_branch_depth = frames.len().saturating_sub(1);
                continue;
            }
            let item = &frame.items[frame.next];
            frame.next += 1;
            item
        };

        progress.items_processed =
            checked_increment(progress.items_processed, "turtle_3d processed-item counter")?;
        match item {
            GenerationItem::Branch(branch) => {
                progress.branches_entered = checked_increment(
                    progress.branches_entered,
                    "turtle_3d entered-branch counter",
                )?;
                let requested = frames
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| resource_error("turtle_3d traversal stack", usize::MAX))?;
                frames
                    .try_reserve(1)
                    .map_err(|_| resource_error("turtle_3d traversal stack", requested))?;
                frames.push(Frame {
                    items: branch.items(),
                    next: 0,
                    restore: Some(state),
                });
                progress.active_branch_depth = frames.len() - 1;
                progress.max_branch_depth =
                    progress.max_branch_depth.max(progress.active_branch_depth);
            }
            GenerationItem::Module(module) => {
                progress.modules_processed = checked_increment(
                    progress.modules_processed,
                    "turtle_3d processed-module counter",
                )?;
                match visit_module(module, config, &mut state, &mut polygon_stack)? {
                    VisitOutcome::Line(line) => {
                        include_bounds(&mut batch_bounds, line.line);
                        include_bounds(&mut overall_bounds, line.line);
                        width_reference.observe_line(&line);
                        lines.push(line);
                        primitive_order.push(Turtle3dPrimitiveKind::Line);
                        progress.lines_emitted = checked_increment(
                            progress.lines_emitted,
                            "turtle_3d emitted-line counter",
                        )?;
                    }
                    VisitOutcome::Polygon(polygon) => {
                        for &point in &polygon.vertices {
                            include_point(&mut batch_bounds, point);
                            include_point(&mut overall_bounds, point);
                        }
                        polygons.try_reserve(1).map_err(|_| {
                            resource_error(
                                "turtle_3d polygon batch",
                                polygons.len().saturating_add(1),
                            )
                        })?;
                        polygons.push(polygon);
                        primitive_order.push(Turtle3dPrimitiveKind::Polygon);
                        progress.polygons_emitted = checked_increment(
                            progress.polygons_emitted,
                            "turtle_3d emitted-polygon counter",
                        )?;
                    }
                    VisitOutcome::Cut => {
                        if let Some(frame) = frames.last_mut() {
                            frame.next = frame.items.len();
                        }
                    }
                    VisitOutcome::None => {}
                }
            }
        }

        batch_items = checked_increment(batch_items, "turtle_3d batch work counter")?;
        if batch_items == batch_size {
            deliver_batch(
                request,
                emit,
                &mut lines,
                &mut polygons,
                &mut primitive_order,
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
            &mut polygons,
            &mut primitive_order,
            &mut batch_bounds,
            overall_bounds,
            progress,
            false,
        )?;
    }
    if !polygon_stack.is_empty() {
        return Err(VisualizeError::InvalidConfiguration(
            "turtle_3d ended with an unterminated polygon".into(),
        ));
    }
    Ok(Turtle3dStreamSummary {
        bounds: overall_bounds,
        progress,
        width_reference: width_reference.width_reference(),
        backend_used: VisualizerBackend::Cpu,
    })
}

pub(crate) fn visualize_cuda(
    _: &Generation,
    _: Turtle3dConfig,
    _: VisualizationContext,
) -> Result<Scene3d, VisualizeError> {
    Err(VisualizeError::Unimplemented {
        visualizer: VisualizerKind::Turtle3d,
        backend: VisualizerBackend::Cuda,
    })
}

enum VisitOutcome {
    Line(StyledLine3d),
    Polygon(Polygon3d),
    Cut,
    None,
}

fn visit_module(
    module: &Module,
    config: &crate::Turtle2dConfig,
    state: &mut State,
    polygons: &mut Vec<(Vec<Point3d>, StrokeColor)>,
) -> Result<VisitOutcome, VisualizeError> {
    let name = resolve_action(module.name.as_str(), config)?;
    // ABOP assigns polygon-edge semantics specifically to F and f. A module
    // such as G can still draw a visible framework without changing the
    // polygon currently being captured.
    let traces_polygon_edge = matches!(name, "F" | "f");
    if name == "HalfForward" {
        return advance(
            module_distance(module, config) * state.step_scale * 0.5,
            true,
            state,
        );
    }
    if name == "HalfMove" {
        return advance(
            module_distance(module, config) * state.step_scale * 0.5,
            false,
            state,
        );
    }
    if config
        .draw_modules
        .iter()
        .any(|candidate| candidate == name)
    {
        return advance_with_polygon_capture(
            module_distance(module, config) * state.step_scale,
            true,
            state,
            traces_polygon_edge,
            polygons,
        );
    }
    if config
        .move_modules
        .iter()
        .any(|candidate| candidate == name)
    {
        return advance_with_polygon_capture(
            module_distance(module, config) * state.step_scale,
            false,
            state,
            traces_polygon_edge,
            polygons,
        );
    }

    let default_angle =
        |direction: f64| module_degrees(module).unwrap_or(state.turn_angle) * direction;
    match name {
        "+" | "Plus" | "ExplicitPlus" | "Left" | "Turn" => {
            yaw(state, default_angle(1.0));
        }
        "-" | "Minus" | "ExplicitMinus" | "Right" => {
            yaw(state, default_angle(-1.0));
        }
        "PitchUp" => pitch(state, default_angle(1.0)),
        "PitchDown" => pitch(state, default_angle(-1.0)),
        "RollLeft" => roll(state, default_angle(1.0)),
        "RollRight" => roll(state, default_angle(-1.0)),
        "TurnAround" | "Around" => yaw(state, std::f64::consts::PI),
        "InvertTurns" => state.invert_turns = !state.invert_turns,
        "Horizontal" | "Vertical" => align_to_vertical(state),
        "Scale" | "ScaleLength" => {
            state.step_scale *= module_number(module).unwrap_or(config.scale_multiplier);
        }
        "ScaleLengthInverse" => {
            state.step_scale *= module_number(module)
                .unwrap_or(config.scale_multiplier)
                .recip();
        }
        "Width" | "SetWidth" => state.width = required_non_negative(module, name)?,
        "WidthIncrease" if module.arguments.is_empty() => state.width += config.width_increment,
        "WidthDecrease" if module.arguments.is_empty() => {
            state.width = (state.width - config.width_increment).max(0.0);
        }
        "WidthIncrease" | "WidthDecrease" => {
            state.width = required_non_negative(module, name)?;
        }
        "TurnAngleIncrease" => {
            state.turn_angle += module_degrees(module).unwrap_or(config.turn_angle_increment);
        }
        "TurnAngleDecrease" => {
            state.turn_angle -= module_degrees(module).unwrap_or(config.turn_angle_increment);
        }
        "Color" => set_color(required_integer(module, name)?, config, state),
        "ColorIncrement" | "ColorNext" => {
            let delta = module_integer(module).unwrap_or(config.color_increment);
            change_color(delta, config, state)?;
        }
        "ColorPrevious" => {
            let delta = module_integer(module).unwrap_or(config.color_increment);
            change_color(-delta, config, state)?;
        }
        "Dot" => {
            return emit_dot(module_number(module).unwrap_or(1.0), state).map(VisitOutcome::Line);
        }
        // L3D objects and named surfaces need external asset data that the
        // normalized files do not carry. Emit a visible point marker so the
        // placement/color command is represented without inventing geometry.
        "Object" | "Surface" => {
            return emit_dot(module_number(module).unwrap_or(1.0), state).map(VisitOutcome::Line);
        }
        "PolygonBegin" => {
            polygons.try_reserve(1).map_err(|_| {
                resource_error("turtle_3d polygon stack", polygons.len().saturating_add(1))
            })?;
            polygons.push((Vec::new(), state.color));
        }
        "Vertex" => {
            let Some((vertices, _)) = polygons.last_mut() else {
                return Err(invalid("Vertex requires an active PolygonBegin"));
            };
            vertices.try_reserve(1).map_err(|_| {
                resource_error(
                    "turtle_3d polygon vertices",
                    vertices.len().saturating_add(1),
                )
            })?;
            vertices.push(state.position.point());
        }
        "PolygonEnd" => {
            let Some((vertices, color)) = polygons.pop() else {
                return Err(invalid("PolygonEnd requires an active PolygonBegin"));
            };
            if vertices.len() >= 3 {
                return Ok(VisitOutcome::Polygon(Polygon3d { vertices, color }));
            }
        }
        "Cut" => return Ok(VisitOutcome::Cut),
        // Query is a derivation/environment channel, not a drawing operation.
        // Its current value remains represented by its module arguments.
        _ => {}
    }
    ensure_finite(state)?;
    Ok(VisitOutcome::None)
}

fn advance(distance: f64, draw: bool, state: &mut State) -> Result<VisitOutcome, VisualizeError> {
    let next = state.position.add(state.heading.scale(distance));
    if !next.is_finite() {
        return Err(invalid("turtle_3d produced a non-finite coordinate"));
    }
    let line = draw.then_some(StyledLine3d {
        line: Line3d(state.position.point(), next.point()),
        width: state.width,
        color: state.color,
    });
    state.position = next;
    Ok(line.map_or(VisitOutcome::None, VisitOutcome::Line))
}

fn advance_with_polygon_capture(
    distance: f64,
    draw: bool,
    state: &mut State,
    traces_polygon_edge: bool,
    polygon_stack: &mut [(Vec<Point3d>, StrokeColor)],
) -> Result<VisitOutcome, VisualizeError> {
    let start = state.position.point();
    let outcome = advance(distance, draw, state)?;
    if traces_polygon_edge {
        capture_polygon_edge(polygon_stack, start, state.position.point())?;
    }
    Ok(outcome)
}

fn capture_polygon_edge(
    polygon_stack: &mut [(Vec<Point3d>, StrokeColor)],
    start: Point3d,
    end: Point3d,
) -> Result<(), VisualizeError> {
    let Some((vertices, _)) = polygon_stack.last_mut() else {
        return Ok(());
    };
    let include_start = vertices.last().copied() != Some(start);
    let additional = 1usize + usize::from(include_start);
    let requested = vertices
        .len()
        .checked_add(additional)
        .ok_or_else(|| resource_error("turtle_3d polygon vertices", usize::MAX))?;
    vertices
        .try_reserve(additional)
        .map_err(|_| resource_error("turtle_3d polygon vertices", requested))?;
    if include_start {
        vertices.push(start);
    }
    vertices.push(end);
    Ok(())
}

fn emit_dot(diameter: f64, state: &State) -> Result<StyledLine3d, VisualizeError> {
    if !diameter.is_finite() || diameter < 0.0 {
        return Err(invalid(
            "turtle_3d point diameter must be finite and non-negative",
        ));
    }
    let width = state.width * diameter;
    if !width.is_finite() {
        return Err(invalid("turtle_3d produced a non-finite point width"));
    }
    let point = state.position.point();
    Ok(StyledLine3d {
        line: Line3d(point, point),
        width,
        color: state.color,
    })
}

fn yaw(state: &mut State, angle: f64) {
    let angle = effective_angle(state, angle);
    let (sin, cos) = angle.sin_cos();
    let heading = state.heading;
    let left = state.left;
    state.heading = heading.scale(cos).add(left.scale(sin));
    state.left = left.scale(cos).add(heading.scale(-sin));
}

fn pitch(state: &mut State, angle: f64) {
    let angle = effective_angle(state, angle);
    let (sin, cos) = angle.sin_cos();
    let heading = state.heading;
    let up = state.up;
    state.heading = heading.scale(cos).add(up.scale(sin));
    state.up = up.scale(cos).add(heading.scale(-sin));
}

fn roll(state: &mut State, angle: f64) {
    let angle = effective_angle(state, angle);
    let (sin, cos) = angle.sin_cos();
    let left = state.left;
    let up = state.up;
    state.left = left.scale(cos).add(up.scale(sin));
    state.up = up.scale(cos).add(left.scale(-sin));
}

fn effective_angle(state: &State, angle: f64) -> f64 {
    if state.invert_turns { -angle } else { angle }
}

fn align_to_vertical(state: &mut State) {
    if let Some(left) = Vec3::Z.cross(state.heading).normalized() {
        state.left = left;
        if let Some(up) = state.heading.cross(left).normalized() {
            state.up = up;
        }
    }
}

fn set_color(index: i32, config: &crate::Turtle2dConfig, state: &mut State) {
    state.color_index = Some(index);
    state.color = palette_color(index, &config.palette);
}

fn change_color(
    delta: i32,
    config: &crate::Turtle2dConfig,
    state: &mut State,
) -> Result<(), VisualizeError> {
    let index = state
        .color_index
        .unwrap_or(0)
        .checked_add(delta)
        .ok_or_else(|| invalid("turtle_3d color index overflow"))?;
    set_color(index, config, state);
    Ok(())
}

fn resolve_action<'a>(
    module_name: &'a str,
    config: &'a crate::Turtle2dConfig,
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
        followed = followed.saturating_add(1);
        if followed > config.module_aliases.len() {
            return Err(invalid("turtle_3d render-map alias cycle"));
        }
        name = action;
    }
}

fn module_distance(module: &Module, config: &crate::Turtle2dConfig) -> f64 {
    module_number(module).unwrap_or(config.default_step)
}

fn module_number(module: &Module) -> Option<f64> {
    module.arguments.first().and_then(|value| match value {
        Value::Number(number) => Some(*number),
        _ => None,
    })
}

fn module_degrees(module: &Module) -> Option<f64> {
    module_number(module).map(f64::to_radians)
}

fn module_integer(module: &Module) -> Option<i32> {
    let value = module_number(module)?;
    (value.is_finite()
        && value.fract() == 0.0
        && value >= f64::from(i32::MIN)
        && value <= f64::from(i32::MAX))
    .then_some(value as i32)
}

fn required_number(module: &Module, command: &str) -> Result<f64, VisualizeError> {
    module_number(module)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid(&format!("{command} requires a finite numeric argument")))
}

fn required_non_negative(module: &Module, command: &str) -> Result<f64, VisualizeError> {
    let value = required_number(module, command)?;
    (value >= 0.0)
        .then_some(value)
        .ok_or_else(|| invalid(&format!("{command} requires a non-negative argument")))
}

fn required_integer(module: &Module, command: &str) -> Result<i32, VisualizeError> {
    module_integer(module)
        .ok_or_else(|| invalid(&format!("{command} requires an integer argument")))
}

fn validate_request(request: &Turtle3dStreamRequest<'_>) -> Result<(), VisualizeError> {
    validate_config(&request.config)?;
    if request.batch_size == 0 {
        return Err(invalid("turtle_3d batch size must be greater than zero"));
    }
    if (request.is_cancelled)() {
        return Err(VisualizeError::Cancelled);
    }
    Ok(())
}

fn validate_config(config: &Turtle3dConfig) -> Result<(), VisualizeError> {
    let config = &config.turtle;
    if config.turn_angle.is_finite()
        && config.initial_angle.is_finite()
        && config.default_step.is_finite()
        && config.scale_multiplier.is_finite()
        && config.initial_width.is_finite()
        && config.initial_width >= 0.0
        && config.width_increment.is_finite()
        && config.width_increment >= 0.0
        && config.turn_angle_increment.is_finite()
    {
        Ok(())
    } else {
        Err(invalid(
            "turtle_3d distances and angles must be finite, and widths must be finite and non-negative",
        ))
    }
}

fn ensure_finite(state: &State) -> Result<(), VisualizeError> {
    if state.position.is_finite()
        && state.heading.is_finite()
        && state.left.is_finite()
        && state.up.is_finite()
        && state.step_scale.is_finite()
        && state.width.is_finite()
        && state.turn_angle.is_finite()
    {
        Ok(())
    } else {
        Err(invalid("turtle_3d produced non-finite state"))
    }
}

fn new_line_batch(size: usize) -> Result<Vec<StyledLine3d>, VisualizeError> {
    let mut lines = Vec::new();
    lines
        .try_reserve_exact(size)
        .map_err(|_| resource_error("turtle_3d line batch", size))?;
    Ok(lines)
}

fn new_primitive_order_batch(size: usize) -> Result<Vec<Turtle3dPrimitiveKind>, VisualizeError> {
    let mut primitive_order = Vec::new();
    primitive_order
        .try_reserve_exact(size)
        .map_err(|_| resource_error("turtle_3d primitive-order batch", size))?;
    Ok(primitive_order)
}

#[allow(clippy::too_many_arguments)]
fn deliver_batch(
    request: &Turtle3dStreamRequest<'_>,
    emit: &mut impl FnMut(Turtle3dBatch) -> Result<(), VisualizeError>,
    lines: &mut Vec<StyledLine3d>,
    polygons: &mut Vec<Polygon3d>,
    primitive_order: &mut Vec<Turtle3dPrimitiveKind>,
    batch_bounds: &mut Option<Bounds3d>,
    overall_bounds: Option<Bounds3d>,
    progress: Turtle3dProgress,
    allocate_next: bool,
) -> Result<(), VisualizeError> {
    if (request.is_cancelled)() {
        return Err(VisualizeError::Cancelled);
    }
    debug_assert_eq!(primitive_order.len(), lines.len() + polygons.len());
    emit(Turtle3dBatch {
        lines: std::mem::take(lines),
        polygons: std::mem::take(polygons),
        primitive_order: std::mem::take(primitive_order),
        batch_bounds: batch_bounds.take(),
        overall_bounds,
        progress,
    })?;
    if allocate_next {
        *lines = new_line_batch(request.batch_size)?;
        *primitive_order = new_primitive_order_batch(request.batch_size)?;
    }
    Ok(())
}

fn include_bounds(bounds: &mut Option<Bounds3d>, line: Line3d) {
    if let Some(bounds) = bounds {
        bounds.include_line(line);
    } else {
        *bounds = Some(Bounds3d::from_line(line));
    }
}

fn include_point(bounds: &mut Option<Bounds3d>, point: Point3d) {
    include_bounds(bounds, Line3d(point, point));
}

fn checked_increment(value: usize, resource: &'static str) -> Result<usize, VisualizeError> {
    value
        .checked_add(1)
        .ok_or_else(|| resource_error(resource, usize::MAX))
}

fn resource_error(resource: &'static str, requested: usize) -> VisualizeError {
    VisualizeError::ResourceExhausted {
        resource,
        requested: Some(requested),
    }
}

fn invalid(message: &str) -> VisualizeError {
    VisualizeError::InvalidConfiguration(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use braken::{
        BackendChoice, CalculationLimits, CalculationRequest, CompiledGrammar, calculate,
    };

    fn module(name: &str) -> GenerationItem {
        GenerationItem::Module(Module::new(name, Vec::new()))
    }

    fn numeric_module(name: &str, value: f64) -> GenerationItem {
        GenerationItem::Module(Module::new(name, vec![Value::Number(value)]))
    }

    fn assert_point_near(actual: Point3d, expected: Point3d) {
        let tolerance = 1.0e-12;
        assert!((actual.0 - expected.0).abs() < tolerance, "{actual:?}");
        assert!((actual.1 - expected.1).abs() < tolerance, "{actual:?}");
        assert!((actual.2 - expected.2).abs() < tolerance, "{actual:?}");
    }

    fn scene_lines(scene: &Scene3d) -> Vec<StyledLine3d> {
        scene
            .primitives
            .iter()
            .filter_map(|primitive| match primitive {
                Primitive3d::Line(line) => Some(*line),
                Primitive3d::Polygon(_) => None,
            })
            .collect()
    }

    fn collect_stream(
        generation: &Generation,
        config: Turtle3dConfig,
        batch_size: usize,
    ) -> (Vec<StyledLine3d>, Vec<Polygon3d>, Turtle3dStreamSummary) {
        let mut lines = Vec::new();
        let mut polygons = Vec::new();
        let summary = stream_cpu(
            Turtle3dStreamRequest {
                generation,
                config,
                batch_size,
                is_cancelled: &|| false,
            },
            |batch| {
                lines.extend(batch.lines);
                polygons.extend(batch.polygons);
                Ok(())
            },
        )
        .unwrap();
        (lines, polygons, summary)
    }

    fn collect_ordered_primitives(
        generation: &Generation,
        config: Turtle3dConfig,
        batch_size: usize,
    ) -> (Vec<Primitive3d>, Turtle3dStreamSummary) {
        let mut primitives = Vec::new();
        let summary = stream_cpu(
            Turtle3dStreamRequest {
                generation,
                config,
                batch_size,
                is_cancelled: &|| false,
            },
            |batch| {
                let mut lines = batch.lines.into_iter();
                let mut polygons = batch.polygons.into_iter();
                for kind in batch.primitive_order {
                    primitives.push(match kind {
                        Turtle3dPrimitiveKind::Line => Primitive3d::Line(lines.next().unwrap()),
                        Turtle3dPrimitiveKind::Polygon => {
                            Primitive3d::Polygon(polygons.next().unwrap())
                        }
                    });
                }
                assert!(lines.next().is_none());
                assert!(polygons.next().is_none());
                Ok(())
            },
        )
        .unwrap();
        (primitives, summary)
    }

    #[test]
    fn pitch_retains_depth_and_branches_restore_the_full_turtle_state() {
        let generation = Generation(vec![
            module("F"),
            GenerationItem::Branch(Generation(vec![
                module("PitchUp"),
                numeric_module("Scale", 2.0),
                numeric_module("Width", 4.0),
                module("InvertTurns"),
                module("F"),
            ])),
            module("F"),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        let lines = scene_lines(&scene);

        assert_eq!(lines.len(), 3);
        assert_point_near(lines[0].line.0, (0.0, 0.0, 0.0));
        assert_point_near(lines[0].line.1, (1.0, 0.0, 0.0));
        assert_point_near(lines[1].line.0, (1.0, 0.0, 0.0));
        assert_point_near(lines[1].line.1, (1.0, 0.0, 2.0));
        assert_eq!(lines[1].width, 4.0);
        assert_point_near(lines[2].line.0, (1.0, 0.0, 0.0));
        assert_point_near(lines[2].line.1, (2.0, 0.0, 0.0));
        assert_eq!(lines[2].width, 1.0);
        assert_eq!(lines[2].color, StrokeColor::ThemeDefault);
    }

    #[test]
    fn yaw_pitch_roll_turnaround_and_planar_aliases_have_xyz_semantics() {
        let branch = |items| GenerationItem::Branch(Generation(items));
        let generation = Generation(vec![
            branch(vec![module("Plus"), module("F")]),
            branch(vec![module("PitchUp"), module("F")]),
            branch(vec![module("RollLeft"), module("Plus"), module("F")]),
            branch(vec![module("Around"), module("F")]),
            branch(vec![module("Left"), module("F")]),
            branch(vec![module("Right"), module("F")]),
            branch(vec![module("Turn"), module("F")]),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        let endpoints = scene_lines(&scene)
            .into_iter()
            .map(|line| line.line.1)
            .collect::<Vec<_>>();

        assert_eq!(endpoints.len(), 7);
        assert_point_near(endpoints[0], (0.0, 1.0, 0.0));
        assert_point_near(endpoints[1], (0.0, 0.0, 1.0));
        assert_point_near(endpoints[2], (0.0, 0.0, 1.0));
        assert_point_near(endpoints[3], (-1.0, 0.0, 0.0));
        assert_point_near(endpoints[4], (0.0, 1.0, 0.0));
        assert_point_near(endpoints[5], (0.0, -1.0, 0.0));
        assert_point_near(endpoints[6], (0.0, 1.0, 0.0));
    }

    #[test]
    fn streaming_is_bounded_reports_progress_only_batches_and_xyz_bounds() {
        let generation = Generation(vec![
            module("Move"),
            module("Plus"),
            module("Move"),
            module("F"),
            module("F"),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let mut batch_line_counts = Vec::new();
        let mut lines = Vec::new();
        let summary = stream_cpu(
            Turtle3dStreamRequest {
                generation: &generation,
                config,
                batch_size: 2,
                is_cancelled: &|| false,
            },
            |batch| {
                batch_line_counts.push(batch.lines.len());
                lines.extend(batch.lines);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(batch_line_counts, [0, 1, 1]);
        assert_eq!(summary.progress.items_processed, 5);
        assert_eq!(summary.progress.modules_processed, 5);
        assert_eq!(summary.progress.lines_emitted, 2);
        assert_eq!(summary.backend_used, VisualizerBackend::Cpu);
        assert_eq!(
            summary.bounds,
            Some(Bounds3d {
                min: (1.0, 1.0, 0.0),
                max: (1.0, 3.0, 0.0),
            })
        );
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn batch_size_does_not_change_geometry_or_progress() {
        let generation = Generation(vec![
            module("F"),
            GenerationItem::Branch(Generation(vec![
                module("PitchUp"),
                module("F"),
                module("Dot"),
            ])),
            module("Plus"),
            module("F"),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let expected = collect_stream(&generation, config.clone(), 64);

        for batch_size in [1, 2, 3, 5] {
            assert_eq!(
                collect_stream(&generation, config.clone(), batch_size),
                expected,
                "batch size {batch_size}"
            );
        }
    }

    #[test]
    fn batch_size_does_not_change_mixed_primitive_order() {
        let generation = Generation(vec![
            module("F"),
            module("PolygonBegin"),
            module("Vertex"),
            module("Vertex"),
            module("Vertex"),
            module("PolygonEnd"),
            module("F"),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let expected = collect_ordered_primitives(&generation, config.clone(), 64);

        assert!(matches!(
            expected.0.as_slice(),
            [
                Primitive3d::Line(_),
                Primitive3d::Polygon(_),
                Primitive3d::Line(_)
            ]
        ));
        for batch_size in [1, 2, 3, 5] {
            assert_eq!(
                collect_ordered_primitives(&generation, config.clone(), batch_size),
                expected,
                "batch size {batch_size}"
            );
        }

        let scene = visualize_cpu(&generation, config, VisualizationContext::default()).unwrap();
        assert_eq!(scene.primitives, expected.0);
    }

    #[test]
    fn polygon_only_batches_contribute_raw_xyz_bounds() {
        let generation = Generation(vec![
            module("PolygonBegin"),
            module("Vertex"),
            module("PitchUp"),
            module("Move"),
            module("Vertex"),
            module("Plus"),
            module("Move"),
            module("Vertex"),
            module("PolygonEnd"),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let (lines, polygons, summary) = collect_stream(&generation, config, 4);

        assert!(lines.is_empty());
        assert_eq!(polygons.len(), 1);
        for (actual, expected) in polygons[0].vertices.iter().copied().zip([
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]) {
            assert_point_near(actual, expected);
        }
        let bounds = summary.bounds.unwrap();
        assert_point_near(bounds.min, (0.0, 0.0, 0.0));
        assert_point_near(bounds.max, (0.0, 1.0, 1.0));
        assert_eq!(summary.progress.polygons_emitted, 1);
    }

    #[test]
    fn abop_forward_moves_trace_spatial_polygon_edges() {
        let generation = Generation(vec![
            module("PolygonBegin"),
            module("F"),
            module("PitchUp"),
            module("f"),
            module("Plus"),
            module("F"),
            module("PolygonEnd"),
        ]);
        let mut config = Turtle3dConfig::default();
        config.turtle.initial_angle = 0.0;
        let (lines, polygons, summary) = collect_stream(&generation, config, 16);

        assert_eq!(lines.len(), 2);
        assert_eq!(polygons.len(), 1);
        assert_eq!(polygons[0].vertices.len(), 4);
        assert_point_near(polygons[0].vertices[0], (0.0, 0.0, 0.0));
        assert_eq!(summary.progress.polygons_emitted, 1);
    }

    #[test]
    fn cancellation_drops_an_unflushed_batch() {
        let generation = Generation(vec![module("F"), module("F"), module("F")]);
        let calls = Cell::new(0usize);
        let emitted = Cell::new(0usize);
        let result = stream_cpu(
            Turtle3dStreamRequest {
                generation: &generation,
                config: Turtle3dConfig::default(),
                batch_size: 10,
                is_cancelled: &|| {
                    let next = calls.get() + 1;
                    calls.set(next);
                    next >= 4
                },
            },
            |_| {
                emitted.set(emitted.get() + 1);
                Ok(())
            },
        );

        assert_eq!(result, Err(VisualizeError::Cancelled));
        assert_eq!(emitted.get(), 0);
    }

    #[test]
    fn rejects_a_zero_batch_size() {
        let error = stream_cpu(
            Turtle3dStreamRequest {
                generation: &Generation(Vec::new()),
                config: Turtle3dConfig::default(),
                batch_size: 0,
                is_cancelled: &|| false,
            },
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(matches!(error, VisualizeError::InvalidConfiguration(_)));
    }

    #[test]
    fn impossible_batch_allocation_is_a_typed_resource_error() {
        let generation = Generation::default();
        let error = stream_cpu(
            Turtle3dStreamRequest {
                generation: &generation,
                config: Turtle3dConfig::default(),
                batch_size: usize::MAX,
                is_cancelled: &|| false,
            },
            |_| Ok(()),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            VisualizeError::ResourceExhausted {
                resource: "turtle_3d line batch",
                requested: Some(usize::MAX),
            }
        ));
    }

    #[test]
    fn explicit_stack_handles_deep_branches() {
        const DEPTH: usize = 5_000;
        let mut generation = Generation(vec![module("F")]);
        for _ in 0..DEPTH {
            generation = Generation(vec![GenerationItem::Branch(generation)]);
        }

        let (lines, _, summary) = collect_stream(&generation, Turtle3dConfig::default(), 8);
        assert_eq!(lines.len(), 1);
        assert_eq!(summary.progress.branches_entered, DEPTH);
        assert_eq!(summary.progress.max_branch_depth, DEPTH);

        // Generation itself has recursive drop semantics; avoid involving that
        // unrelated recursion in this traversal test.
        std::mem::forget(generation);
    }

    #[test]
    fn abop_hilbert_retains_all_three_axes() {
        let source = include_str!(
            "../../../lib/tests/fixtures/abop/pass/abop-025-three-dimensional-hilbert-curve.lsys"
        );
        let calculation = calculate(CalculationRequest {
            grammar: CompiledGrammar::parse(source).unwrap(),
            iterations: 2,
            backend: BackendChoice::Cpu,
            seed: 0,
            semantics: Default::default(),
            limits: CalculationLimits::default(),
        })
        .unwrap();
        let mut lines = Vec::new();
        let summary = stream_cpu(
            Turtle3dStreamRequest {
                generation: &calculation.generation,
                config: Turtle3dConfig::default(),
                batch_size: 7,
                is_cancelled: &|| false,
            },
            |batch| {
                lines.extend(batch.lines);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(lines.len(), 63);
        assert_eq!(summary.progress.lines_emitted, 63);
        assert_eq!(summary.width_reference, 1.0);
        let bounds = summary.bounds.unwrap();
        assert!(bounds.max.0 - bounds.min.0 > 0.5, "{bounds:?}");
        assert!(bounds.max.1 - bounds.min.1 > 0.5, "{bounds:?}");
        assert!(bounds.max.2 - bounds.min.2 > 0.5, "{bounds:?}");
        for line in lines {
            let changed_axes = [
                (line.line.1.0 - line.line.0.0).abs(),
                (line.line.1.1 - line.line.0.1).abs(),
                (line.line.1.2 - line.line.0.2).abs(),
            ]
            .into_iter()
            .filter(|delta| *delta > 1.0e-10)
            .count();
            assert_eq!(changed_axes, 1, "non-axis-aligned segment: {:?}", line.line);
        }
    }

    #[test]
    fn spatial_width_reference_is_length_weighted_and_bounds_display_multipliers() {
        let generation = Generation(vec![
            numeric_module("Width", 78.0),
            numeric_module("F", 1.0),
            numeric_module("Width", 93.0),
            numeric_module("F", 3.0),
        ]);
        let (_, _, summary) = collect_stream(&generation, Turtle3dConfig::default(), 2);

        let expected = (78.0 * 1.0 + 93.0 * 3.0) / 4.0;
        assert!((summary.width_reference - expected).abs() < 1.0e-12);
        assert!(
            (crate::normalized_turtle_3d_width(78.0, summary.width_reference) - 78.0 / expected)
                .abs()
                < 1.0e-12
        );
        assert!(
            (crate::normalized_turtle_3d_width(93.0, summary.width_reference) - 93.0 / expected)
                .abs()
                < 1.0e-12
        );
        assert_eq!(
            crate::normalized_turtle_3d_width(10_000.0, summary.width_reference),
            crate::MAX_NORMALIZED_TURTLE_3D_WIDTH
        );

        let mut huge = crate::Turtle3dWidthReferenceEstimator::default();
        huge.observe(f64::MAX, 1.0);
        huge.observe(f64::MAX, 3.0);
        huge.observe(f64::MAX, 5.0);
        assert_eq!(huge.width_reference(), 3.0);
    }
}
