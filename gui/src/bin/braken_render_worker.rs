#[cfg(target_arch = "wasm32")]
#[path = "../worker_protocol.rs"]
mod worker_protocol;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("braken-render-worker is only used by the wasm32 GUI build");
}

/// Tracks the cumulative line position used to compactly retain mixed turtle
/// primitive order while the line and polygon payloads stay type-separated.
#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Default)]
struct SpatialPrimitiveOrderCursor {
    lines_before: usize,
}

#[cfg(any(target_arch = "wasm32", test))]
impl SpatialPrimitiveOrderCursor {
    fn validate_batch(
        primitive_order: &[braken_viz::Turtle3dPrimitiveKind],
        line_count: usize,
        polygon_count: usize,
    ) -> Result<(), &'static str> {
        let expected = line_count
            .checked_add(polygon_count)
            .ok_or("turtle_3d batch primitive count overflow")?;
        let tagged_lines = primitive_order
            .iter()
            .filter(|kind| matches!(kind, braken_viz::Turtle3dPrimitiveKind::Line))
            .count();
        let tagged_polygons = primitive_order.len().saturating_sub(tagged_lines);
        if primitive_order.len() != expected
            || tagged_lines != line_count
            || tagged_polygons != polygon_count
        {
            return Err("turtle_3d batch primitive order did not match its payloads");
        }
        Ok(())
    }

    fn advance(
        &mut self,
        kind: braken_viz::Turtle3dPrimitiveKind,
    ) -> Result<Option<usize>, &'static str> {
        match kind {
            braken_viz::Turtle3dPrimitiveKind::Line => {
                self.lines_before = self
                    .lines_before
                    .checked_add(1)
                    .ok_or("turtle_3d cumulative line order overflow")?;
                Ok(None)
            }
            braken_viz::Turtle3dPrimitiveKind::Polygon => Ok(Some(self.lines_before)),
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::{JsCast, JsValue, closure::Closure};
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

    console_error_panic_hook::set_once();
    let scope = DedicatedWorkerGlobalScope::from(JsValue::from(js_sys::global()));
    let callback_scope = scope.clone();
    let runtime = Rc::new(RefCell::new(Some(WorkerRuntime::default())));
    let control = Rc::new(WorkerControl::default());
    let onmessage = Closure::wrap(Box::new(move |message: MessageEvent| {
        let result = message
            .data()
            .as_string()
            .ok_or_else(|| String::from("render worker expected a JSON string"))
            .and_then(|json| {
                serde_json::from_str::<worker_protocol::WorkerCommand>(&json)
                    .map_err(|error| format!("invalid worker command: {error}"))
            });

        match result {
            Ok(worker_protocol::WorkerCommand::Run(request)) => {
                let request_id = request.request_id;
                if !control.start(request_id) {
                    post_event(
                        &callback_scope,
                        &worker_protocol::WorkerEvent::Finished {
                            request_id,
                            result: Err(String::from("render worker received concurrent work")),
                        },
                    );
                    return;
                }
                let render_scope = callback_scope.clone();
                let render_runtime = Rc::clone(&runtime);
                let render_control = Rc::clone(&control);
                wasm_bindgen_futures::spawn_local(async move {
                    let mut runtime = render_runtime
                        .borrow_mut()
                        .take()
                        .expect("an idle worker owns its runtime");
                    let result = render(
                        request,
                        &render_scope,
                        &mut runtime,
                        Rc::clone(&render_control),
                    )
                    .await;
                    *render_runtime.borrow_mut() = Some(runtime);
                    let cancelled = render_control.finish(request_id);
                    if cancelled {
                        post_event(
                            &render_scope,
                            &worker_protocol::WorkerEvent::Cancelled { request_id },
                        );
                        return;
                    }
                    match result {
                        Ok((metadata, line_chunks, morph_chunks)) => {
                            if let Err(message) = post_finished(
                                &render_scope,
                                request_id,
                                metadata,
                                &line_chunks,
                                morph_chunks.as_ref(),
                            ) {
                                post_event(
                                    &render_scope,
                                    &worker_protocol::WorkerEvent::Finished {
                                        request_id,
                                        result: Err(message),
                                    },
                                );
                            }
                        }
                        Err(message) => post_event(
                            &render_scope,
                            &worker_protocol::WorkerEvent::Finished {
                                request_id,
                                result: Err(message),
                            },
                        ),
                    }
                });
            }
            Ok(worker_protocol::WorkerCommand::Cancel { request_id }) => {
                control.cancel(request_id);
            }
            Ok(worker_protocol::WorkerCommand::Shutdown) => {
                control.cancel_active();
                callback_scope.close();
            }
            Err(message) => post_event(
                &callback_scope,
                &worker_protocol::WorkerEvent::Fatal { message },
            ),
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    scope.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();
    post_event(&scope, &worker_protocol::WorkerEvent::Ready);
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WorkerControl {
    active: std::cell::Cell<Option<u64>>,
    cancelled: std::cell::Cell<Option<u64>>,
}

#[cfg(target_arch = "wasm32")]
impl WorkerControl {
    fn start(&self, request_id: u64) -> bool {
        if self.active.get().is_some() {
            return false;
        }
        self.active.set(Some(request_id));
        self.cancelled.set(None);
        true
    }

    fn cancel(&self, request_id: u64) {
        if self.active.get() == Some(request_id) {
            self.cancelled.set(Some(request_id));
        }
    }

    fn cancel_active(&self) {
        if let Some(request_id) = self.active.get() {
            self.cancelled.set(Some(request_id));
        }
    }

    fn is_cancelled(&self, request_id: u64) -> bool {
        self.cancelled.get() == Some(request_id)
    }

    fn finish(&self, request_id: u64) -> bool {
        let cancelled = self.is_cancelled(request_id);
        if self.active.get() == Some(request_id) {
            self.active.set(None);
        }
        if cancelled {
            self.cancelled.set(None);
        }
        cancelled
    }
}

#[cfg(target_arch = "wasm32")]
fn post_finished(
    scope: &web_sys::DedicatedWorkerGlobalScope,
    request_id: u64,
    metadata: worker_protocol::WorkerRenderResult,
    line_chunks: &js_sys::Array,
    morph_chunks: Option<&js_sys::Array>,
) -> Result<(), String> {
    use wasm_bindgen::{JsCast, JsValue};

    let layout = metadata.transfer_layout()?;
    validate_transfer_chunks(
        line_chunks,
        metadata.line_count,
        worker_protocol::LINE_TRANSFER_CHUNK_LINES,
        metadata.scene.line_transfer_values(),
        layout.line_chunks,
        layout.line_values,
        "line",
        true,
    )?;
    match (metadata.transition.as_ref(), morph_chunks) {
        (Some(transition), Some(morph_chunks)) => validate_transfer_chunks(
            morph_chunks,
            transition.morph_count,
            worker_protocol::MORPH_TRANSFER_CHUNK_LINES,
            worker_protocol::MORPH_TRANSFER_VALUES,
            layout.morph_chunks,
            layout.morph_values,
            "morph",
            false,
        )?,
        (None, None) => {}
        (Some(_), None) => {
            return Err(String::from(
                "render worker omitted transition morph buffers",
            ));
        }
        (None, Some(_)) => {
            return Err(String::from(
                "render worker attached morph buffers without transition metadata",
            ));
        }
    }

    let event = worker_protocol::WorkerEvent::Finished {
        request_id,
        result: Ok(metadata),
    };
    let event_json = serde_json::to_string(&event)
        .map_err(|error| format!("could not serialize render metadata: {error}"))?;
    let message = js_sys::Object::new();
    js_sys::Reflect::set(
        &message,
        &JsValue::from_str("metadata"),
        &JsValue::from_str(&event_json),
    )
    .map_err(|error| format!("could not attach render metadata: {error:?}"))?;
    js_sys::Reflect::set(&message, &JsValue::from_str("lines"), line_chunks)
        .map_err(|error| format!("could not attach rendered lines: {error:?}"))?;
    if let Some(morph_chunks) = morph_chunks {
        js_sys::Reflect::set(&message, &JsValue::from_str("morphs"), morph_chunks)
            .map_err(|error| format!("could not attach rendered morphs: {error:?}"))?;
    }
    let transfer = js_sys::Array::new();
    for value in line_chunks.iter() {
        let chunk = value
            .dyn_into::<js_sys::Float32Array>()
            .map_err(|_| String::from("rendered line chunk was not a Float32Array"))?;
        transfer.push(&chunk.buffer());
    }
    if let Some(morph_chunks) = morph_chunks {
        for value in morph_chunks.iter() {
            let chunk = value
                .dyn_into::<js_sys::Float32Array>()
                .map_err(|_| String::from("rendered morph chunk was not a Float32Array"))?;
            transfer.push(&chunk.buffer());
        }
    }
    scope
        .post_message_with_transfer(&message, &transfer)
        .map_err(|error| format!("could not transfer rendered lines: {error:?}"))
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn validate_transfer_chunks(
    chunks: &js_sys::Array,
    records: usize,
    records_per_chunk: usize,
    values_per_record: usize,
    expected_chunks: usize,
    expected_values: usize,
    kind: &str,
    validate_lines: bool,
) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    if chunks.length() as usize != expected_chunks {
        return Err(format!(
            "render worker produced {} {kind} chunks; expected {expected_chunks}",
            chunks.length(),
        ));
    }
    let mut actual_values = 0usize;
    for (chunk_index, value) in chunks.iter().enumerate() {
        let chunk = value
            .dyn_into::<js_sys::Float32Array>()
            .map_err(|_| format!("render worker {kind} chunk was not a Float32Array"))?;
        let expected_chunk_values = worker_protocol::expected_chunk_values(
            records,
            chunk_index,
            records_per_chunk,
            values_per_record,
            kind,
        )?;
        let chunk_values = chunk.length() as usize;
        if chunk_values != expected_chunk_values {
            return Err(format!(
                "render worker produced {chunk_values} values in {kind} chunk {chunk_index}; expected {expected_chunk_values}",
            ));
        }
        actual_values = actual_values
            .checked_add(chunk_values)
            .ok_or_else(|| format!("render worker {kind} transfer size overflow"))?;
        if validate_lines {
            validate_line_chunk(&chunk, values_per_record, kind)?;
        }
    }
    if actual_values != expected_values {
        return Err(format!(
            "render worker produced {actual_values} {kind} values; expected {expected_values}",
        ));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn validate_line_chunk(
    chunk: &js_sys::Float32Array,
    values_per_record: usize,
    kind: &str,
) -> Result<(), String> {
    let coordinate_values = match values_per_record {
        worker_protocol::LINE_2D_TRANSFER_VALUES => 4,
        worker_protocol::LINE_3D_TRANSFER_VALUES => 6,
        _ => {
            return Err(format!(
                "render worker {kind} chunk has an unknown line stride"
            ));
        }
    };
    let record_values = u32::try_from(values_per_record)
        .map_err(|_| format!("render worker {kind} line stride does not fit u32"))?;
    let coordinate_values = u32::try_from(coordinate_values)
        .map_err(|_| format!("render worker {kind} coordinate stride does not fit u32"))?;
    for offset in (0..chunk.length()).step_by(values_per_record) {
        for coordinate_offset in 0..coordinate_values {
            if !chunk.get_index(offset + coordinate_offset).is_finite() {
                return Err(format!(
                    "render worker {kind} chunk contains a non-finite coordinate"
                ));
            }
        }
        let width = chunk.get_index(offset + record_values - 2);
        if !width.is_finite() || width < 0.0 {
            return Err(format!(
                "render worker {kind} chunk contains an invalid line width"
            ));
        }
        let color = chunk.get_index(offset + record_values - 1);
        if !color.is_nan()
            && (!color.is_finite()
                || color.fract() != 0.0
                || color < -16_777_216.0_f32
                || color > f32::from(u16::MAX))
        {
            return Err(format!(
                "render worker {kind} chunk contains an invalid color token"
            ));
        }
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn post_event(scope: &web_sys::DedicatedWorkerGlobalScope, event: &worker_protocol::WorkerEvent) {
    let value = serde_json::to_string(event)
        .map(|json| wasm_bindgen::JsValue::from_str(&json))
        .unwrap_or_else(|_| {
            wasm_bindgen::JsValue::from_str(
                "{\"Fatal\":{\"message\":\"worker event serialization failed\"}}",
            )
        });
    let _result = scope.post_message(&value);
}

#[cfg(target_arch = "wasm32")]
struct LineChunkBuilder {
    chunks: js_sys::Array,
    pending: Vec<f32>,
    line_count: usize,
    bounds: Option<[f32; 4]>,
    width_estimator: braken_viz::targets::StrokeWidthEstimator,
    orientation_landmarks: braken_gui::orientation::OrientationLandmarks,
}

#[cfg(target_arch = "wasm32")]
impl LineChunkBuilder {
    fn new() -> Result<Self, String> {
        let mut pending = Vec::new();
        pending
            .try_reserve(
                worker_protocol::LINE_TRANSFER_CHUNK_LINES
                    .saturating_mul(worker_protocol::LINE_TRANSFER_VALUES),
            )
            .map_err(|_| String::from("not enough worker memory for a rendered line batch"))?;
        Ok(Self {
            chunks: js_sys::Array::new(),
            pending,
            line_count: 0,
            bounds: None,
            width_estimator: braken_viz::targets::StrokeWidthEstimator::default(),
            orientation_landmarks: braken_gui::orientation::OrientationLandmarks::default(),
        })
    }

    fn push(&mut self, line: braken_viz::StyledLine2d) -> Result<(), String> {
        if self
            .line_count
            .is_multiple_of(worker_protocol::LINE_TRANSFER_CHUNK_LINES)
            && !self.pending.is_empty()
        {
            self.flush();
        }
        self.pending
            .try_reserve(worker_protocol::LINE_TRANSFER_VALUES)
            .map_err(|_| {
                String::from("not enough worker memory to extend a rendered line batch")
            })?;
        let start_x = line.line.0.0 as f32;
        let start_y = line.line.0.1 as f32;
        let end_x = line.line.1.0 as f32;
        let end_y = line.line.1.1 as f32;
        let width = line.width as f32;
        self.width_estimator.observe(line.line);
        self.orientation_landmarks.observe(line.line.0, line.line.1);
        self.bounds = Some(match self.bounds {
            Some([min_x, max_x, min_y, max_y]) => [
                min_x.min(start_x).min(end_x),
                max_x.max(start_x).max(end_x),
                min_y.min(start_y).min(end_y),
                max_y.max(start_y).max(end_y),
            ],
            None => [
                start_x.min(end_x),
                start_x.max(end_x),
                start_y.min(end_y),
                start_y.max(end_y),
            ],
        });
        self.pending.extend_from_slice(&[
            start_x,
            start_y,
            end_x,
            end_y,
            width,
            worker_protocol::encode_stroke_color(line.color),
        ]);
        self.line_count = self
            .line_count
            .checked_add(1)
            .ok_or_else(|| String::from("rendered line count overflow"))?;
        Ok(())
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let chunk = js_sys::Float32Array::from(self.pending.as_slice());
        self.chunks.push(&chunk);
        self.pending.clear();
    }

    fn finish(
        mut self,
        orientation_anchor: Option<braken_gui::orientation::OrientationAnchor>,
        orientation_reference: f64,
    ) -> (js_sys::Array, usize, Option<[f32; 4]>, Option<f64>) {
        use wasm_bindgen::JsCast;

        self.flush();
        let transform = self
            .orientation_landmarks
            .transform(orientation_anchor, orientation_reference);
        if !transform.is_identity() {
            self.bounds = None;
            for value in self.chunks.iter() {
                let chunk = value.unchecked_into::<js_sys::Float32Array>();
                for offset in (0..chunk.length()).step_by(worker_protocol::LINE_TRANSFER_VALUES) {
                    let start = transform.apply((
                        f64::from(chunk.get_index(offset)),
                        f64::from(chunk.get_index(offset + 1)),
                    ));
                    let end = transform.apply((
                        f64::from(chunk.get_index(offset + 2)),
                        f64::from(chunk.get_index(offset + 3)),
                    ));
                    let [start_x, start_y, end_x, end_y] =
                        [start.0 as f32, start.1 as f32, end.0 as f32, end.1 as f32];
                    chunk.set_index(offset, start_x);
                    chunk.set_index(offset + 1, start_y);
                    chunk.set_index(offset + 2, end_x);
                    chunk.set_index(offset + 3, end_y);
                    self.bounds =
                        include_worker_line_bounds(self.bounds, start_x, start_y, end_x, end_y);
                }
            }
        }
        (
            self.chunks,
            self.line_count,
            self.bounds,
            self.width_estimator.total_line_length(),
        )
    }
}

#[cfg(target_arch = "wasm32")]
struct SpatialLineChunkBuilder {
    chunks: js_sys::Array,
    pending: Vec<f32>,
    line_count: usize,
    total_line_length: Option<f64>,
}

#[cfg(target_arch = "wasm32")]
impl SpatialLineChunkBuilder {
    fn new() -> Result<Self, String> {
        let capacity = worker_protocol::LINE_TRANSFER_CHUNK_LINES
            .checked_mul(worker_protocol::LINE_3D_TRANSFER_VALUES)
            .ok_or_else(|| String::from("browser spatial line chunk capacity overflow"))?;
        let mut pending = Vec::new();
        pending
            .try_reserve(capacity)
            .map_err(|_| String::from("not enough worker memory for a spatial line batch"))?;
        Ok(Self {
            chunks: js_sys::Array::new(),
            pending,
            line_count: 0,
            total_line_length: Some(0.0),
        })
    }

    fn push(&mut self, line: braken_viz::StyledLine3d) -> Result<(), braken_viz::VisualizeError> {
        if self
            .line_count
            .is_multiple_of(worker_protocol::LINE_TRANSFER_CHUNK_LINES)
            && !self.pending.is_empty()
        {
            self.flush();
        }
        let requested = self
            .pending
            .len()
            .checked_add(worker_protocol::LINE_3D_TRANSFER_VALUES)
            .ok_or_else(|| spatial_resource_error("browser worker 3D line chunk", usize::MAX))?;
        self.pending
            .try_reserve(worker_protocol::LINE_3D_TRANSFER_VALUES)
            .map_err(|_| spatial_resource_error("browser worker 3D line chunk", requested))?;
        let (x1, y1, z1) = line.line.0;
        let (x2, y2, z2) = line.line.1;
        if line.width < 0.0 {
            return Err(braken_viz::VisualizeError::InvalidConfiguration(
                String::from("turtle_3d produced a negative browser line width"),
            ));
        }
        let length = (x2 - x1).hypot(y2 - y1).hypot(z2 - z1);
        if length.is_finite() && length > 0.0 {
            self.total_line_length = self
                .total_line_length
                .and_then(|total| (total + length).is_finite().then_some(total + length));
        }
        let x1 = spatial_transfer_f32(x1)?;
        let y1 = spatial_transfer_f32(y1)?;
        let z1 = spatial_transfer_f32(z1)?;
        let x2 = spatial_transfer_f32(x2)?;
        let y2 = spatial_transfer_f32(y2)?;
        let z2 = spatial_transfer_f32(z2)?;
        let width = spatial_transfer_f32(line.width)?;
        self.pending.extend_from_slice(&[
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            width,
            worker_protocol::encode_stroke_color(line.color),
        ]);
        self.line_count = self
            .line_count
            .checked_add(1)
            .ok_or_else(|| spatial_resource_error("browser worker 3D line count", usize::MAX))?;
        Ok(())
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let chunk = js_sys::Float32Array::from(self.pending.as_slice());
        self.chunks.push(&chunk);
        self.pending.clear();
    }

    fn finish(mut self) -> (js_sys::Array, usize, Option<f64>) {
        self.flush();
        let total_line_length = self.total_line_length.filter(|length| *length > 0.0);
        (self.chunks, self.line_count, total_line_length)
    }
}

#[cfg(target_arch = "wasm32")]
fn encode_worker_polygon_3d(
    polygon: braken_viz::Polygon3d,
    lines_before: usize,
) -> Result<worker_protocol::WorkerPolygon3d, braken_viz::VisualizeError> {
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(polygon.vertices.len())
        .map_err(|_| {
            spatial_resource_error("browser worker 3D polygon vertices", polygon.vertices.len())
        })?;
    for (x, y, z) in polygon.vertices {
        // Polygon vertices stay f64 in the wire protocol, but the fitted
        // bounds are f32. Reject a scene that the display cannot fit instead
        // of transferring finite coordinates with infinite bounds.
        spatial_transfer_f32(x)?;
        spatial_transfer_f32(y)?;
        spatial_transfer_f32(z)?;
        vertices.push([x, y, z]);
    }
    Ok(worker_protocol::WorkerPolygon3d {
        vertices,
        color: polygon.color.into(),
        lines_before,
    })
}

#[cfg(target_arch = "wasm32")]
fn encode_worker_bounds_3d(
    bounds: braken_viz::Bounds3d,
) -> Result<[f32; 6], braken_viz::VisualizeError> {
    Ok([
        spatial_transfer_f32(bounds.min.0)?,
        spatial_transfer_f32(bounds.max.0)?,
        spatial_transfer_f32(bounds.min.1)?,
        spatial_transfer_f32(bounds.max.1)?,
        spatial_transfer_f32(bounds.min.2)?,
        spatial_transfer_f32(bounds.max.2)?,
    ])
}

#[cfg(target_arch = "wasm32")]
fn spatial_transfer_f32(value: f64) -> Result<f32, braken_viz::VisualizeError> {
    let encoded = value as f32;
    if value.is_finite() && encoded.is_finite() {
        Ok(encoded)
    } else {
        Err(braken_viz::VisualizeError::InvalidConfiguration(
            String::from("turtle_3d geometry is outside the browser f32 transfer range"),
        ))
    }
}

#[cfg(target_arch = "wasm32")]
fn spatial_resource_error(resource: &'static str, requested: usize) -> braken_viz::VisualizeError {
    braken_viz::VisualizeError::ResourceExhausted {
        resource,
        requested: Some(requested),
    }
}

#[cfg(target_arch = "wasm32")]
fn include_worker_line_bounds(
    bounds: Option<[f32; 4]>,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
) -> Option<[f32; 4]> {
    Some(match bounds {
        Some([min_x, max_x, min_y, max_y]) => [
            min_x.min(start_x).min(end_x),
            max_x.max(start_x).max(end_x),
            min_y.min(start_y).min(end_y),
            max_y.max(start_y).max(end_y),
        ],
        None => [
            start_x.min(end_x),
            start_x.max(end_x),
            start_y.min(end_y),
            start_y.max(end_y),
        ],
    })
}

#[cfg(target_arch = "wasm32")]
struct MorphChunkBuilder {
    chunks: js_sys::Array,
    pending: Vec<f32>,
    line_count: usize,
}

#[cfg(target_arch = "wasm32")]
impl MorphChunkBuilder {
    fn new() -> Result<Self, String> {
        let capacity = worker_protocol::MORPH_TRANSFER_CHUNK_LINES
            .checked_mul(worker_protocol::MORPH_TRANSFER_VALUES)
            .ok_or_else(|| String::from("browser morph chunk capacity overflow"))?;
        let mut pending = Vec::new();
        pending
            .try_reserve(capacity)
            .map_err(|_| String::from("not enough worker memory for a morph batch"))?;
        Ok(Self {
            chunks: js_sys::Array::new(),
            pending,
            line_count: 0,
        })
    }

    fn push(&mut self, line: braken_viz::MorphLine2d) -> Result<(), String> {
        if self
            .line_count
            .is_multiple_of(worker_protocol::MORPH_TRANSFER_CHUNK_LINES)
            && !self.pending.is_empty()
        {
            self.flush();
        }
        self.pending
            .try_reserve(worker_protocol::MORPH_TRANSFER_VALUES)
            .map_err(|_| String::from("not enough worker memory to extend a morph batch"))?;
        self.pending.extend_from_slice(&[
            line.source.line.0.0 as f32,
            line.source.line.0.1 as f32,
            line.source.line.1.0 as f32,
            line.source.line.1.1 as f32,
            line.source.width as f32,
            line.target.line.0.0 as f32,
            line.target.line.0.1 as f32,
            line.target.line.1.0 as f32,
            line.target.line.1.1 as f32,
            line.target.width as f32,
            line.source_opacity,
            line.target_opacity,
            line.source_palette_line.0.0 as f32,
            line.source_palette_line.0.1 as f32,
            line.source_palette_line.1.0 as f32,
            line.source_palette_line.1.1 as f32,
            line.target_palette_line.0.0 as f32,
            line.target_palette_line.0.1 as f32,
            line.target_palette_line.1.0 as f32,
            line.target_palette_line.1.1 as f32,
            worker_protocol::encode_stroke_color(line.source.color),
            worker_protocol::encode_stroke_color(line.target.color),
            worker_protocol::encode_morph_space(line.source_space),
            worker_protocol::encode_morph_space(line.target_space),
        ]);
        self.line_count = self
            .line_count
            .checked_add(1)
            .ok_or_else(|| String::from("rendered morph count overflow"))?;
        Ok(())
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let chunk = js_sys::Float32Array::from(self.pending.as_slice());
        self.chunks.push(&chunk);
        self.pending.clear();
    }

    fn finish(mut self) -> (js_sys::Array, usize) {
        self.flush();
        (self.chunks, self.line_count)
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WorkerRuntime {
    wgpu_backend: Option<braken::WgpuBackend>,
    cached_generation: Option<CachedGeneration>,
}

#[cfg(target_arch = "wasm32")]
#[derive(PartialEq, Eq)]
struct WorkerGenerationKey {
    source: String,
    ir_json: Option<String>,
    iterations: usize,
    seed: u64,
    semantics: braken::DerivationSemantics,
}

#[cfg(target_arch = "wasm32")]
fn worker_semantics(request: &worker_protocol::WorkerRenderRequest) -> braken::DerivationSemantics {
    braken::DerivationSemantics {
        float_width: match request.float_width {
            worker_protocol::WorkerFloatWidth::F32 => braken::FloatWidth::F32,
            worker_protocol::WorkerFloatWidth::F64 => braken::FloatWidth::F64,
        },
        ambiguous_rules: match request.ambiguous_rules {
            worker_protocol::WorkerAmbiguousRules::First => braken::AmbiguousRulePolicy::First,
            worker_protocol::WorkerAmbiguousRules::Error => braken::AmbiguousRulePolicy::Error,
            worker_protocol::WorkerAmbiguousRules::Uniform => braken::AmbiguousRulePolicy::Uniform,
        },
    }
}

#[cfg(target_arch = "wasm32")]
struct CachedGeneration {
    key: WorkerGenerationKey,
    generation: braken::Generation,
    stats: braken::ExecutionStats,
    backend: &'static str,
    lineage: Option<braken::RewriteLineage>,
}

#[cfg(target_arch = "wasm32")]
impl CachedGeneration {
    fn matches(&self, request: &worker_protocol::WorkerRenderRequest) -> bool {
        self.key.source == request.source
            && self.key.ir_json == request.ir_json
            && self.key.iterations == request.iterations
            && self.key.seed == request.seed
            && self.key.semantics == worker_semantics(request)
    }
}

#[cfg(target_arch = "wasm32")]
struct WorkerCalculation {
    generation: braken::Generation,
    stats: braken::ExecutionStats,
    backend: &'static str,
    lineage: Option<braken::RewriteLineage>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerProgressPhase {
    Calculation(braken::CalculationPhase),
    Visualizing,
}

#[cfg(target_arch = "wasm32")]
struct ProgressReporter<'a> {
    scope: &'a web_sys::DedicatedWorkerGlobalScope,
    request_id: u64,
    control: std::rc::Rc<WorkerControl>,
    last_phase: Option<WorkerProgressPhase>,
    last_sent_at: Option<web_time::Instant>,
}

#[cfg(target_arch = "wasm32")]
impl<'a> ProgressReporter<'a> {
    const MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

    fn new(
        scope: &'a web_sys::DedicatedWorkerGlobalScope,
        request_id: u64,
        control: std::rc::Rc<WorkerControl>,
    ) -> Self {
        Self {
            scope,
            request_id,
            control,
            last_phase: None,
            last_sent_at: None,
        }
    }

    fn calculation(&mut self, progress: braken::CalculationProgress) -> bool {
        let phase = WorkerProgressPhase::Calculation(progress.phase);
        let final_update = progress.phase == braken::CalculationPhase::Complete;
        self.post(
            phase,
            final_update,
            worker_protocol::WorkerEvent::Progress {
                request_id: self.request_id,
                phase: format!("{:?}", progress.phase),
                phase_completed: progress.phase_completed,
                phase_total: progress.phase_total,
                completed_iterations: progress.completed_iterations,
                total_iterations: progress.total_iterations,
                modules: progress.modules,
                items: progress.items,
                elapsed_millis: u64::try_from(progress.elapsed.as_millis()).unwrap_or(u64::MAX),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn visualizing(
        &mut self,
        phase_completed: usize,
        phase_total: usize,
        completed_iterations: usize,
        modules: usize,
        items: usize,
        elapsed: std::time::Duration,
    ) -> bool {
        self.post(
            WorkerProgressPhase::Visualizing,
            phase_completed >= phase_total,
            worker_protocol::WorkerEvent::Progress {
                request_id: self.request_id,
                phase: String::from("Visualizing"),
                phase_completed,
                phase_total: Some(phase_total),
                completed_iterations,
                total_iterations: completed_iterations,
                modules,
                items,
                elapsed_millis: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            },
        )
    }

    fn post(
        &mut self,
        phase: WorkerProgressPhase,
        final_update: bool,
        event: worker_protocol::WorkerEvent,
    ) -> bool {
        if self.control.is_cancelled(self.request_id) {
            return false;
        }
        let phase_changed = self.last_phase != Some(phase);
        let interval_elapsed = self
            .last_sent_at
            .as_ref()
            .is_none_or(|last| last.elapsed() >= Self::MIN_INTERVAL);
        if !phase_changed && !final_update && !interval_elapsed {
            // Suppression is not cancellation: callers retain every existing
            // internal cancellation checkpoint while reducing postMessage load.
            return true;
        }

        let sent = serde_json::to_string(&event)
            .ok()
            .and_then(|json| self.scope.post_message(&json.into()).ok())
            .is_some();
        if sent {
            self.last_phase = Some(phase);
            self.last_sent_at = Some(web_time::Instant::now());
        }
        // A best-effort progress message failing to serialize or post is not
        // cancellation. Only the explicit control flag may produce a
        // Cancelled result; the final result path will report its own posting
        // failure if the Worker channel itself is unavailable.
        true
    }
}

#[cfg(target_arch = "wasm32")]
async fn yield_to_worker_messages(scope: &web_sys::DedicatedWorkerGlobalScope) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if scope
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .is_err()
        {
            let _result = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _result = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
async fn calculate_cpu(
    grammar: braken::CompiledGrammar,
    iterations: usize,
    seed: u64,
    semantics: braken::DerivationSemantics,
    progress: &mut ProgressReporter<'_>,
    started: &web_time::Instant,
    scope: &web_sys::DedicatedWorkerGlobalScope,
    control: &WorkerControl,
    capture_last_lineage: bool,
) -> Result<WorkerCalculation, String> {
    use braken::{
        CalculationLimits, CalculationPhase, CalculationProgress, CpuBackend, CpuStepPhase,
        ExecutionStats,
    };

    let request_id = progress.request_id;
    let program = CpuBackend::new()
        .float_width(semantics.float_width)
        .ambiguous_rules(semantics.ambiguous_rules)
        .limits(CalculationLimits::default().into())
        .compile_ir(grammar.derivation_ir())
        .map_err(|error| error.to_string())?;
    let mut state = program.start_with_seed(seed);
    let mut modules = state.generation().module_count();
    let mut items = state.generation().item_count();
    if !progress.calculation(CalculationProgress {
        phase: CalculationPhase::Preparing,
        phase_completed: 0,
        phase_total: None,
        completed_iterations: 0,
        total_iterations: iterations,
        modules,
        items,
        elapsed: started.elapsed(),
    }) {
        return Err(String::from("CPU calculation cancelled"));
    }

    let mut lineage = None;
    for completed_iterations in 1..=iterations {
        // A zero-delay timer yields to the Worker's message queue. This is how
        // a posted Cancel command becomes visible to the cooperative checks.
        yield_to_worker_messages(scope).await;
        if control.is_cancelled(request_id) {
            return Err(String::from("CPU calculation cancelled"));
        }
        let input_modules = modules;
        let input_items = items;
        let callback_cancelled = std::cell::Cell::new(false);
        let traced = capture_last_lineage && completed_iterations == iterations;
        let (stats, completed_lineage) = if traced {
            let (stats, lineage) = state
                .step_with_lineage_and_control(
                    || control.is_cancelled(request_id) || callback_cancelled.get(),
                    |step| {
                        let phase = match step.phase {
                            CpuStepPhase::InspectingInput => CalculationPhase::InspectingInput,
                            CpuStepPhase::Indexing => CalculationPhase::Indexing,
                            CpuStepPhase::SelectingProductions => {
                                CalculationPhase::SelectingProductions
                            }
                            CpuStepPhase::Rewriting => CalculationPhase::Rewriting,
                            CpuStepPhase::ValidatingOutput => CalculationPhase::ValidatingOutput,
                        };
                        if !progress.calculation(CalculationProgress {
                            phase,
                            phase_completed: step.completed_items,
                            phase_total: step.total_items,
                            completed_iterations: completed_iterations - 1,
                            total_iterations: iterations,
                            modules: input_modules,
                            items: input_items,
                            elapsed: started.elapsed(),
                        }) {
                            callback_cancelled.set(true);
                        }
                    },
                )
                .map_err(|error| error.to_string())?;
            (stats, Some(lineage))
        } else {
            let stats = state
                .step_with_control(
                    || control.is_cancelled(request_id) || callback_cancelled.get(),
                    |step| {
                        let phase = match step.phase {
                            CpuStepPhase::InspectingInput => CalculationPhase::InspectingInput,
                            CpuStepPhase::Indexing => CalculationPhase::Indexing,
                            CpuStepPhase::SelectingProductions => {
                                CalculationPhase::SelectingProductions
                            }
                            CpuStepPhase::Rewriting => CalculationPhase::Rewriting,
                            CpuStepPhase::ValidatingOutput => CalculationPhase::ValidatingOutput,
                        };
                        if !progress.calculation(CalculationProgress {
                            phase,
                            phase_completed: step.completed_items,
                            phase_total: step.total_items,
                            completed_iterations: completed_iterations - 1,
                            total_iterations: iterations,
                            modules: input_modules,
                            items: input_items,
                            elapsed: started.elapsed(),
                        }) {
                            callback_cancelled.set(true);
                        }
                    },
                )
                .map_err(|error| error.to_string())?;
            (stats, None)
        };
        lineage = completed_lineage.or(lineage);
        modules = stats.output_modules;
        items = stats.output_items;
        if !progress.calculation(CalculationProgress {
            phase: if completed_iterations == iterations {
                CalculationPhase::Complete
            } else {
                CalculationPhase::Preparing
            },
            phase_completed: completed_iterations,
            phase_total: Some(iterations),
            completed_iterations,
            total_iterations: iterations,
            modules,
            items,
            elapsed: started.elapsed(),
        }) {
            return Err(String::from("CPU calculation cancelled"));
        }
    }

    let generation = state.into_generation();
    Ok(WorkerCalculation {
        stats: ExecutionStats {
            iterations,
            modules: generation.module_count(),
            items: generation.item_count(),
            elapsed: started.elapsed(),
        },
        generation,
        backend: "CPU",
        lineage,
    })
}

#[cfg(target_arch = "wasm32")]
fn is_wgpu_preflight_fallback(error: &braken::WgpuError) -> bool {
    use braken::WgpuError;

    matches!(
        error,
        WgpuError::AdapterUnavailable(_)
            | WgpuError::SoftwareAdapter { .. }
            | WgpuError::RequestDevice(_)
            | WgpuError::UnsupportedGrammar(_)
            | WgpuError::ResourceExhausted { .. }
            | WgpuError::DeviceLost(_)
            | WgpuError::OutOfMemory(_)
            | WgpuError::Validation(_)
            | WgpuError::Internal(_)
            | WgpuError::MapFailed(_)
    )
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
async fn calculate_wgpu(
    program: braken::WgpuProgram,
    iterations: usize,
    seed: u64,
    progress: &mut ProgressReporter<'_>,
    started: &web_time::Instant,
    scope: &web_sys::DedicatedWorkerGlobalScope,
    control: &WorkerControl,
    capture_last_lineage: bool,
) -> Result<WorkerCalculation, braken::WgpuError> {
    use braken::{CalculationPhase, CalculationProgress, ExecutionStats, WgpuError};

    let axiom = program.axiom()?;
    let mut modules = axiom.module_count();
    let mut items = axiom.item_count();
    drop(axiom);
    if !progress.calculation(CalculationProgress {
        phase: CalculationPhase::Preparing,
        phase_completed: 0,
        phase_total: None,
        completed_iterations: 0,
        total_iterations: iterations,
        modules,
        items,
        elapsed: started.elapsed(),
    }) {
        return Err(WgpuError::Cancelled);
    }

    // From state creation onward, errors are surfaced. Restarting a large
    // derivation on the CPU could duplicate work and hide a lost device or an
    // out-of-memory failure.
    let mut state = program.start_with_seed(seed)?;
    let mut lineage = None;
    for completed_iterations in 1..=iterations {
        // WGPU futures yield while submitted work completes. The explicit
        // timer also covers iterations that complete synchronously.
        yield_to_worker_messages(scope).await;
        if control.is_cancelled(progress.request_id) {
            return Err(WgpuError::Cancelled);
        }
        if !progress.calculation(CalculationProgress {
            phase: CalculationPhase::SelectingProductions,
            phase_completed: completed_iterations - 1,
            phase_total: Some(iterations),
            completed_iterations: completed_iterations - 1,
            total_iterations: iterations,
            modules,
            items,
            elapsed: started.elapsed(),
        }) {
            return Err(WgpuError::Cancelled);
        }
        if capture_last_lineage && completed_iterations == iterations {
            lineage = Some(
                state
                    .step_with_lineage_and_cancel(&mut || control.is_cancelled(progress.request_id))
                    .await?,
            );
        } else {
            state
                .step_with_cancel(&mut || control.is_cancelled(progress.request_id))
                .await?;
        }
        // Exact module/item counts require downloading and decoding. Keep the
        // generation device-resident and report its flat token count while it
        // is still computing.
        modules = state.token_count();
        items = state.token_count();
        if !progress.calculation(CalculationProgress {
            phase: CalculationPhase::Rewriting,
            phase_completed: completed_iterations,
            phase_total: Some(iterations),
            completed_iterations,
            total_iterations: iterations,
            modules,
            items,
            elapsed: started.elapsed(),
        }) {
            return Err(WgpuError::Cancelled);
        }
    }

    if !progress.calculation(CalculationProgress {
        phase: CalculationPhase::Transferring,
        phase_completed: iterations,
        phase_total: Some(iterations),
        completed_iterations: iterations,
        total_iterations: iterations,
        modules,
        items,
        elapsed: started.elapsed(),
    }) {
        return Err(WgpuError::Cancelled);
    }
    let generation = state
        .into_generation_with_cancel(&mut || control.is_cancelled(progress.request_id))
        .await?;
    modules = generation.module_count();
    items = generation.item_count();
    if !progress.calculation(CalculationProgress {
        phase: CalculationPhase::Complete,
        phase_completed: iterations,
        phase_total: Some(iterations),
        completed_iterations: iterations,
        total_iterations: iterations,
        modules,
        items,
        elapsed: started.elapsed(),
    }) {
        return Err(WgpuError::Cancelled);
    }

    Ok(WorkerCalculation {
        generation,
        stats: ExecutionStats {
            iterations,
            modules,
            items,
            elapsed: started.elapsed(),
        },
        backend: "WGPU",
        lineage,
    })
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
async fn calculate_auto(
    grammar: braken::CompiledGrammar,
    iterations: usize,
    seed: u64,
    semantics: braken::DerivationSemantics,
    runtime: &mut WorkerRuntime,
    progress: &mut ProgressReporter<'_>,
    scope: &web_sys::DedicatedWorkerGlobalScope,
    control: &WorkerControl,
    capture_last_lineage: bool,
) -> Result<WorkerCalculation, String> {
    use braken::{CalculationLimits, WgpuBackend};

    let started = web_time::Instant::now();
    let backend = match runtime.wgpu_backend.clone() {
        Some(backend) => backend,
        None => match WgpuBackend::request().await {
            Ok(backend) => {
                let backend = backend.limits(CalculationLimits::default());
                runtime.wgpu_backend = Some(backend.clone());
                backend
            }
            Err(error) if is_wgpu_preflight_fallback(&error) => {
                return calculate_cpu(
                    grammar,
                    iterations,
                    seed,
                    semantics,
                    progress,
                    &started,
                    scope,
                    control,
                    capture_last_lineage,
                )
                .await;
            }
            Err(error) => {
                return Err(format!(
                    "WebGPU preflight failed and was not safe to retry on CPU: {error}"
                ));
            }
        },
    };
    let backend = backend
        .float_width(semantics.float_width)
        .ambiguous_rules(semantics.ambiguous_rules);
    let program = match backend.compile_ir(grammar.derivation_ir()) {
        Ok(program) => program,
        Err(error) if is_wgpu_preflight_fallback(&error) => {
            return calculate_cpu(
                grammar,
                iterations,
                seed,
                semantics,
                progress,
                &started,
                scope,
                control,
                capture_last_lineage,
            )
            .await;
        }
        Err(error) => {
            runtime.wgpu_backend = None;
            return Err(format!(
                "WebGPU compilation failed and was not safe to retry on CPU: {error}"
            ));
        }
    };

    match calculate_wgpu(
        program,
        iterations,
        seed,
        progress,
        &started,
        scope,
        control,
        capture_last_lineage,
    )
    .await
    {
        Ok(calculation) => Ok(calculation),
        Err(braken::WgpuError::Cancelled) => Err(String::from("WebGPU calculation cancelled")),
        Err(error) => {
            // Never retry the same request after execution began. A future
            // request may obtain a fresh device instead of inheriting a lost
            // or otherwise unhealthy runtime.
            runtime.wgpu_backend = None;
            Err(format!(
                "WebGPU calculation failed after hardware execution was selected; CPU fallback was not attempted: {error}"
            ))
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
async fn render_transition(
    request: worker_protocol::WorkerRenderRequest,
    transition: worker_protocol::WorkerIterationTransition,
    scope: &web_sys::DedicatedWorkerGlobalScope,
    runtime: &mut WorkerRuntime,
    control: std::rc::Rc<WorkerControl>,
    started: &web_time::Instant,
    progress: &mut ProgressReporter<'_>,
) -> Result<
    (
        worker_protocol::WorkerRenderResult,
        js_sys::Array,
        Option<js_sys::Array>,
    ),
    String,
> {
    use braken_viz::{Turtle2dConfig, build_line_transition};

    if transition.from_iteration.abs_diff(transition.to_iteration) != 1
        || request.visualizer != "turtle_2d"
    {
        return Err(String::from(
            "iteration deformation requires adjacent Turtle 2D generations",
        ));
    }
    let request_id = request.request_id;
    let grammar = worker_compiled_grammar(&request)?;
    let (ir_disassembly, ir_json) = worker_ir_tooling(&grammar, worker_semantics(&request));
    let orientation_anchor = request
        .orientation_anchor
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(|error: braken_gui::orientation::ParseOrientationAnchorError| error.to_string())?;
    let reverse = transition.from_iteration > transition.to_iteration;
    let source_calculation = take_or_calculate_worker_generation(
        &request,
        transition.from_iteration,
        grammar.clone(),
        runtime,
        progress,
        scope,
        &control,
        reverse,
    )
    .await?;
    let target_calculation = take_or_calculate_worker_generation(
        &request,
        transition.to_iteration,
        grammar.clone(),
        runtime,
        progress,
        scope,
        &control,
        true,
    )
    .await?;
    yield_to_worker_messages(scope).await;
    if control.is_cancelled(request_id) {
        return Err(String::from("browser transition cancelled"));
    }

    let lineage = if reverse {
        source_calculation.lineage.as_ref()
    } else {
        target_calculation.lineage.as_ref()
    }
    .ok_or_else(|| String::from("browser derivation omitted requested rewrite lineage"))?;

    let turtle = Turtle2dConfig {
        turn_angle: (request.angle as f64).to_radians(),
        initial_angle: request.turtle.initial_angle,
        default_step: request.turtle.default_step,
        scale_multiplier: request.turtle.scale_multiplier,
        initial_width: request.turtle.initial_width,
        width_increment: request.turtle.width_increment,
        turn_angle_increment: request.turtle.turn_angle_increment,
        initial_color: request.turtle.initial_color.into(),
        color_increment: request.turtle.color_increment,
        palette: request.turtle.palette.clone(),
        background: request.turtle.background,
        draw_modules: request.turtle.draw_modules.clone(),
        move_modules: request.turtle.move_modules.clone(),
        module_aliases: request.turtle.module_aliases.clone(),
    };
    let orientation_reference = turtle.initial_angle;
    let source_lines = collect_indexed_worker_lines(
        &source_calculation.generation,
        turtle.clone(),
        &source_calculation.stats,
        request_id,
        &control,
        progress,
        started,
        !reverse,
        orientation_anchor,
        orientation_reference,
    )?;
    let target_lines = collect_indexed_worker_lines(
        &target_calculation.generation,
        turtle,
        &target_calculation.stats,
        request_id,
        &control,
        progress,
        started,
        reverse,
        orientation_anchor,
        orientation_reference,
    )?;
    let morphs = if reverse {
        build_line_transition(
            &target_lines.lines,
            &target_lines.module_positions,
            &source_lines.lines,
            lineage,
            true,
        )
    } else {
        build_line_transition(
            &source_lines.lines,
            &source_lines.module_positions,
            &target_lines.lines,
            lineage,
            false,
        )
    }
    .map_err(|error| error.to_string())?;

    let mut target_chunks = LineChunkBuilder::new()?;
    for line in &target_lines.lines {
        if control.is_cancelled(request_id) {
            return Err(String::from("browser transition cancelled"));
        }
        target_chunks.push(line.line)?;
    }
    let (line_chunks, line_count, target_bounds, total_line_length) =
        target_chunks.finish(None, 0.0);
    let mut morph_chunks = MorphChunkBuilder::new()?;
    for line in morphs {
        if control.is_cancelled(request_id) {
            return Err(String::from("browser transition cancelled"));
        }
        morph_chunks.push(line)?;
    }
    let (morph_chunks, morph_count) = morph_chunks.finish();

    let target_backend = target_calculation.backend;
    let target_stats = target_calculation.stats.clone();
    let target_lineage = target_calculation.lineage.clone();
    let semantics = worker_semantics(&request);
    runtime.cached_generation = Some(CachedGeneration {
        key: WorkerGenerationKey {
            source: request.source,
            ir_json: request.ir_json,
            iterations: transition.to_iteration,
            seed: request.seed,
            semantics,
        },
        generation: target_calculation.generation,
        stats: target_stats,
        backend: target_backend,
        lineage: target_lineage,
    });

    Ok((
        worker_protocol::WorkerRenderResult {
            line_count,
            total_line_length,
            width_reference: None,
            scene: worker_protocol::WorkerScene::TwoD {
                polygons: Vec::new(),
                texts: Vec::new(),
                bounds: target_bounds,
            },
            elapsed_millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            derivation_backend: target_backend.to_owned(),
            visualization_backend: String::from("CPU"),
            ir_disassembly,
            ir_json,
            background: request.turtle.background,
            transition: Some(worker_protocol::WorkerTransitionResult {
                from_iteration: transition.from_iteration,
                to_iteration: transition.to_iteration,
                morph_count,
                source_bounds: source_lines.bounds,
                source_line_count: source_lines.lines.len(),
                source_total_line_length: source_lines.total_line_length,
            }),
        },
        line_chunks,
        Some(morph_chunks),
    ))
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
async fn take_or_calculate_worker_generation(
    request: &worker_protocol::WorkerRenderRequest,
    iterations: usize,
    grammar: braken::CompiledGrammar,
    runtime: &mut WorkerRuntime,
    progress: &mut ProgressReporter<'_>,
    scope: &web_sys::DedicatedWorkerGlobalScope,
    control: &WorkerControl,
    capture_last_lineage: bool,
) -> Result<WorkerCalculation, String> {
    if runtime.cached_generation.as_ref().is_some_and(|cached| {
        cached.key.source == request.source
            && cached.key.iterations == iterations
            && cached.key.seed == request.seed
            && cached.key.semantics == worker_semantics(request)
            && (!capture_last_lineage || cached.lineage.is_some())
    }) {
        let cached = runtime
            .cached_generation
            .take()
            .expect("matching browser generation cache entry");
        return Ok(WorkerCalculation {
            generation: cached.generation,
            stats: cached.stats,
            backend: cached.backend,
            lineage: cached.lineage,
        });
    }
    runtime.cached_generation = None;
    let semantics = worker_semantics(request);
    calculate_auto(
        grammar,
        iterations,
        request.seed,
        semantics,
        runtime,
        progress,
        scope,
        control,
        capture_last_lineage,
    )
    .await
}

#[cfg(target_arch = "wasm32")]
struct IndexedWorkerLines {
    lines: Vec<braken_viz::IndexedLine2d>,
    module_positions: Vec<braken_viz::IndexedModulePosition2d>,
    bounds: Option<[f32; 4]>,
    total_line_length: Option<f64>,
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn collect_indexed_worker_lines(
    generation: &braken::Generation,
    config: braken_viz::Turtle2dConfig,
    stats: &braken::ExecutionStats,
    request_id: u64,
    control: &WorkerControl,
    progress: &mut ProgressReporter<'_>,
    started: &web_time::Instant,
    capture_module_positions: bool,
    orientation_anchor: Option<braken_gui::orientation::OrientationAnchor>,
    orientation_reference: f64,
) -> Result<IndexedWorkerLines, String> {
    use braken_viz::{IndexedLine2d, Turtle2dStreamRequest, VisualizeError, stream_turtle_2d};

    let mut lines = Vec::new();
    let mut module_positions = Vec::new();
    let mut width_estimator = braken_viz::targets::StrokeWidthEstimator::default();
    let summary = stream_turtle_2d(
        Turtle2dStreamRequest {
            generation,
            config,
            batch_size: 16 * 1024,
            is_cancelled: &|| control.is_cancelled(request_id),
        },
        |batch| {
            if !batch.polygons.is_empty() {
                return Err(VisualizeError::UnsupportedInput {
                    visualizer: braken_viz::VisualizerKind::Turtle2d,
                    backend: braken_viz::VisualizerBackend::Cpu,
                    reason: String::from(
                        "iteration deformation is unavailable for filled polygons",
                    ),
                });
            }
            if batch.lines.len() != batch.module_indices.len() {
                return Err(VisualizeError::InvalidConfiguration(String::from(
                    "turtle line/module-index batch length mismatch",
                )));
            }
            lines.try_reserve(batch.lines.len()).map_err(|_| {
                VisualizeError::ResourceExhausted {
                    resource: "browser indexed turtle scene",
                    requested: Some(lines.len().saturating_add(batch.lines.len())),
                }
            })?;
            lines.extend(batch.lines.into_iter().zip(batch.module_indices).map(
                |(line, module_index)| {
                    width_estimator.observe(line.line);
                    IndexedLine2d { module_index, line }
                },
            ));
            if capture_module_positions {
                module_positions
                    .try_reserve(batch.module_positions.len())
                    .map_err(|_| VisualizeError::ResourceExhausted {
                        resource: "browser indexed turtle module positions",
                        requested: Some(
                            module_positions
                                .len()
                                .saturating_add(batch.module_positions.len()),
                        ),
                    })?;
                module_positions.extend(batch.module_positions);
            }
            if !progress.visualizing(
                batch.progress.items_processed,
                stats.items,
                stats.iterations,
                batch.progress.modules_processed,
                batch.progress.items_processed,
                started.elapsed(),
            ) {
                return Err(VisualizeError::Cancelled);
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?;
    let mut bounds = summary.bounds.map(|bounds| {
        [
            bounds.min.0 as f32,
            bounds.max.0 as f32,
            bounds.min.1 as f32,
            bounds.max.1 as f32,
        ]
    });
    let mut landmarks = braken_gui::orientation::OrientationLandmarks::default();
    for line in &lines {
        landmarks.observe(line.line.line.0, line.line.line.1);
    }
    let transform = landmarks.transform(orientation_anchor, orientation_reference);
    if !transform.is_identity() {
        bounds = None;
        for line in &mut lines {
            line.line.line.0 = transform.apply(line.line.line.0);
            line.line.line.1 = transform.apply(line.line.line.1);
            bounds = include_worker_line_bounds(
                bounds,
                line.line.line.0.0 as f32,
                line.line.line.0.1 as f32,
                line.line.line.1.0 as f32,
                line.line.line.1.1 as f32,
            );
        }
        for position in &mut module_positions {
            position.position = transform.apply(position.position);
        }
    }
    Ok(IndexedWorkerLines {
        lines,
        module_positions,
        bounds,
        total_line_length: width_estimator.total_line_length(),
    })
}

#[cfg(target_arch = "wasm32")]
async fn render(
    request: worker_protocol::WorkerRenderRequest,
    scope: &web_sys::DedicatedWorkerGlobalScope,
    runtime: &mut WorkerRuntime,
    control: std::rc::Rc<WorkerControl>,
) -> Result<
    (
        worker_protocol::WorkerRenderResult,
        js_sys::Array,
        Option<js_sys::Array>,
    ),
    String,
> {
    use braken::{CalculationPhase, CalculationProgress};
    use braken_viz::{
        Primitive2d, TextRole, Turtle2dConfig, Turtle2dStreamRequest, Turtle3dStreamRequest,
        Visualization, VisualizationContext, VisualizeError, VisualizeRequest, VisualizerBackend,
        VisualizerConfig, VisualizerKind, stream_turtle_2d, stream_turtle_3d, visualize,
    };
    use std::str::FromStr;
    use web_time::Instant;
    use worker_protocol::{WorkerText, WorkerTextRole};

    let started = Instant::now();
    let request_id = request.request_id;
    let mut progress = ProgressReporter::new(scope, request_id, std::rc::Rc::clone(&control));
    if let Some(transition) = request.transition {
        return render_transition(
            request,
            transition,
            scope,
            runtime,
            control,
            &started,
            &mut progress,
        )
        .await;
    }
    yield_to_worker_messages(scope).await;
    if control.is_cancelled(request_id) {
        return Err(String::from("browser render cancelled"));
    }
    let cache_hit = runtime
        .cached_generation
        .as_ref()
        .is_some_and(|cached| cached.matches(&request));
    let grammar = worker_compiled_grammar(&request)?;
    let (ir_disassembly, ir_json) = worker_ir_tooling(&grammar, worker_semantics(&request));
    if !cache_hit {
        // Keep the cache at capacity one even while deriving. Retaining the
        // previous large Generation beside its replacement can otherwise
        // double peak WASM memory and crash before either result is useful.
        runtime.cached_generation = None;
        let calculation = calculate_auto(
            grammar.clone(),
            request.iterations,
            request.seed,
            worker_semantics(&request),
            runtime,
            &mut progress,
            scope,
            &control,
            false,
        )
        .await?;
        runtime.cached_generation = Some(CachedGeneration {
            key: WorkerGenerationKey {
                source: request.source.clone(),
                ir_json: request.ir_json.clone(),
                iterations: request.iterations,
                seed: request.seed,
                semantics: worker_semantics(&request),
            },
            generation: calculation.generation,
            stats: calculation.stats,
            backend: calculation.backend,
            lineage: calculation.lineage,
        });
    }
    let calculation = runtime
        .cached_generation
        .as_ref()
        .expect("a completed calculation must populate the exact worker cache");
    yield_to_worker_messages(scope).await;
    if control.is_cancelled(request_id) {
        return Err(String::from("browser render cancelled"));
    }
    if cache_hit
        && !progress.calculation(CalculationProgress {
            phase: CalculationPhase::Complete,
            phase_completed: request.iterations,
            phase_total: Some(request.iterations),
            completed_iterations: request.iterations,
            total_iterations: request.iterations,
            modules: calculation.stats.modules,
            items: calculation.stats.items,
            elapsed: started.elapsed(),
        })
    {
        return Err(String::from("cached browser calculation cancelled"));
    }

    let visualizer = VisualizerKind::from_str(&request.visualizer)
        .map_err(|error| format!("invalid visualizer: {error}"))?;
    let orientation_anchor = request
        .orientation_anchor
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(|error: braken_gui::orientation::ParseOrientationAnchorError| error.to_string())?;
    let orientation_reference = request.turtle.initial_angle;
    let config = if matches!(
        visualizer,
        VisualizerKind::Turtle2d | VisualizerKind::Turtle3d
    ) {
        let turtle = Turtle2dConfig {
            turn_angle: (request.angle as f64).to_radians(),
            initial_angle: request.turtle.initial_angle,
            default_step: request.turtle.default_step,
            scale_multiplier: request.turtle.scale_multiplier,
            initial_width: request.turtle.initial_width,
            width_increment: request.turtle.width_increment,
            turn_angle_increment: request.turtle.turn_angle_increment,
            initial_color: request.turtle.initial_color.into(),
            color_increment: request.turtle.color_increment,
            palette: request.turtle.palette,
            background: request.turtle.background,
            draw_modules: request.turtle.draw_modules,
            move_modules: request.turtle.move_modules,
            module_aliases: request.turtle.module_aliases,
        };
        if visualizer == VisualizerKind::Turtle2d {
            VisualizerConfig::Turtle2d(turtle)
        } else {
            VisualizerConfig::Turtle3d(braken_viz::Turtle3dConfig::from(turtle))
        }
    } else {
        VisualizerConfig::for_kind(visualizer)
    };
    let backend = if cache_hit {
        match calculation.backend {
            "WGPU" => "WGPU (cached)",
            "CPU" => "CPU (cached)",
            backend => backend,
        }
    } else {
        calculation.backend
    };
    let context = VisualizationContext {
        iterations: request.iterations,
        seed: request.seed,
        derivation_backend: Some(backend),
        elapsed: Some(calculation.stats.elapsed),
    };
    if visualizer == VisualizerKind::Turtle3d {
        let VisualizerConfig::Turtle3d(turtle) = config else {
            return Err(String::from(
                "3D turtle visualizer received the wrong configuration",
            ));
        };
        let mut lines = SpatialLineChunkBuilder::new()?;
        let mut polygons = Vec::new();
        let mut primitive_order = SpatialPrimitiveOrderCursor::default();
        let summary = stream_turtle_3d(
            Turtle3dStreamRequest {
                generation: &calculation.generation,
                config: turtle,
                batch_size: 16 * 1024,
                is_cancelled: &|| control.is_cancelled(request_id),
            },
            |batch| {
                SpatialPrimitiveOrderCursor::validate_batch(
                    &batch.primitive_order,
                    batch.lines.len(),
                    batch.polygons.len(),
                )
                .map_err(|detail| VisualizeError::InvalidConfiguration(detail.to_owned()))?;
                let requested_polygons = polygons
                    .len()
                    .checked_add(batch.polygons.len())
                    .ok_or_else(|| {
                        spatial_resource_error("browser worker 3D polygons", usize::MAX)
                    })?;
                polygons.try_reserve(batch.polygons.len()).map_err(|_| {
                    spatial_resource_error("browser worker 3D polygons", requested_polygons)
                })?;
                let batch_progress = batch.progress;
                let mut batch_polygons = batch.polygons.into_iter();
                for kind in batch.primitive_order {
                    let Some(lines_before) = primitive_order.advance(kind).map_err(|detail| {
                        VisualizeError::InvalidConfiguration(detail.to_owned())
                    })?
                    else {
                        continue;
                    };
                    let polygon = batch_polygons.next().ok_or_else(|| {
                        VisualizeError::InvalidConfiguration(String::from(
                            "turtle_3d polygon order exhausted its payload",
                        ))
                    })?;
                    polygons.push(encode_worker_polygon_3d(polygon, lines_before)?);
                }
                debug_assert!(batch_polygons.next().is_none());
                for line in batch.lines {
                    lines.push(line)?;
                }
                if lines.line_count != primitive_order.lines_before {
                    return Err(VisualizeError::InvalidConfiguration(String::from(
                        "turtle_3d cumulative line order did not match its payload",
                    )));
                }
                if !progress.visualizing(
                    batch_progress.items_processed,
                    calculation.stats.items,
                    calculation.stats.iterations,
                    batch_progress.modules_processed,
                    batch_progress.items_processed,
                    started.elapsed(),
                ) {
                    return Err(VisualizeError::Cancelled);
                }
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;
        let bounds = summary
            .bounds
            .map(encode_worker_bounds_3d)
            .transpose()
            .map_err(|error| error.to_string())?;
        let (line_chunks, line_count, total_line_length) = lines.finish();
        if summary.progress.lines_emitted != line_count
            || summary.progress.polygons_emitted != polygons.len()
            || summary.progress.items_processed != calculation.stats.items
        {
            return Err(String::from(
                "turtle_3d stream summary did not match the transferred scene",
            ));
        }
        return Ok((
            worker_protocol::WorkerRenderResult {
                line_count,
                total_line_length,
                width_reference: Some(summary.width_reference),
                scene: worker_protocol::WorkerScene::ThreeD { polygons, bounds },
                elapsed_millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                derivation_backend: backend.to_owned(),
                visualization_backend: String::from("CPU"),
                ir_disassembly,
                ir_json,
                background: request.turtle.background,
                transition: None,
            },
            line_chunks,
            None,
        ));
    }
    let mut lines = LineChunkBuilder::new()?;
    let mut polygons = Vec::new();
    let mut texts = Vec::new();
    if visualizer == VisualizerKind::Turtle2d {
        let VisualizerConfig::Turtle2d(turtle) = config else {
            return Err(String::from(
                "turtle visualizer received the wrong configuration",
            ));
        };
        stream_turtle_2d(
            Turtle2dStreamRequest {
                generation: &calculation.generation,
                config: turtle,
                batch_size: 16 * 1024,
                is_cancelled: &|| control.is_cancelled(request_id),
            },
            |batch| {
                polygons.try_reserve(batch.polygons.len()).map_err(|_| {
                    VisualizeError::ResourceExhausted {
                        resource: "browser worker polygons",
                        requested: Some(polygons.len().saturating_add(batch.polygons.len())),
                    }
                })?;
                polygons.extend(batch.polygons.into_iter().map(|polygon| {
                    worker_protocol::WorkerPolygon {
                        vertices: polygon.vertices.into_iter().map(|(x, y)| [x, y]).collect(),
                        color: polygon.color.into(),
                    }
                }));
                for line in batch.lines {
                    lines
                        .push(line)
                        .map_err(|_| VisualizeError::ResourceExhausted {
                            resource: "browser worker line chunks",
                            requested: batch
                                .progress
                                .lines_emitted
                                .checked_mul(worker_protocol::LINE_TRANSFER_VALUES),
                        })?;
                }
                if !progress.visualizing(
                    batch.progress.items_processed,
                    calculation.stats.items,
                    calculation.stats.iterations,
                    batch.progress.modules_processed,
                    batch.progress.items_processed,
                    started.elapsed(),
                ) {
                    return Err(VisualizeError::Cancelled);
                }
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;
    } else {
        let first = visualize(VisualizeRequest {
            generation: &calculation.generation,
            backend: VisualizerBackend::Auto,
            config: config.clone(),
            context,
        });
        let visualization = match first {
            Err(VisualizeError::Unimplemented { .. }) => visualize(VisualizeRequest {
                generation: &calculation.generation,
                backend: VisualizerBackend::Cpu,
                config: VisualizerConfig::for_kind(VisualizerKind::Inspector),
                context,
            }),
            result => result,
        }
        .map_err(|error| error.to_string())?;
        let Visualization::Scene2d(scene) = visualization else {
            return Err(String::from(
                "non-3D visualizer unexpectedly returned a three-dimensional scene",
            ));
        };
        for primitive in scene.primitives {
            match primitive {
                Primitive2d::Line(line) => lines.push(line)?,
                Primitive2d::Polygon(polygon) => {
                    polygons.try_reserve(1).map_err(|_| {
                        String::from("not enough worker memory for a visualization polygon")
                    })?;
                    polygons.push(worker_protocol::WorkerPolygon {
                        vertices: polygon.vertices.into_iter().map(|(x, y)| [x, y]).collect(),
                        color: polygon.color.into(),
                    });
                }
                Primitive2d::Text(text) => {
                    texts.try_reserve(1).map_err(|_| {
                        String::from("not enough worker memory for visualization text")
                    })?;
                    texts.push(WorkerText {
                        x: text.position.0,
                        y: text.position.1,
                        content: text.content,
                        size: text.size,
                        role: match text.role {
                            TextRole::Title => WorkerTextRole::Title,
                            TextRole::Heading => WorkerTextRole::Heading,
                            TextRole::Body => WorkerTextRole::Body,
                            TextRole::Muted => WorkerTextRole::Muted,
                            TextRole::Error => WorkerTextRole::Error,
                        },
                    });
                }
            }
        }
    }
    let orientation_anchor = (visualizer == VisualizerKind::Turtle2d && polygons.is_empty())
        .then_some(orientation_anchor)
        .flatten();
    let (line_chunks, line_count, bounds, total_line_length) =
        lines.finish(orientation_anchor, orientation_reference);

    Ok((
        worker_protocol::WorkerRenderResult {
            line_count,
            total_line_length,
            width_reference: None,
            scene: worker_protocol::WorkerScene::TwoD {
                polygons,
                texts,
                bounds,
            },
            elapsed_millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            derivation_backend: backend.to_owned(),
            // CUDA is unavailable in browsers and there is no WGPU compute
            // visualizer yet; WGPU below is the independent display stage.
            visualization_backend: String::from("CPU"),
            ir_disassembly,
            ir_json,
            background: request.turtle.background,
            transition: None,
        },
        line_chunks,
        None,
    ))
}

#[cfg(target_arch = "wasm32")]
fn worker_compiled_grammar(
    request: &worker_protocol::WorkerRenderRequest,
) -> Result<braken::CompiledGrammar, String> {
    if let Some(json) = request.ir_json.as_deref() {
        braken_gui::ir_tooling::compile_json(json).map_err(|error| error.to_string())
    } else {
        braken::CompiledGrammar::parse(&request.source).map_err(|error| error.to_string())
    }
}

#[cfg(target_arch = "wasm32")]
fn worker_ir_tooling(
    grammar: &braken::CompiledGrammar,
    semantics: braken::DerivationSemantics,
) -> (String, Option<String>) {
    let snapshot = braken_gui::ir_tooling::snapshot_with_semantics(grammar, semantics);
    (snapshot.disassembly, snapshot.json)
}

#[cfg(test)]
mod tests {
    use super::SpatialPrimitiveOrderCursor;
    use braken_viz::Turtle3dPrimitiveKind::{Line, Polygon};

    #[test]
    fn spatial_primitive_cursor_keeps_consecutive_and_cross_batch_offsets() {
        let mut cursor = SpatialPrimitiveOrderCursor::default();
        SpatialPrimitiveOrderCursor::validate_batch(&[Line, Polygon], 1, 1).unwrap();
        let first = [Line, Polygon]
            .into_iter()
            .filter_map(|kind| cursor.advance(kind).unwrap())
            .collect::<Vec<_>>();
        SpatialPrimitiveOrderCursor::validate_batch(&[Polygon, Line, Polygon, Polygon], 1, 3)
            .unwrap();
        let second = [Polygon, Line, Polygon, Polygon]
            .into_iter()
            .filter_map(|kind| cursor.advance(kind).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(first, [1]);
        assert_eq!(second, [1, 2, 2]);
        assert_eq!(cursor.lines_before, 2);
    }

    #[test]
    fn spatial_primitive_cursor_rejects_malformed_batch_tags() {
        assert!(SpatialPrimitiveOrderCursor::validate_batch(&[Line], 0, 1).is_err());
        assert!(SpatialPrimitiveOrderCursor::validate_batch(&[Polygon], 1, 0).is_err());
    }
}
