//! Persistent browser-worker transport with latest-request-wins coordination.
//!
//! The render worker owns derivation and visualization state. This client keeps
//! it alive between requests, asks it to cancel superseded work exactly once,
//! and recreates it only after a transport failure or an unacknowledged cancel.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{ErrorEvent, MessageEvent, Worker};
use yew::Callback;

use crate::worker_protocol::{WorkerCommand, WorkerEvent, WorkerRenderRequest, WorkerRenderResult};

const CANCEL_WATCHDOG_MILLIS: i32 = 1_000;

#[derive(Debug, Clone)]
pub(crate) struct ClientRequest {
    pub(crate) wire: WorkerRenderRequest,
    pub(crate) view_epoch: u64,
    pub(crate) iteration: usize,
}

impl ClientRequest {
    fn request_id(&self) -> u64 {
        self.wire.request_id
    }
}

// Keeping the completed request and metadata inline makes the Yew message API
// direct; completions are infrequent compared with lightweight progress events.
#[allow(clippy::large_enum_variant)]
pub(crate) enum WorkerUpdate {
    Ready,
    Progress {
        request_id: u64,
        phase: String,
        phase_completed: usize,
        phase_total: Option<usize>,
        completed_iterations: usize,
        total_iterations: usize,
        modules: usize,
        items: usize,
        elapsed_millis: u64,
    },
    Finished {
        request: ClientRequest,
        result: Result<WorkerRenderResult, String>,
        lines: Option<js_sys::Array>,
        morphs: Option<js_sys::Array>,
    },
    Cancelled {
        request_id: u64,
    },
    Fatal(String),
}

/// A persistent Web Worker client.
///
/// At most one request is running and one replacement is waiting. Submitting a
/// third request cancels the older waiting request locally. Dropping the client
/// sends `Shutdown` on a best-effort basis and always terminates the Worker.
pub(crate) struct WorkerClient {
    state: Rc<RefCell<ClientState>>,
}

struct ClientState {
    callback: Callback<WorkerUpdate>,
    run: Option<WorkerRun>,
    active: Option<ActiveRequest>,
    pending: Option<ClientRequest>,
    cancelling: bool,
    next_worker_id: u64,
    stopped: bool,
}

struct ActiveRequest {
    request: ClientRequest,
    sent: bool,
}

struct WorkerRun {
    worker_id: u64,
    ready: bool,
    worker: Worker,
    _onmessage: Closure<dyn FnMut(MessageEvent)>,
    _onerror: Closure<dyn FnMut(ErrorEvent)>,
}

impl WorkerClient {
    pub(crate) fn new(callback: Callback<WorkerUpdate>) -> Self {
        Self {
            state: Rc::new(RefCell::new(ClientState {
                callback,
                run: None,
                active: None,
                pending: None,
                cancelling: false,
                next_worker_id: 0,
                stopped: false,
            })),
        }
    }

    pub(crate) fn submit(&self, request: ClientRequest) {
        let request_id = request.request_id();
        let (callback, replaced, should_start, should_cancel) = {
            let mut state = self.state.borrow_mut();
            let callback = state.callback.clone();
            if state.stopped {
                (callback, Some(request_id), false, None)
            } else if state.active.is_none() {
                state.active = Some(ActiveRequest {
                    request,
                    sent: false,
                });
                (callback, None, true, None)
            } else {
                let replaced = state
                    .pending
                    .replace(request)
                    .map(|request| request.request_id());
                let should_cancel = if state.cancelling {
                    None
                } else {
                    state.cancelling = true;
                    state
                        .active
                        .as_ref()
                        .map(|active| active.request.request_id())
                };
                (callback, replaced, false, should_cancel)
            }
        };

        if let Some(request_id) = replaced {
            callback.emit(WorkerUpdate::Cancelled { request_id });
        }
        if should_start {
            ensure_worker_and_dispatch(&self.state);
        }
        if let Some(request_id) = should_cancel {
            cancel_active(&self.state, request_id);
        }
    }

    pub(crate) fn cancel_current(&self) {
        let (callback, pending, active) = {
            let mut state = self.state.borrow_mut();
            let callback = state.callback.clone();
            let pending = state.pending.take().map(|request| request.request_id());
            let active = if state.cancelling {
                None
            } else {
                let request_id = state
                    .active
                    .as_ref()
                    .map(|active| active.request.request_id());
                state.cancelling = request_id.is_some();
                request_id
            };
            (callback, pending, active)
        };

        if let Some(request_id) = pending {
            callback.emit(WorkerUpdate::Cancelled { request_id });
        }
        if let Some(request_id) = active {
            cancel_active(&self.state, request_id);
        }
    }

    fn shutdown(&self) {
        let run = {
            let mut state = self.state.borrow_mut();
            if state.stopped {
                return;
            }
            state.stopped = true;
            state.active = None;
            state.pending = None;
            state.cancelling = false;
            state.run.take()
        };

        if let Some(run) = run {
            if let Ok(json) = serde_json::to_string(&WorkerCommand::Shutdown) {
                let _result = run.worker.post_message(&JsValue::from_str(&json));
            }
            terminate_worker(run);
        }
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn ensure_worker_and_dispatch(state: &Rc<RefCell<ClientState>>) {
    let spawn_id = {
        let mut state = state.borrow_mut();
        if state.stopped || state.active.is_none() {
            return;
        }
        if state.run.is_none() {
            state.next_worker_id = state.next_worker_id.wrapping_add(1);
            Some(state.next_worker_id)
        } else {
            None
        }
    };

    if let Some(worker_id) = spawn_id {
        match spawn_coordinated_worker(state, worker_id) {
            Ok(run) => state.borrow_mut().run = Some(run),
            Err(message) => {
                fail_active_without_worker(state, message);
                return;
            }
        }
    }

    if let Err((worker_id, request_id, message)) = dispatch_active(state) {
        recover_worker(state, worker_id, Some(request_id), message);
    }
}

fn spawn_coordinated_worker(
    state: &Rc<RefCell<ClientState>>,
    worker_id: u64,
) -> Result<WorkerRun, String> {
    let worker = spawn_web_worker()?;

    let message_state = Rc::downgrade(state);
    let onmessage = Closure::wrap(Box::new(move |message: MessageEvent| {
        let Some(state) = message_state.upgrade() else {
            return;
        };
        if !worker_is_current(&state, worker_id) {
            return;
        }
        handle_worker_message(&state, worker_id, message);
    }) as Box<dyn FnMut(MessageEvent)>);

    let error_state = Rc::downgrade(state);
    let onerror = Closure::wrap(Box::new(move |error: ErrorEvent| {
        let Some(state) = error_state.upgrade() else {
            return;
        };
        if !worker_is_current(&state, worker_id) {
            return;
        }
        let message = if error.message().is_empty() {
            String::from("browser render worker failed to load")
        } else {
            format!("browser render worker failed: {}", error.message())
        };
        recover_worker(&state, worker_id, None, message);
    }) as Box<dyn FnMut(ErrorEvent)>);

    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
    Ok(WorkerRun {
        worker_id,
        ready: false,
        worker,
        _onmessage: onmessage,
        _onerror: onerror,
    })
}

fn handle_worker_message(state: &Rc<RefCell<ClientState>>, worker_id: u64, message: MessageEvent) {
    let data = message.data();
    let (json, lines, morphs) = if let Some(json) = data.as_string() {
        (json, None, None)
    } else {
        let metadata = js_sys::Reflect::get(&data, &JsValue::from_str("metadata"))
            .ok()
            .and_then(|value| value.as_string());
        let lines = js_sys::Reflect::get(&data, &JsValue::from_str("lines"))
            .ok()
            .and_then(|value| value.dyn_into::<js_sys::Array>().ok());
        let morphs = js_sys::Reflect::get(&data, &JsValue::from_str("morphs"))
            .ok()
            .and_then(|value| value.dyn_into::<js_sys::Array>().ok());
        let Some(metadata) = metadata else {
            recover_worker(
                state,
                worker_id,
                None,
                String::from("browser render worker returned an invalid payload"),
            );
            return;
        };
        (metadata, lines, morphs)
    };

    let event = match serde_json::from_str::<WorkerEvent>(&json) {
        Ok(event) => event,
        Err(error) => {
            recover_worker(
                state,
                worker_id,
                None,
                format!("invalid browser render response: {error}"),
            );
            return;
        }
    };

    match event {
        WorkerEvent::Ready => worker_ready(state, worker_id),
        WorkerEvent::Fatal { message } => recover_worker(state, worker_id, None, message),
        WorkerEvent::Progress {
            request_id,
            phase,
            phase_completed,
            phase_total,
            completed_iterations,
            total_iterations,
            modules,
            items,
            elapsed_millis,
        } => {
            let callback = {
                let state = state.borrow();
                state
                    .active
                    .as_ref()
                    .filter(|active| active.request.request_id() == request_id)
                    .map(|_| state.callback.clone())
            };
            if let Some(callback) = callback {
                callback.emit(WorkerUpdate::Progress {
                    request_id,
                    phase,
                    phase_completed,
                    phase_total,
                    completed_iterations,
                    total_iterations,
                    modules,
                    items,
                    elapsed_millis,
                });
            }
        }
        WorkerEvent::Cancelled { request_id } => {
            complete_cancelled(state, worker_id, request_id);
        }
        WorkerEvent::Finished { request_id, result } => {
            complete_finished(state, worker_id, request_id, result, lines, morphs);
        }
    }
}

fn worker_ready(state: &Rc<RefCell<ClientState>>, worker_id: u64) {
    let callback = {
        let mut state = state.borrow_mut();
        let Some(run) = state.run.as_mut().filter(|run| run.worker_id == worker_id) else {
            return;
        };
        run.ready = true;
        state.callback.clone()
    };
    callback.emit(WorkerUpdate::Ready);

    if let Err((worker_id, request_id, message)) = dispatch_active(state) {
        recover_worker(state, worker_id, Some(request_id), message);
    }
}

fn dispatch_active(state: &Rc<RefCell<ClientState>>) -> Result<(), (u64, u64, String)> {
    let (worker, worker_id, request_id, json) = {
        let mut state = state.borrow_mut();
        let Some(run) = state.run.as_ref().filter(|run| run.ready) else {
            return Ok(());
        };
        let worker = run.worker.clone();
        let worker_id = run.worker_id;
        let Some(active) = state.active.as_mut() else {
            return Ok(());
        };
        if active.sent {
            return Ok(());
        }
        let request_id = active.request.request_id();
        let json = serde_json::to_string(&WorkerCommand::Run(active.request.wire.clone()))
            .map_err(|error| {
                (
                    worker_id,
                    request_id,
                    format!("could not serialize browser render request: {error}"),
                )
            })?;
        active.sent = true;
        (worker, worker_id, request_id, json)
    };

    worker
        .post_message(&JsValue::from_str(&json))
        .map_err(|error| {
            (
                worker_id,
                request_id,
                format!("could not send work to browser render worker: {error:?}"),
            )
        })
}

fn cancel_active(state: &Rc<RefCell<ClientState>>, request_id: u64) {
    let dispatch = {
        let state = state.borrow();
        let Some(active) = state
            .active
            .as_ref()
            .filter(|active| active.request.request_id() == request_id)
        else {
            return;
        };
        if !active.sent {
            None
        } else {
            state
                .run
                .as_ref()
                .map(|run| (run.worker.clone(), run.worker_id))
        }
    };

    let Some((worker, worker_id)) = dispatch else {
        complete_cancelled_locally(state, request_id);
        return;
    };
    let sent = serde_json::to_string(&WorkerCommand::Cancel { request_id })
        .map_err(|error| error.to_string())
        .and_then(|json| {
            worker
                .post_message(&JsValue::from_str(&json))
                .map_err(|error| format!("could not cancel browser render work: {error:?}"))
        });
    if let Err(message) = sent {
        recover_worker(state, worker_id, Some(request_id), message);
    } else {
        arm_cancel_watchdog(state, worker_id, request_id);
    }
}

fn complete_finished(
    state: &Rc<RefCell<ClientState>>,
    worker_id: u64,
    request_id: u64,
    result: Result<WorkerRenderResult, String>,
    lines: Option<js_sys::Array>,
    morphs: Option<js_sys::Array>,
) {
    if !worker_is_current(state, worker_id) {
        return;
    }
    let completion = take_active_and_promote(state, request_id);
    let Some((callback, request, has_next)) = completion else {
        return;
    };
    callback.emit(WorkerUpdate::Finished {
        request,
        result,
        lines,
        morphs,
    });
    if has_next {
        ensure_worker_and_dispatch(state);
    }
}

fn complete_cancelled(state: &Rc<RefCell<ClientState>>, worker_id: u64, request_id: u64) {
    if !worker_is_current(state, worker_id) {
        return;
    }
    complete_cancelled_locally(state, request_id);
}

fn complete_cancelled_locally(state: &Rc<RefCell<ClientState>>, request_id: u64) {
    let completion = take_active_and_promote(state, request_id);
    let Some((callback, _request, has_next)) = completion else {
        return;
    };
    callback.emit(WorkerUpdate::Cancelled { request_id });
    if has_next {
        ensure_worker_and_dispatch(state);
    }
}

fn take_active_and_promote(
    state: &Rc<RefCell<ClientState>>,
    request_id: u64,
) -> Option<(Callback<WorkerUpdate>, ClientRequest, bool)> {
    let mut state = state.borrow_mut();
    if state
        .active
        .as_ref()
        .is_none_or(|active| active.request.request_id() != request_id)
    {
        return None;
    }
    let active = state.active.take().expect("the active request was checked");
    state.cancelling = false;
    if !state.stopped
        && let Some(request) = state.pending.take()
    {
        state.active = Some(ActiveRequest {
            request,
            sent: false,
        });
    }
    let has_next = state.active.is_some();
    Some((state.callback.clone(), active.request, has_next))
}

fn fail_active_without_worker(state: &Rc<RefCell<ClientState>>, message: String) {
    let request_id = state
        .borrow()
        .active
        .as_ref()
        .map(|active| active.request.request_id());
    let Some(request_id) = request_id else {
        emit_fatal(state, message);
        return;
    };
    let Some((callback, request, has_next)) = take_active_and_promote(state, request_id) else {
        return;
    };
    callback.emit(WorkerUpdate::Finished {
        request,
        result: Err(message),
        lines: None,
        morphs: None,
    });
    if has_next {
        ensure_worker_and_dispatch(state);
    }
}

fn recover_worker(
    state: &Rc<RefCell<ClientState>>,
    worker_id: u64,
    expected_request_id: Option<u64>,
    message: String,
) {
    let (run, active, callback, was_cancelling, has_next) = {
        let mut state = state.borrow_mut();
        if state
            .run
            .as_ref()
            .is_none_or(|run| run.worker_id != worker_id)
            || expected_request_id.is_some_and(|expected| {
                state
                    .active
                    .as_ref()
                    .map(|active| active.request.request_id())
                    != Some(expected)
            })
        {
            return;
        }
        let run = state.run.take();
        let active = state.active.take();
        let callback = state.callback.clone();
        let was_cancelling = state.cancelling;
        state.cancelling = false;
        if !state.stopped
            && let Some(request) = state.pending.take()
        {
            state.active = Some(ActiveRequest {
                request,
                sent: false,
            });
        }
        let has_next = state.active.is_some();
        (run, active, callback, was_cancelling, has_next)
    };

    if let Some(run) = run {
        terminate_worker(run);
    }
    match active {
        Some(active) if was_cancelling => callback.emit(WorkerUpdate::Cancelled {
            request_id: active.request.request_id(),
        }),
        Some(active) => callback.emit(WorkerUpdate::Finished {
            request: active.request,
            result: Err(message),
            lines: None,
            morphs: None,
        }),
        None => callback.emit(WorkerUpdate::Fatal(message)),
    }
    if has_next {
        ensure_worker_and_dispatch(state);
    }
}

fn arm_cancel_watchdog(state: &Rc<RefCell<ClientState>>, worker_id: u64, request_id: u64) {
    // Cooperative work normally acknowledges cancellation first. The timeout
    // only recovers a panic or a long synchronous call that prevents the
    // Worker event loop from observing the cancellation message.
    let weak = Rc::downgrade(state);
    let callback = Closure::once_into_js(move || {
        let Some(state) = weak.upgrade() else {
            return;
        };
        let still_unresponsive = {
            let state = state.borrow();
            state.cancelling
                && state
                    .active
                    .as_ref()
                    .is_some_and(|active| active.request.request_id() == request_id)
                && state
                    .run
                    .as_ref()
                    .is_some_and(|run| run.worker_id == worker_id)
        };
        if still_unresponsive {
            recover_worker(
                &state,
                worker_id,
                Some(request_id),
                String::from("browser render worker did not acknowledge cancellation"),
            );
        }
    });
    if let Some(window) = web_sys::window() {
        let _result = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.unchecked_ref(),
            CANCEL_WATCHDOG_MILLIS,
        );
    }
}

fn worker_is_current(state: &Rc<RefCell<ClientState>>, worker_id: u64) -> bool {
    state
        .borrow()
        .run
        .as_ref()
        .is_some_and(|run| run.worker_id == worker_id)
}

fn emit_fatal(state: &Rc<RefCell<ClientState>>, message: String) {
    let callback = state.borrow().callback.clone();
    callback.emit(WorkerUpdate::Fatal(message));
}

fn terminate_worker(run: WorkerRun) {
    run.worker.set_onmessage(None);
    run.worker.set_onerror(None);
    run.worker.terminate();
}

fn spawn_web_worker() -> Result<Worker, String> {
    let window = web_sys::window().ok_or_else(|| String::from("browser window is unavailable"))?;
    let base = window
        .location()
        .href()
        .map_err(|error| format!("browser URL is unavailable: {error:?}"))?;
    let script_url = web_sys::Url::new_with_base("braken-render-worker.js", &base)
        .map_err(|error| format!("could not resolve render worker script: {error:?}"))?
        .href();
    let wasm_url = web_sys::Url::new_with_base("braken-render-worker_bg.wasm", &base)
        .map_err(|error| format!("could not resolve render worker WASM: {error:?}"))?
        .href();
    let script_literal = serde_json::to_string(&script_url).map_err(|error| error.to_string())?;
    let wasm_literal = serde_json::to_string(&wasm_url).map_err(|error| error.to_string())?;
    let bootstrap = format!(
        "importScripts({script_literal});wasm_bindgen({wasm_literal}).catch(error=>{{setTimeout(()=>{{throw error;}},0);}});"
    );
    let parts = js_sys::Array::new();
    parts.push(&JsValue::from_str(&bootstrap));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("text/javascript");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options)
        .map_err(|error| format!("could not create render worker bootstrap: {error:?}"))?;
    let object_url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|error| format!("could not create render worker URL: {error:?}"))?;
    let worker = Worker::new(&object_url)
        .map_err(|error| format!("could not start browser render worker: {error:?}"));
    let _result = web_sys::Url::revoke_object_url(&object_url);
    worker
}
