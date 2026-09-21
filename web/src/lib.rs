//! Yew/DOM/Canvas2D browser frontend for Braken.
//!
//! Calculation and visualization remain in the same dedicated browser Worker
//! used by the established GUI. This crate owns only browser coordination,
//! controls, input, and display; it has no dependency on Iced.

mod model;

// These UI-neutral modules stay source-shared with the established frontend so
// camera constraints, preset metadata, and the Worker wire format cannot drift.
// Including them directly also keeps the Yew application independent of Iced.
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
#[path = "../../gui/src/camera.rs"]
mod camera;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
#[path = "../../gui/src/generated_preset_previews.rs"]
mod generated_preset_previews;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
#[path = "../../gui/src/orientation.rs"]
mod orientation;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
#[path = "../../gui/src/presets.rs"]
mod presets;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
#[path = "../../gui/src/theme_palette.rs"]
mod theme_palette;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
#[path = "../../gui/src/worker_protocol.rs"]
mod worker_protocol;

#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(target_arch = "wasm32")]
mod canvas;
#[cfg(target_arch = "wasm32")]
mod worker;

#[cfg(target_arch = "wasm32")]
pub fn run() {
    install_panic_hook();
    let root = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("app"))
        .expect("web/index.html must contain #app");
    yew::Renderer::<app::App>::with_root(root).render();
}

#[cfg(target_arch = "wasm32")]
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        show_wasm_failure("The application panicked", &info.to_string());
        console_error_panic_hook::hook(info);
    }));
}

#[cfg(target_arch = "wasm32")]
fn show_wasm_failure(title: &str, message: &str) {
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    if let Some(heading) = document.get_element_by_id("wasm-status-title") {
        heading.set_text_content(Some(title));
    }
    if let Some(summary) = document.get_element_by_id("wasm-status-summary") {
        summary.set_text_content(Some(
            "The browser build stopped unexpectedly. Reload the page, or copy the details below when reporting the problem.",
        ));
    }
    if let Some(details) = document.get_element_by_id("wasm-status-details") {
        details.set_text_content(Some(message));
        let _ = details.remove_attribute("hidden");
    }
    if let Some(reload) = document.get_element_by_id("wasm-status-reload") {
        let _ = reload.remove_attribute("hidden");
    }
    if let Some(panel) = document.get_element_by_id("wasm-status") {
        let _ = panel.set_attribute("data-state", "error");
        let _ = panel.set_attribute("data-source", "rust");
        let _ = panel.remove_attribute("hidden");
    }
}
