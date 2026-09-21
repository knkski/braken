//! Panel and mobile-draft regressions, independent of the display backend.

use super::*;

fn app() -> BrakenGui {
    let (app, _) = BrakenGui::new();
    app.render_coordinator.cancel_current();
    app
}

fn replace_draft(app: &mut BrakenGui, source: &str) {
    let _ = app.update(Message::DraftEdited(text_editor::Action::SelectAll));
    let _ = app.update(Message::DraftEdited(text_editor::Action::Edit(
        text_editor::Edit::Paste(Arc::new(source.to_owned())),
    )));
}

#[test]
fn panels_start_closed_and_toggling_is_display_only() {
    let mut app = app();
    let request = app.request_id;
    let source = app.source_editor.text();
    let preset = app.selected_preset;
    let camera = app.live_camera;
    assert!(!app.editor_open && !app.catalog_open && !app.panel_expanded);

    for message in [
        Message::ToggleSettings,
        Message::ToggleEditor(LayoutMode::Desktop),
        Message::CloseOverlay,
        Message::ToggleSettings,
        Message::ToggleDiagnostics,
    ] {
        let _ = app.update(message);
    }

    assert!(!app.editor_open && !app.panel_expanded);
    assert_eq!(app.request_id, request);
    assert_eq!(app.source_editor.text(), source);
    assert_eq!(app.selected_preset, preset);
    assert_eq!(app.live_camera, camera);
}

#[test]
fn mobile_draft_survives_back_and_only_apply_queues_work() {
    let mut app = app();
    let request = app.request_id;
    let view_epoch = app.system_view_epoch;
    let source = app.source_editor.text();
    let preset = app.selected_preset;
    let _ = app.update(Message::ToggleEditor(LayoutMode::Mobile));
    assert!(app.editor_draft_mode);
    assert!(!app.canvas_is_visible());
    replace_draft(&mut app, "axiom Draw Draw;");
    let _ = app.update(Message::CloseEditor);
    assert!(app.canvas_is_visible());
    assert_eq!(app.source_editor.text(), source);
    assert_eq!(app.request_id, request);
    assert_eq!(app.selected_preset, preset);

    let _ = app.update(Message::ToggleEditor(LayoutMode::Mobile));
    assert_eq!(app.editor_draft.text(), "axiom Draw Draw;");
    let _ = app.update(Message::ApplyDraft);
    app.render_coordinator.cancel_current();
    assert!(!app.editor_open && !app.draft_is_modified());
    assert_eq!(app.source_editor.text(), "axiom Draw Draw;");
    // Cancelling the superseded request also advances the request ID. A draft
    // application replaces the source exactly once, regardless of that detail.
    assert_ne!(app.request_id, request);
    assert_eq!(app.system_view_epoch, view_epoch.wrapping_add(1));
    assert_eq!(app.selected_preset, None);
    let applied_request = app.request_id;
    let _ = app.update(Message::ApplyDraft);
    assert_eq!(app.request_id, applied_request);
}

#[test]
fn preset_reselection_invalidates_a_saved_draft_even_for_the_same_source() {
    let mut app = app();
    let preset = app.selected_preset.expect("initial preset");
    let _ = app.update(Message::ToggleEditor(LayoutMode::Mobile));
    replace_draft(&mut app, "axiom Draw Draw;");
    let _ = app.update(Message::CloseEditor);
    let _ = app.update(Message::PresetSelected(preset));
    app.render_coordinator.cancel_current();
    let request = app.request_id;
    assert!(!app.draft_is_modified());
    assert_eq!(app.editor_draft.text(), app.source_editor.text());
    let _ = app.update(Message::ApplyDraft);
    assert_eq!(app.request_id, request);
}

#[test]
fn a_stale_draft_cannot_overwrite_replaced_source() {
    let mut app = app();
    let _ = app.update(Message::ToggleEditor(LayoutMode::Mobile));
    replace_draft(&mut app, "axiom Draw Draw;");
    app.source_editor = text_editor::Content::with_text("axiom Turn(90);");
    let request = app.request_id;
    let _ = app.update(Message::ApplyDraft);
    assert_eq!(app.source_editor.text(), "axiom Turn(90);");
    assert_eq!(app.request_id, request);
    assert!(!app.draft_is_modified());
}

#[test]
fn desktop_to_mobile_resize_uses_a_draft_and_keeps_it_on_return() {
    let mut app = app();
    let source = app.source_editor.text();
    let request = app.request_id;
    let _ = app.update(Message::ToggleEditor(LayoutMode::Desktop));
    assert!(!app.editor_draft_mode);
    let _ = app.update(Message::WindowResized(Size::new(390.0, 844.0)));
    assert!(app.editor_draft_mode && !app.canvas_is_visible());
    // An action queued by the old live editor must also respect the new mode.
    let _ = app.update(Message::SourceEdited(text_editor::Action::SelectAll));
    let _ = app.update(Message::SourceEdited(text_editor::Action::Edit(
        text_editor::Edit::Paste(Arc::new(String::from("axiom Draw;"))),
    )));
    assert_eq!(app.source_editor.text(), source);
    assert_eq!(app.request_id, request);
    let _ = app.update(Message::WindowResized(Size::new(1280.0, 820.0)));
    assert!(app.editor_draft_mode && app.canvas_is_visible());
    let _ = app.update(Message::ToggleEditor(LayoutMode::Desktop));
    let _ = app.update(Message::ToggleEditor(LayoutMode::Desktop));
    assert!(app.editor_draft_mode);
    assert_eq!(app.editor_draft.text(), "axiom Draw;");
}

#[test]
fn catalog_and_editor_are_exclusive_without_discarding_a_draft() {
    let mut app = app();
    let request = app.request_id;
    let _ = app.update(Message::ToggleEditor(LayoutMode::Mobile));
    replace_draft(&mut app, "axiom Draw;");
    let _ = app.update(Message::OpenCatalog);
    assert!(app.catalog_open && !app.editor_open && !app.canvas_is_visible());
    let _ = app.update(Message::CloseCatalog);
    assert!(app.canvas_is_visible());
    let _ = app.update(Message::ToggleEditor(LayoutMode::Mobile));
    assert!(!app.catalog_open && app.editor_open);
    assert_eq!(app.editor_draft.text(), "axiom Draw;");
    assert_eq!(app.request_id, request);
}
