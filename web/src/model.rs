//! Framework-independent UI state helpers.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

/// A mobile editor draft that survives closing the editor for the same source.
///
/// Replacing the active source invalidates the draft. Explicit preset selection
/// should call [`Self::reset`] even when the preset has the same source text.
#[derive(Debug, Clone, Default)]
pub(crate) struct GrammarDraft {
    base: String,
    text: String,
}

impl GrammarDraft {
    pub(crate) fn open(&mut self, source: &str) {
        if self.base != source {
            self.reset(source);
        }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn edit(&mut self, text: String) {
        self.text = text;
    }

    pub(crate) fn is_modified(&self) -> bool {
        self.text != self.base
    }

    /// Returns an edited source once, unless the active source was replaced.
    pub(crate) fn apply(&mut self, current_source: &str) -> Option<String> {
        self.open(current_source);
        if !self.is_modified() {
            return None;
        }
        self.base.clone_from(&self.text);
        Some(self.text.clone())
    }

    pub(crate) fn reset(&mut self, source: &str) {
        source.clone_into(&mut self.base);
        source.clone_into(&mut self.text);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThemePreference {
    Auto,
    Light,
    Dark,
}

impl ThemePreference {
    pub(crate) const fn next(self) -> Self {
        match self {
            Self::Auto => Self::Light,
            Self::Light => Self::Dark,
            Self::Dark => Self::Auto,
        }
    }

    pub(crate) const fn button_label(self) -> &'static str {
        match self {
            Self::Auto => "Switch to light theme",
            Self::Light => "Switch to dark theme",
            Self::Dark => "Follow system theme",
        }
    }

    pub(crate) const fn icon(self) -> &'static str {
        match self {
            Self::Auto => "◐",
            Self::Light => "☀",
            Self::Dark => "☾",
        }
    }

    pub(crate) const fn resolve_dark(self, system_dark: bool) -> bool {
        match self {
            Self::Auto => system_dark,
            Self::Light => false,
            Self::Dark => true,
        }
    }
}

pub(crate) fn formatted_angle(angle: f32) -> String {
    let rounded = angle.round();
    if (angle - rounded).abs() < 0.05 {
        format!("{rounded:.0}")
    } else {
        format!("{angle:.1}")
    }
}

pub(crate) fn adjacent_integer_angle(current: f32, direction: f32) -> f32 {
    let nearest = current.round();
    let normalized = if (current - nearest).abs() < 1.0e-4 {
        nearest
    } else {
        current
    };
    let value = if direction.is_sign_positive() {
        normalized.floor() + 1.0
    } else {
        normalized.ceil() - 1.0
    };
    value.clamp(0.0, 180.0)
}

pub(crate) fn normalized_search_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|term| term.to_lowercase())
        .collect()
}

pub(crate) fn next_iteration(current: usize, target: usize) -> Option<usize> {
    match current.cmp(&target) {
        std::cmp::Ordering::Less => Some(current.saturating_add(1)),
        std::cmp::Ordering::Greater => Some(current.saturating_sub(1)),
        std::cmp::Ordering::Equal => None,
    }
}

/// Touch count needed to acquire navigation in the Yew canvas.
///
/// Mobile 2D views leave the first finger available for page scrolling. A 3D
/// view owns the first touch so the app can stop preset autorotation on a tap
/// and rotate the model on a drag. Desktop views also own the first touch.
pub(crate) const fn required_canvas_touches(is_mobile: bool, is_spatial: bool) -> usize {
    if is_mobile && !is_spatial { 2 } else { 1 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TouchOwnership {
    /// The current browser event belongs to an established or newly acquired
    /// canvas gesture and should suppress native scrolling.
    pub(crate) handle_event: bool,
    /// Ownership remains sticky while at least one finger from the captured
    /// gesture is still down.
    pub(crate) retain_after_event: bool,
}

pub(crate) const fn touch_ownership(
    was_owned: bool,
    navigation_enabled: bool,
    required_touches: usize,
    touch_count: usize,
) -> TouchOwnership {
    let required_touches = if required_touches == 0 {
        1
    } else {
        required_touches
    };
    let handle_event = was_owned || (navigation_enabled && touch_count >= required_touches);
    TouchOwnership {
        handle_event,
        retain_after_event: handle_event && touch_count > 0,
    }
}

pub(crate) fn is_primary_mouse_button(pointer_type: &str, button: i16) -> bool {
    pointer_type == "mouse" && button == 0
}

pub(crate) const fn preset_autorotation_enabled(is_spatial: bool, reduced_motion: bool) -> bool {
    is_spatial && !reduced_motion
}

pub(crate) const fn autorotation_after_motion_preference(
    active: bool,
    pending_epoch: Option<u64>,
    reduced_motion: bool,
) -> (bool, Option<u64>) {
    if reduced_motion {
        (false, None)
    } else {
        (active, pending_epoch)
    }
}

pub(crate) const fn display_refinement_is_running(
    is_refining: bool,
    autorotating: bool,
    interacting: bool,
) -> bool {
    is_refining && !autorotating && !interacting
}

pub(crate) fn source_uses_filled_turtle_polygons(source: &str) -> bool {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .flat_map(|line| {
            line.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        })
        .any(|word| matches!(word, "PolygonBegin" | "PolygonEnd" | "Vertex"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_draft_survives_closing_and_reopening_the_same_source() {
        let mut draft = GrammarDraft::default();
        draft.open("axiom Draw;");
        draft.edit(String::from("axiom Draw Draw;"));

        draft.open("axiom Draw;");

        assert_eq!(draft.text(), "axiom Draw Draw;");
        assert!(draft.is_modified());
    }

    #[test]
    fn opening_a_replaced_source_discards_the_previous_grammar_draft() {
        let mut draft = GrammarDraft::default();
        draft.open("axiom Draw;");
        draft.edit(String::from("axiom Draw Draw;"));

        draft.open("axiom Turn(90);");

        assert_eq!(draft.text(), "axiom Turn(90);");
        assert!(!draft.is_modified());
    }

    #[test]
    fn unchanged_or_reverted_grammar_drafts_do_not_apply() {
        let mut draft = GrammarDraft::default();
        draft.open("axiom Draw;");
        assert_eq!(draft.apply("axiom Draw;"), None);

        draft.edit(String::from("axiom Draw Draw;"));
        draft.edit(String::from("axiom Draw;"));

        assert!(!draft.is_modified());
        assert_eq!(draft.apply("axiom Draw;"), None);
    }

    #[test]
    fn applying_a_grammar_draft_commits_it_once() {
        let mut draft = GrammarDraft::default();
        draft.open("axiom Draw;");
        draft.edit(String::from("axiom Draw Draw;"));

        let applied = draft
            .apply("axiom Draw;")
            .expect("edited draft should apply");

        assert_eq!(applied, "axiom Draw Draw;");
        assert!(!draft.is_modified());
        draft.open(&applied);
        assert_eq!(draft.text(), applied);
        assert_eq!(draft.apply(&applied), None);
    }

    #[test]
    fn a_grammar_draft_cannot_overwrite_a_replaced_active_source() {
        let mut draft = GrammarDraft::default();
        draft.open("axiom Draw;");
        draft.edit(String::from("axiom Draw Draw;"));

        assert_eq!(draft.apply("axiom Turn(90);"), None);
        assert_eq!(draft.text(), "axiom Turn(90);");
        assert!(!draft.is_modified());
    }

    #[test]
    fn explicitly_reselecting_a_source_resets_its_grammar_draft() {
        let mut draft = GrammarDraft::default();
        draft.open("axiom Draw;");
        draft.edit(String::from("axiom Draw Draw;"));

        draft.reset("axiom Draw;");

        assert_eq!(draft.text(), "axiom Draw;");
        assert!(!draft.is_modified());
        assert_eq!(draft.apply("axiom Draw;"), None);
    }

    #[test]
    fn theme_cycles_and_resolves() {
        assert_eq!(ThemePreference::Auto.next(), ThemePreference::Light);
        assert_eq!(ThemePreference::Light.next(), ThemePreference::Dark);
        assert_eq!(ThemePreference::Dark.next(), ThemePreference::Auto);
        assert!(ThemePreference::Auto.resolve_dark(true));
        assert!(!ThemePreference::Light.resolve_dark(true));
        assert!(ThemePreference::Dark.resolve_dark(false));
    }

    #[test]
    fn fractional_angle_buttons_choose_adjacent_integer() {
        assert_eq!(adjacent_integer_angle(42.5, -1.0), 42.0);
        assert_eq!(adjacent_integer_angle(42.5, 1.0), 43.0);
        assert_eq!(adjacent_integer_angle(42.0, -1.0), 41.0);
        assert_eq!(adjacent_integer_angle(42.0, 1.0), 43.0);
        assert_eq!(adjacent_integer_angle(0.0, -1.0), 0.0);
        assert_eq!(adjacent_integer_angle(180.0, 1.0), 180.0);
    }

    #[test]
    fn angle_format_retains_meaningful_fraction() {
        assert_eq!(formatted_angle(42.0), "42");
        assert_eq!(formatted_angle(42.54), "42.5");
        assert_eq!(formatted_angle(42.96), "43");
    }

    #[test]
    fn search_terms_are_case_insensitive_and_whitespace_split() {
        assert_eq!(
            normalized_search_terms("  Gothic  STAR "),
            ["gothic", "star"]
        );
    }

    #[test]
    fn iteration_routes_one_adjacent_step_at_a_time() {
        assert_eq!(next_iteration(0, 3), Some(1));
        assert_eq!(next_iteration(3, 0), Some(2));
        assert_eq!(next_iteration(3, 3), None);
    }

    #[test]
    fn mobile_planar_touch_leaves_one_finger_to_scroll_and_captures_two() {
        let required = required_canvas_touches(true, false);
        assert_eq!(
            touch_ownership(false, true, required, 1),
            TouchOwnership {
                handle_event: false,
                retain_after_event: false,
            }
        );
        assert_eq!(
            touch_ownership(false, true, required, 2),
            TouchOwnership {
                handle_event: true,
                retain_after_event: true,
            }
        );
        assert!(touch_ownership(true, false, required, 1).retain_after_event);
        assert_eq!(
            touch_ownership(true, false, required, 0),
            TouchOwnership {
                handle_event: true,
                retain_after_event: false,
            }
        );
    }

    #[test]
    fn mobile_spatial_touch_claims_taps_and_keeps_drag_ownership_until_release() {
        let required = required_canvas_touches(true, true);
        let pressed = touch_ownership(false, true, required, 1);
        assert!(pressed.handle_event);
        assert!(pressed.retain_after_event);

        let second_finger = touch_ownership(pressed.retain_after_event, true, required, 2);
        assert!(second_finger.handle_event);
        assert!(second_finger.retain_after_event);

        let remaining_finger =
            touch_ownership(second_finger.retain_after_event, false, required, 1);
        assert!(remaining_finger.handle_event);
        assert!(remaining_finger.retain_after_event);

        let released = touch_ownership(remaining_finger.retain_after_event, false, required, 0);
        assert!(released.handle_event);
        assert!(!released.retain_after_event);
        assert!(!touch_ownership(released.retain_after_event, false, required, 1).handle_event);
    }

    #[test]
    fn wide_layout_can_acquire_one_finger_touch_navigation() {
        for is_spatial in [false, true] {
            let required = required_canvas_touches(false, is_spatial);
            assert_eq!(
                touch_ownership(false, true, required, 1),
                TouchOwnership {
                    handle_event: true,
                    retain_after_event: true,
                }
            );
            assert!(!touch_ownership(false, false, required, 1).handle_event);
        }
    }

    #[test]
    fn pointer_drag_requires_the_primary_mouse_button() {
        assert!(is_primary_mouse_button("mouse", 0));
        assert!(!is_primary_mouse_button("mouse", 1));
        assert!(!is_primary_mouse_button("mouse", 2));
        assert!(!is_primary_mouse_button("touch", 0));
        assert!(!is_primary_mouse_button("pen", 0));
    }

    #[test]
    fn reduced_motion_suppresses_only_spatial_preset_autorotation() {
        assert!(preset_autorotation_enabled(true, false));
        assert!(!preset_autorotation_enabled(true, true));
        assert!(!preset_autorotation_enabled(false, false));
        assert!(!preset_autorotation_enabled(false, true));
    }

    #[test]
    fn motion_preference_stops_active_and_pending_rotation_without_auto_resume() {
        assert_eq!(
            autorotation_after_motion_preference(true, Some(17), true),
            (false, None)
        );
        assert_eq!(
            autorotation_after_motion_preference(false, None, false),
            (false, None)
        );
    }

    #[test]
    fn display_refinement_is_running_only_while_the_view_is_stationary() {
        assert!(display_refinement_is_running(true, false, false));
        assert!(!display_refinement_is_running(true, true, false));
        assert!(!display_refinement_is_running(true, false, true));
        assert!(!display_refinement_is_running(true, true, true));
        assert!(!display_refinement_is_running(false, false, false));
    }

    #[test]
    fn polygon_detection_ignores_metadata_comments() {
        assert!(!source_uses_filled_turtle_polygons(
            "# Features: PolygonBegin\naxiom Draw;"
        ));
        assert!(source_uses_filled_turtle_polygons(
            "axiom PolygonBegin Vertex PolygonEnd;"
        ));
    }
}
