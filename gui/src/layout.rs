//! Preset-first layouts shared by the native and Iced browser applications.
//!
//! These views only publish coordination messages. Opening panels, browsing
//! presets, and editing a mobile draft never perform derivation in the view.

use super::*;
use iced::widget::column;

const FEATURED_PRESETS: [&str; 6] = [
    "Braken",
    "Koch Snowflake",
    "3D Hilbert Curve",
    "Orthogonal Virus",
    "Stochastic Plant",
    "Penrose Tiling",
];

impl BrakenGui {
    pub(super) fn responsive_layout(&self, size: Size) -> Element<'_, Message> {
        match layout_mode(size.width) {
            LayoutMode::Desktop => self.desktop_layout(size),
            LayoutMode::Mobile => self.mobile_layout(size),
        }
    }

    fn desktop_layout(&self, size: Size) -> Element<'_, Message> {
        let gallery_width = (size.width * 0.26).clamp(300.0, 340.0);
        let content_width = (size.width - gallery_width - 44.0).max(280.0);
        let content_height = (size.height - CANVAS_PADDING * 2.0).max(240.0);
        let editor_height = if self.editor_open {
            (content_height * 0.28).clamp(190.0, 280.0)
        } else {
            0.0
        };
        let gaps = 24.0
            + if self.editor_open { 12.0 } else { 0.0 }
            + if self.panel_expanded { 12.0 } else { 0.0 };
        let minimum_height = |width: f32| {
            let bar_height = if width >= 830.0 {
                90.0
            } else if width >= 616.0 {
                154.0
            } else {
                222.0
            };
            let advanced_height = if self.panel_expanded {
                if width >= 740.0 { 98.0 } else { 168.0 }
            } else {
                0.0
            };
            44.0 + bar_height + editor_height + advanced_height + gaps + 140.0
        };
        let needs_scrollbar = minimum_height(content_width) > content_height;
        // Reserve Iced's 10px scrollbar plus its gap only when the workspace
        // actually scrolls; narrow controls must wrap within that reduced width.
        let content_width = content_width - if needs_scrollbar { 12.0 } else { 0.0 };
        let workspace_height = content_height.max(minimum_height(content_width));
        let mut workspace = column![
            self.workspace_header(LayoutMode::Desktop),
            container_widget(self.canvas(LayoutMode::Desktop))
                .height(Length::Fill)
                .width(Length::Fill),
        ]
        .spacing(12)
        .height(workspace_height)
        .width(Length::Fill);
        if self.editor_open {
            workspace = workspace.push(self.grammar_editor(LayoutMode::Desktop, editor_height));
        }
        if self.panel_expanded {
            workspace = workspace.push(self.advanced_controls(LayoutMode::Desktop, content_width));
        }
        workspace = workspace.push(self.desktop_control_bar(content_width));
        let mut workspace = scrollable(workspace)
            .width(Length::Fill)
            .height(Length::Fill);
        if needs_scrollbar {
            workspace = workspace.spacing(2);
        }

        row![
            container_widget(self.preset_gallery(LayoutMode::Desktop))
                .width(gallery_width)
                .height(Length::Fill),
            workspace,
        ]
        .spacing(12)
        .padding(CANVAS_PADDING)
        .height(Length::Fill)
        .width(Length::Fill)
        .into()
    }

    fn mobile_layout(&self, size: Size) -> Element<'_, Message> {
        if self.catalog_open {
            return container_widget(self.preset_gallery(LayoutMode::Mobile))
                .padding(12)
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        }
        if self.editor_open {
            return container_widget(
                self.grammar_editor(LayoutMode::Mobile, (size.height - 24.0).max(260.0)),
            )
            .padding(12)
            .width(Length::Fill)
            .height(Length::Fill)
            .into();
        }

        let content_width = (size.width - 24.0).max(240.0);
        let canvas_height = (content_width * 0.82).clamp(250.0, 440.0);
        let mut controls = column![
            self.iteration_controls(LayoutMode::Mobile),
            self.angle_controls(LayoutMode::Mobile),
            button_widget("Advanced")
                .on_press(Message::ToggleSettings)
                .style(button::secondary)
                .padding([10, 14])
                .width(Length::Fill),
        ]
        .spacing(14);
        if self.panel_expanded {
            controls = controls.push(self.advanced_controls(LayoutMode::Mobile, content_width));
        }
        controls = controls.push(self.status_footer(LayoutMode::Mobile, content_width - 24.0));
        scrollable(
            column![
                self.workspace_header(LayoutMode::Mobile),
                container_widget(self.canvas(LayoutMode::Mobile))
                    .height(canvas_height)
                    .width(Length::Fill),
                self.featured_presets(),
                container_widget(controls)
                    .padding(12)
                    .width(Length::Fill)
                    .style(panel_style),
            ]
            .spacing(12)
            .padding(12)
            .width(Length::Fill),
        )
        .id("explore-scroll")
        .on_scroll(|viewport| Message::ExploreScrolled(viewport.absolute_offset()))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn workspace_header(&self, layout: LayoutMode) -> Element<'_, Message> {
        let title = self
            .selected_preset
            .and_then(|choice| self.presets.get(choice.0))
            .map_or("Custom system", |preset| preset.name.as_str());
        let heading = column![text("BRAKEN").size(10), text(title).size(21)].spacing(2);
        let mut header = row![heading, Space::new().width(Length::Fill)]
            .spacing(10)
            .align_y(Alignment::Center)
            .width(Length::Fill);
        if layout == LayoutMode::Desktop {
            header = header.push(self.theme_button());
        }
        header.into()
    }

    fn theme_button(&self) -> Element<'_, Message> {
        let (icon, help) = match self.theme_preference {
            ThemePreference::Auto => (
                include_bytes!("../assets/sun-moon.svg").as_slice(),
                "Switch to light theme",
            ),
            ThemePreference::Light => (
                include_bytes!("../assets/sun.svg").as_slice(),
                "Switch to dark theme",
            ),
            ThemePreference::Dark => (
                include_bytes!("../assets/moon.svg").as_slice(),
                "Follow system theme",
            ),
        };
        tooltip(
            icon_button(icon, true).on_press(Message::CycleTheme),
            container_widget(text(help).size(12))
                .padding([6, 9])
                .style(tooltip_style),
            tooltip::Position::Bottom,
        )
        .into()
    }

    fn desktop_control_bar(&self, width: f32) -> Element<'_, Message> {
        let wide = width >= 830.0;
        let control_width = if wide { 170.0 } else { 180.0 };
        let status_width = if wide {
            (width - 122.0 - 2.0 * control_width - 86.0 - 56.0).max(226.0)
        } else {
            (width - 24.0).max(240.0)
        };
        let bar = row![
            button_widget(
                text(if self.editor_open {
                    "Close editor"
                } else {
                    "Edit grammar"
                })
                .size(14)
            )
            .on_press(Message::ToggleEditor(LayoutMode::Desktop))
            .style(button::secondary)
            .padding([8, 12])
            .height(36)
            .width(122),
            container_widget(self.iteration_controls(LayoutMode::Desktop)).width(control_width),
            container_widget(self.angle_controls(LayoutMode::Desktop)).width(control_width),
            button_widget(text("Advanced").size(14))
                .on_press(Message::ToggleSettings)
                .style(button::secondary)
                .padding([8, 8])
                .height(36)
                .width(86),
            container_widget(self.status_footer(LayoutMode::Desktop, status_width))
                .width(status_width),
        ]
        .spacing(8)
        .align_y(Alignment::End)
        .width(Length::Fill)
        .wrap()
        .vertical_spacing(12);
        container_widget(bar)
            .padding(12)
            .width(Length::Fill)
            .style(status_style)
            .into()
    }

    fn iteration_controls(&self, layout: LayoutMode) -> Element<'_, Message> {
        let mobile = layout == LayoutMode::Mobile;
        let value: Element<'_, Message> = if mobile {
            text(self.iterations.to_string()).size(14).into()
        } else {
            text_input("0", &self.iterations_input)
                .on_input(Message::IterationsInputChanged)
                .padding([2, 5])
                .size(13)
                .width(58)
                .into()
        };
        let soft_max = self.suggested_max_iterations.max(1);
        let slider = mouse_area(
            slider(
                0.0..=soft_max as f32,
                self.iterations.min(soft_max) as f32,
                Message::IterationsChanged,
            )
            .step(1.0),
        )
        .on_scroll(Message::IterationsScrolled);
        let touch_size = if mobile { 44.0 } else { 36.0 };
        let mut controls = column![
            row![
                text("Iterations").size(13),
                Space::new().width(Length::Fill),
                value
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        ]
        .spacing(5)
        .width(Length::Fill);
        if let Some(notice) = &self.iterations_notice {
            controls = controls.push(text(notice).size(12));
        }
        controls
            .push(
                row![
                    step_button("−", Message::IterationsDecrement, touch_size),
                    slider,
                    step_button("+", Message::IterationsIncrement, touch_size),
                ]
                .spacing(6)
                .align_y(Alignment::Center),
            )
            .into()
    }

    fn angle_controls(&self, layout: LayoutMode) -> Element<'_, Message> {
        let touch_size = if layout == LayoutMode::Mobile {
            44.0
        } else {
            36.0
        };
        column![
            row![
                text("Angle").size(13),
                Space::new().width(Length::Fill),
                text(format!("{}°", formatted_angle(self.angle))).size(14),
            ]
            .spacing(6)
            .align_y(Alignment::Center)
            .height(25),
            row![
                step_button("−", Message::AngleDecrement, touch_size),
                mouse_area(
                    slider(MIN_ANGLE..=MAX_ANGLE, self.angle, Message::AngleChanged)
                        .step(ANGLE_SLIDER_STEP),
                )
                .on_scroll(Message::AngleScrolled),
                step_button("+", Message::AngleIncrement, touch_size),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        ]
        .spacing(5)
        .width(Length::Fill)
        .into()
    }

    fn advanced_controls(&self, layout: LayoutMode, width: f32) -> Element<'_, Message> {
        let line_width: Element<'_, Message> = column![
            text(format!("Line width · {:.0}%", self.line_width_percent)).size(13),
            container_widget(
                slider(
                    MIN_LINE_WIDTH_PERCENT..=MAX_LINE_WIDTH_PERCENT,
                    self.line_width_percent,
                    Message::LineWidthChanged,
                )
                .step(LINE_WIDTH_SLIDER_STEP),
            )
            .height(38)
            .center_y(38),
        ]
        .spacing(6)
        .into();
        let seed: Element<'_, Message> = column![
            text("Seed").size(13),
            row![
                text_input(&self.seed_placeholder, &self.seed_input)
                    .on_input(Message::SeedChanged)
                    .padding(8)
                    .size(13),
                tooltip(
                    icon_button(include_bytes!("../assets/dices.svg").as_slice(), true)
                        .on_press(Message::RandomizeSeed),
                    container_widget(text("Generate random seed").size(12))
                        .padding([6, 9])
                        .style(tooltip_style),
                    tooltip::Position::Top,
                ),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        ]
        .spacing(6)
        .into();
        let precision: Element<'_, Message> = column![
            text("Floating-point precision").size(13),
            pick_list(
                [FloatWidth::F32, FloatWidth::F64],
                Some(self.derivation_semantics.float_width),
                Message::FloatWidthChanged,
            )
            .padding(8)
            .text_size(13)
            .width(Length::Fill),
        ]
        .spacing(6)
        .into();
        let rules: Element<'_, Message> = column![
            text("Ambiguous rules").size(13),
            pick_list(
                [
                    AmbiguousRulePolicy::Uniform,
                    AmbiguousRulePolicy::First,
                    AmbiguousRulePolicy::Error
                ],
                Some(self.derivation_semantics.ambiguous_rules),
                Message::AmbiguousRulesChanged,
            )
            .padding(8)
            .text_size(13)
            .width(Length::Fill),
        ]
        .spacing(6)
        .into();
        let fields: Element<'_, Message> = if layout == LayoutMode::Mobile {
            column![line_width, seed, precision, rules]
                .spacing(14)
                .into()
        } else {
            let field_width = if width >= 740.0 {
                (width - 60.0) / 4.0
            } else {
                (width - 36.0) / 2.0
            };
            row![
                container_widget(line_width).width(field_width),
                container_widget(seed).width(field_width),
                container_widget(precision).width(field_width),
                container_widget(rules).width(field_width),
            ]
            .spacing(12)
            .width(Length::Fill)
            .wrap()
            .vertical_spacing(12)
            .into()
        };
        let mut content = column![fields].spacing(12);
        if let Some(notice) = &self.seed_notice {
            content = content.push(text(notice).size(12));
        }
        if layout == LayoutMode::Mobile {
            content = content.push(
                button_widget("Edit grammar")
                    .on_press(Message::ToggleEditor(LayoutMode::Mobile))
                    .style(button::secondary)
                    .padding([10, 14])
                    .width(Length::Fill),
            );
        }
        container_widget(content)
            .padding(if layout == LayoutMode::Desktop { 12 } else { 0 })
            .width(Length::Fill)
            .style(panel_style)
            .into()
    }

    fn grammar_editor(&self, layout: LayoutMode, height: f32) -> Element<'_, Message> {
        let mobile = layout == LayoutMode::Mobile;
        let tabs = row![
            button_widget("Grammar")
                .on_press(Message::DockViewSelected(DockView::System))
                .style(if self.dock_view == DockView::System {
                    button::primary
                } else {
                    button::secondary
                }),
            button_widget("IR")
                .on_press(Message::DockViewSelected(DockView::Ir))
                .style(if self.dock_view == DockView::Ir {
                    button::primary
                } else {
                    button::secondary
                }),
        ]
        .spacing(6);
        let mut actions = row![].spacing(8);
        if self.editor_draft_mode {
            actions = actions.push(button_widget("Apply & return").on_press(Message::ApplyDraft));
        }
        let header: Element<'_, Message> = if mobile {
            column![
                row![
                    text("Edit grammar").size(21),
                    Space::new().width(Length::Fill),
                    button_widget("Back")
                        .on_press(Message::CloseEditor)
                        .style(button::secondary),
                ]
                .align_y(Alignment::Center),
                row![tabs, Space::new().width(Length::Fill), actions]
                    .spacing(8)
                    .align_y(Alignment::Center),
            ]
            .spacing(10)
            .into()
        } else {
            row![tabs, actions]
                .spacing(8)
                .align_y(Alignment::Center)
                .wrap()
                .vertical_spacing(8)
                .into()
        };
        let content: Element<'_, Message> = match self.dock_view {
            DockView::System => {
                let editor = if self.editor_draft_mode {
                    text_editor(&self.editor_draft).on_action(Message::DraftEdited)
                } else {
                    text_editor(&self.source_editor).on_action(Message::SourceEdited)
                };
                let hint = if self.editor_draft_mode {
                    if self.draft_is_modified() {
                        "Unapplied changes · Apply to update the scene"
                    } else {
                        "Changes stay here until you apply them"
                    }
                } else {
                    "Changes update the scene automatically"
                };
                column![
                    text(hint).size(12),
                    editor
                        .placeholder("axiom Draw;\nmatch Draw then Draw Turn(60) Draw;")
                        .font(iced::Font::MONOSPACE)
                        .size(13)
                        .padding(10)
                        .height(Length::Fill),
                ]
                .spacing(8)
                .height(Length::Fill)
                .into()
            }
            DockView::Ir => self.ir_panel(),
        };
        container_widget(column![header, content].spacing(10).height(Length::Fill))
            .padding(12)
            .height(height)
            .width(Length::Fill)
            .style(panel_style)
            .into()
    }

    fn preset_gallery(&self, layout: LayoutMode) -> Element<'_, Message> {
        let mut heading = row![text("Presets").size(21), Space::new().width(Length::Fill)]
            .align_y(Alignment::Center);
        if layout == LayoutMode::Mobile {
            heading = heading.push(
                button_widget("Back")
                    .on_press(Message::CloseCatalog)
                    .style(button::secondary)
                    .padding([10, 14]),
            );
        }
        let matches = self.matching_preset_indices();
        let count = matches.len();
        let mut cards = column![].spacing(8).width(Length::Fill);
        if matches.is_empty() {
            cards = cards.push(text("No presets match this search.").size(13));
        }
        for index in matches {
            cards = cards.push(preset_button(
                index,
                &self.presets[index],
                self.preset_previews[index].for_theme(self.effective_dark()),
                self.selected_preset == Some(PresetChoice(index)),
            ));
        }
        container_widget(
            column![
                heading,
                row![
                    text_input("Search presets", &self.preset_search)
                        .on_input(Message::PresetSearchChanged)
                        .padding(8)
                        .size(13)
                        .width(Length::Fill),
                    button_widget("Clear")
                        .on_press(Message::ClearPresetSearch)
                        .style(button::secondary),
                ]
                .spacing(8)
                .align_y(Alignment::Center),
                text(format!("{count} presets")).size(12),
                scrollable(container_widget(cards).padding(iced::Padding {
                    right: 12.0,
                    ..iced::Padding::default()
                }))
                .height(Length::Fill)
                .width(Length::Fill),
            ]
            .spacing(10)
            .height(Length::Fill),
        )
        .padding(12)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(panel_style)
        .into()
    }

    fn featured_presets(&self) -> Element<'_, Message> {
        let mut cards = row![].spacing(10);
        for name in FEATURED_PRESETS {
            let Some((index, preset)) = self
                .presets
                .iter()
                .enumerate()
                .find(|(_, preset)| preset.name == name)
            else {
                continue;
            };
            let selected = self.selected_preset == Some(PresetChoice(index));
            let is_3d = preset.visualizer == VisualizerKind::Turtle3d;
            let preview = container_widget(
                svg(self.preset_previews[index]
                    .for_theme(self.effective_dark())
                    .clone())
                .width(Length::Fill)
                .height(72),
            )
            .width(Length::Fill)
            .height(72)
            .style(move |theme| preview_style(theme, selected, is_3d));
            let mut copy = column![preview, text(&preset.name).size(13)].spacing(6);
            if let Some(label) = preset.dimension_badge_label() {
                copy = copy.push(
                    container_widget(text(label).size(10))
                        .padding([2, 6])
                        .style(move |theme| dimension_badge_style(theme, is_3d)),
                );
            }
            cards = cards.push(
                button_widget(copy)
                    .width(146)
                    .height(142)
                    .padding(8)
                    .style(move |theme, status| preset_button_style(theme, status, selected))
                    .on_press(Message::PresetSelected(PresetChoice(index))),
            );
        }
        column![
            row![
                text("Presets").size(17),
                Space::new().width(Length::Fill),
                button_widget("Browse all")
                    .on_press(Message::OpenCatalog)
                    .style(button::secondary)
                    .padding([10, 12]),
            ]
            .align_y(Alignment::Center),
            scrollable(cards)
                .horizontal()
                .height(154)
                .width(Length::Fill),
        ]
        .spacing(8)
        .into()
    }

    fn status_footer(&self, layout: LayoutMode, width: f32) -> Element<'_, Message> {
        let now = RenderInstant::now();
        let (_, status) = self.render_scene_and_status_at(now);
        let mut performance = match (self.last_render_ms, self.last_render_backend.as_deref()) {
            (Some(milliseconds), Some(backend)) => {
                format!("{milliseconds} ms on {backend} · {} FPS", self.fps)
            }
            _ => format!("— ms · {} FPS", self.fps),
        };
        let slow = self
            .render
            .active
            .as_ref()
            .is_some_and(|job| should_offer_render_cancel(now.duration_since(job.started_at)))
            || self.render.displayed.as_ref().is_some_and(|result| {
                (result
                    .line_scene
                    .as_ref()
                    .is_some_and(|scene| scene.is_refining())
                    || (!self.autorotate_3d
                        && !self.view_gesture_active
                        && result
                            .spatial_scene
                            .as_ref()
                            .is_some_and(|scene| scene.is_refining())))
                    && should_offer_render_cancel(now.duration_since(result.refinement_started_at))
            });
        if slow {
            performance.push_str(" · This is taking a while");
        }
        let primary = self
            .render
            .failure
            .as_ref()
            .map_or(status.clone(), |message| format!("{status} · {message}"));
        let failed = self.render.failure.is_some();
        let mut summary = column![
            text(primary)
                .size(12)
                .style(move |theme: &Theme| text::Style {
                    color: Some(if failed {
                        theme.extended_palette().danger.base.color
                    } else {
                        theme.palette().text
                    }),
                })
        ]
        .spacing(4)
        .width(Length::Fill);
        if layout == LayoutMode::Desktop || self.diagnostics_open {
            summary = summary.push(text(performance).size(11));
        }
        if layout == LayoutMode::Mobile {
            summary = summary.push(
                button_widget(
                    text(if self.diagnostics_open {
                        "Hide details"
                    } else {
                        "Details"
                    })
                    .size(12),
                )
                .style(button::text)
                .padding([4, 0])
                .on_press(Message::ToggleDiagnostics),
            );
        }
        if let Some(notice) = &self.export_notice {
            summary = summary.push(text(notice).size(12));
        }
        let export: Element<'_, Message> = if self.export_job.is_some() {
            button_widget(text("Cancel SVG").size(12))
                .style(button::secondary)
                .on_press(Message::CancelExport)
                .into()
        } else {
            let download = icon_button(
                include_bytes!("../assets/download.svg").as_slice(),
                self.render.displayed.is_some(),
            )
            .on_press_maybe(self.render.displayed.as_ref().map(|_| Message::ExportSvg));
            tooltip(
                download,
                container_widget(text("Download SVG").size(12))
                    .padding([6, 9])
                    .style(tooltip_style),
                tooltip::Position::Top,
            )
            .into()
        };
        let mut actions = row![].spacing(6).align_y(Alignment::End);
        if layout == LayoutMode::Mobile {
            actions = actions.push(self.theme_button());
        }
        if slow {
            actions = actions.push(
                button_widget(text("Cancel").size(12))
                    .style(button::secondary)
                    .on_press(Message::CancelRender),
            );
        }
        actions = actions.push(export);
        if width < 360.0 && self.export_job.is_some() && slow {
            column![summary, container_widget(actions).align_right(Length::Fill)]
                .spacing(6)
                .into()
        } else {
            row![summary, actions]
                .spacing(8)
                .align_y(Alignment::End)
                .into()
        }
    }
}

fn icon_button(icon: &'static [u8], enabled: bool) -> button::Button<'static, Message> {
    button_widget(
        container_widget(
            svg(svg::Handle::from_memory(icon))
                .width(20)
                .height(20)
                .style(move |theme, status| {
                    if enabled {
                        header_icon_style(theme, status)
                    } else {
                        disabled_icon_style(theme, status)
                    }
                }),
        )
        .center_x(Length::Fill)
        .center_y(Length::Fill),
    )
    .width(38)
    .height(38)
    .padding(0)
    .style(header_icon_button_style)
}

fn step_button(
    label: &'static str,
    message: Message,
    size: f32,
) -> button::Button<'static, Message> {
    button_widget(
        container_widget(text(label).size(18))
            .center_x(Length::Fill)
            .center_y(Length::Fill),
    )
    .on_press(message)
    .style(button::secondary)
    .padding(0)
    .width(size)
    .height(size)
}

#[cfg(test)]
mod tests {
    use super::FEATURED_PRESETS;

    #[test]
    fn featured_presets_resolve_once_in_the_shared_catalog() {
        let presets = crate::presets::get_presets();
        for name in FEATURED_PRESETS {
            assert_eq!(
                presets.iter().filter(|preset| preset.name == name).count(),
                1,
                "featured preset {name} must exist exactly once",
            );
        }
    }
}
