//! Adjacent-generation Turtle line correspondence.
//!
//! Derivation owns module parentage, Turtle visualization owns the indexed
//! lines, and this module combines those backend-neutral products. It does not
//! derive, traverse turtle state, or rasterize.

use crate::{Line2d, StyledLine2d, VisualizeError};
use braken::RewriteLineage;

/// A Turtle line tagged with the depth-first module index that emitted it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexedLine2d {
    pub module_index: usize,
    pub line: StyledLine2d,
}

/// Turtle position at which a generation module begins executing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexedModulePosition2d {
    pub module_index: usize,
    pub position: (f64, f64),
}

/// Generation coordinate system used by one morph endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphCoordinateSpace {
    Source,
    Target,
}

impl MorphCoordinateSpace {
    fn reversed(self) -> Self {
        match self {
            Self::Source => Self::Target,
            Self::Target => Self::Source,
        }
    }
}

/// One independently interpolated piece of an adjacent-generation transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorphLine2d {
    pub source: StyledLine2d,
    pub target: StyledLine2d,
    pub source_space: MorphCoordinateSpace,
    pub target_space: MorphCoordinateSpace,
    pub source_opacity: f32,
    pub target_opacity: f32,
    /// Full source line used to keep subdivided pieces in the parent's color.
    pub source_palette_line: Line2d,
    /// Target descendant used for its final palette position.
    pub target_palette_line: Line2d,
}

impl MorphLine2d {
    fn reversed(self) -> Self {
        Self {
            source: self.target,
            target: self.source,
            source_space: self.target_space.reversed(),
            target_space: self.source_space.reversed(),
            source_opacity: self.target_opacity,
            target_opacity: self.source_opacity,
            source_palette_line: self.target_palette_line,
            target_palette_line: self.source_palette_line,
        }
    }
}

/// Builds line morphs for one exact adjacent rewrite.
///
/// A drawing predecessor is divided into equal ordered pieces for its drawing
/// descendants. A predecessor with no drawing descendants collapses into its
/// midpoint, while drawing descendants of a non-drawing predecessor grow from
/// that predecessor's saved Turtle position. Endpoint coordinate spaces are
/// retained explicitly so independently fitted views do not reinterpret a
/// source position as target coordinates. `reverse` swaps the prepared
/// endpoints; it never samples an inverse grammar.
pub fn build_line_transition(
    source: &[IndexedLine2d],
    source_positions: &[IndexedModulePosition2d],
    target: &[IndexedLine2d],
    lineage: &RewriteLineage,
    reverse: bool,
) -> Result<Vec<MorphLine2d>, VisualizeError> {
    validate_indices(source, lineage.input_modules(), "source")?;
    validate_positions(source_positions, lineage.input_modules())?;
    let target_modules = lineage
        .output_modules()
        .ok_or_else(|| resource_error("transition successor-module count", usize::MAX))?;
    validate_indices(target, target_modules, "target")?;

    let capacity = source
        .len()
        .checked_add(target.len())
        .ok_or_else(|| resource_error("transition line count", usize::MAX))?;
    let mut output = Vec::new();
    output
        .try_reserve(capacity)
        .map_err(|_| resource_error("transition lines", capacity))?;

    let mut source_line = 0usize;
    let mut target_line = 0usize;
    let mut target_module = 0usize;
    for (parent, successor_count) in lineage
        .successor_modules_per_input()
        .iter()
        .copied()
        .enumerate()
    {
        let target_end = target_module
            .checked_add(successor_count)
            .ok_or_else(|| resource_error("transition successor range", usize::MAX))?;
        let source_for_parent = source
            .get(source_line)
            .filter(|line| line.module_index == parent)
            .copied();
        if source_for_parent.is_some() {
            source_line += 1;
        }

        let descendant_start = target_line;
        while target
            .get(target_line)
            .is_some_and(|line| line.module_index < target_end)
        {
            if target[target_line].module_index < target_module {
                return Err(invalid_lineage("target line precedes its lineage range"));
            }
            target_line += 1;
        }
        let descendants = &target[descendant_start..target_line];

        match (source_for_parent, descendants.is_empty()) {
            (Some(parent_line), false) => {
                let pieces = descendants.len();
                for (piece, descendant) in descendants.iter().enumerate() {
                    let start = piece_point(parent_line.line.line, piece, pieces);
                    let end = piece_point(parent_line.line.line, piece + 1, pieces);
                    output.push(MorphLine2d {
                        source: StyledLine2d {
                            line: Line2d(start, end),
                            width: parent_line.line.width,
                            color: parent_line.line.color,
                        },
                        target: descendant.line,
                        source_space: MorphCoordinateSpace::Source,
                        target_space: MorphCoordinateSpace::Target,
                        source_opacity: 1.0,
                        target_opacity: 1.0,
                        source_palette_line: parent_line.line.line,
                        target_palette_line: descendant.line.line,
                    });
                }
            }
            (Some(parent_line), true) => {
                let midpoint = midpoint(parent_line.line.line);
                output.push(MorphLine2d {
                    source: parent_line.line,
                    target: StyledLine2d {
                        line: Line2d(midpoint, midpoint),
                        width: 0.0,
                        color: parent_line.line.color,
                    },
                    source_space: MorphCoordinateSpace::Source,
                    target_space: MorphCoordinateSpace::Source,
                    source_opacity: 1.0,
                    target_opacity: 0.0,
                    source_palette_line: parent_line.line.line,
                    target_palette_line: parent_line.line.line,
                });
            }
            (None, false) => {
                let anchor = source_positions[parent].position;
                for descendant in descendants {
                    output.push(MorphLine2d {
                        source: StyledLine2d {
                            line: Line2d(anchor, anchor),
                            width: 0.0,
                            color: descendant.line.color,
                        },
                        target: descendant.line,
                        source_space: MorphCoordinateSpace::Source,
                        target_space: MorphCoordinateSpace::Target,
                        source_opacity: 0.0,
                        target_opacity: 1.0,
                        source_palette_line: Line2d(anchor, anchor),
                        target_palette_line: descendant.line.line,
                    });
                }
            }
            (None, true) => {}
        }
        target_module = target_end;
    }

    if source_line != source.len() || target_line != target.len() || target_module != target_modules
    {
        return Err(invalid_lineage(
            "indexed line data does not cover the declared rewrite lineage",
        ));
    }
    if reverse {
        for line in &mut output {
            *line = line.reversed();
        }
    }
    Ok(output)
}

fn validate_positions(
    positions: &[IndexedModulePosition2d],
    module_count: usize,
) -> Result<(), VisualizeError> {
    if positions.len() != module_count
        || positions.iter().enumerate().any(|(expected, position)| {
            position.module_index != expected
                || !position.position.0.is_finite()
                || !position.position.1.is_finite()
        })
    {
        return Err(invalid_lineage(
            "source module positions are missing, invalid, or out of order",
        ));
    }
    Ok(())
}

fn validate_indices(
    lines: &[IndexedLine2d],
    module_count: usize,
    role: &'static str,
) -> Result<(), VisualizeError> {
    let mut previous = None;
    for line in lines {
        if line.module_index >= module_count
            || previous.is_some_and(|index| line.module_index <= index)
        {
            return Err(invalid_lineage(match role {
                "source" => "source line module indices are invalid or out of order",
                _ => "target line module indices are invalid or out of order",
            }));
        }
        previous = Some(line.module_index);
    }
    Ok(())
}

fn piece_point(line: Line2d, numerator: usize, denominator: usize) -> (f64, f64) {
    let progress = numerator as f64 / denominator as f64;
    (
        line.0.0 + (line.1.0 - line.0.0) * progress,
        line.0.1 + (line.1.1 - line.0.1) * progress,
    )
}

fn midpoint(line: Line2d) -> (f64, f64) {
    ((line.0.0 + line.1.0) * 0.5, (line.0.1 + line.1.1) * 0.5)
}

fn resource_error(resource: &'static str, requested: usize) -> VisualizeError {
    VisualizeError::ResourceExhausted {
        resource,
        requested: Some(requested),
    }
}

fn invalid_lineage(reason: &'static str) -> VisualizeError {
    VisualizeError::InvalidConfiguration(format!("invalid rewrite lineage: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Turtle2dConfig, Turtle2dStreamRequest, stream_turtle_2d};
    use braken::Grammar;

    fn indexed(module_index: usize, start: f64, end: f64) -> IndexedLine2d {
        IndexedLine2d {
            module_index,
            line: StyledLine2d {
                line: Line2d((start, 0.0), (end, 0.0)),
                width: 1.0,
                color: crate::StrokeColor::ThemeDefault,
            },
        }
    }

    fn positions(values: &[(f64, f64)]) -> Vec<IndexedModulePosition2d> {
        values
            .iter()
            .copied()
            .enumerate()
            .map(|(module_index, position)| IndexedModulePosition2d {
                module_index,
                position,
            })
            .collect()
    }

    fn lineage(source: &str) -> RewriteLineage {
        let grammar = Grammar::parse(source).unwrap();
        let program = grammar.compile_cpu().unwrap();
        let mut state = program.start();
        state
            .step_with_lineage_and_control(|| false, |_| {})
            .unwrap()
            .1
    }

    fn indexed_geometry(
        generation: &braken::Generation,
        config: Turtle2dConfig,
    ) -> (Vec<IndexedLine2d>, Vec<IndexedModulePosition2d>) {
        let mut lines = Vec::new();
        let mut positions = Vec::new();
        stream_turtle_2d(
            Turtle2dStreamRequest {
                generation,
                config,
                batch_size: 8,
                is_cancelled: &|| false,
            },
            |batch| {
                lines.extend(
                    batch
                        .lines
                        .into_iter()
                        .zip(batch.module_indices)
                        .map(|(line, module_index)| IndexedLine2d { module_index, line }),
                );
                positions.extend(batch.module_positions);
                Ok(())
            },
        )
        .unwrap();
        (lines, positions)
    }

    #[test]
    fn subdivides_parent_in_descendant_order() {
        let lineage = lineage("axiom A; match A then B C D;");
        let morphs = build_line_transition(
            &[indexed(0, 0.0, 3.0)],
            &positions(&[(0.0, 0.0)]),
            &[
                indexed(0, 0.0, 1.0),
                indexed(1, 1.0, 2.0),
                indexed(2, 2.0, 3.0),
            ],
            &lineage,
            false,
        )
        .unwrap();
        assert_eq!(morphs.len(), 3);
        assert_eq!(morphs[0].source.line, Line2d((0.0, 0.0), (1.0, 0.0)));
        assert_eq!(morphs[2].source.line, Line2d((2.0, 0.0), (3.0, 0.0)));
    }

    #[test]
    fn deletion_collapses_and_reverse_grows() {
        let lineage = lineage("axiom A; match A then nothing;");
        let source_positions = positions(&[(0.0, 0.0)]);
        let forward = build_line_transition(
            &[indexed(0, 0.0, 2.0)],
            &source_positions,
            &[],
            &lineage,
            false,
        )
        .unwrap();
        assert_eq!(forward[0].target.line, Line2d((1.0, 0.0), (1.0, 0.0)));
        assert_eq!(forward[0].target_space, MorphCoordinateSpace::Source);
        assert_eq!(forward[0].target_opacity, 0.0);
        let reverse = build_line_transition(
            &[indexed(0, 0.0, 2.0)],
            &source_positions,
            &[],
            &lineage,
            true,
        )
        .unwrap();
        assert_eq!(reverse[0].source_opacity, 0.0);
        assert_eq!(reverse[0].source_space, MorphCoordinateSpace::Target);
        assert_eq!(reverse[0].target.line, Line2d((0.0, 0.0), (2.0, 0.0)));
    }

    #[test]
    fn non_drawing_parent_grows_drawing_descendant() {
        let lineage = lineage("axiom A; match A then F;");
        let morphs = build_line_transition(
            &[],
            &positions(&[(2.0, 3.0)]),
            &[indexed(0, 4.0, 5.0)],
            &lineage,
            false,
        )
        .unwrap();
        assert_eq!(morphs[0].source.line, Line2d((2.0, 3.0), (2.0, 3.0)));
        assert_eq!(morphs[0].source_space, MorphCoordinateSpace::Source);
        assert_eq!(morphs[0].target_space, MorphCoordinateSpace::Target);
        assert_eq!(morphs[0].source_opacity, 0.0);
    }

    #[test]
    fn hilbert_descendants_grow_from_their_invisible_parents_source_positions() {
        let grammar = Grammar::parse(
            "axiom L; \
             match L then Left R Draw Right L Draw L Right Draw R Left; \
             match R then Right L Draw Left R Draw R Left Draw L Right;",
        )
        .unwrap();
        let mut state = grammar.compile_cpu().unwrap().start();
        state.step().unwrap();
        let source_generation = state.generation().clone();
        let (_, lineage) = state
            .step_with_lineage_and_control(|| false, |_| {})
            .unwrap();
        let target_generation = state.generation().clone();
        let config = Turtle2dConfig::default();
        let (source_lines, source_positions) = indexed_geometry(&source_generation, config.clone());
        let (target_lines, _) = indexed_geometry(&target_generation, config);

        let morphs = build_line_transition(
            &source_lines,
            &source_positions,
            &target_lines,
            &lineage,
            false,
        )
        .unwrap();
        assert_eq!(source_lines.len(), 3);
        assert_eq!(target_lines.len(), 15);
        assert_eq!(morphs.len(), 15);

        let born = morphs
            .iter()
            .filter(|morph| morph.source_opacity == 0.0)
            .collect::<Vec<_>>();
        assert_eq!(born.len(), 12);
        assert!(born.iter().all(|morph| {
            morph.source.line.0 == morph.source.line.1
                && morph.source_space == MorphCoordinateSpace::Source
                && morph.target_space == MorphCoordinateSpace::Target
        }));
        assert!(
            born.iter()
                .any(|morph| morph.source.line.0 != morph.target.line.0),
            "the regression would put target coordinates through the source fit"
        );
    }
}
