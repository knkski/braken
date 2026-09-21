use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{self, Write};

use braken::{Generation, GenerationItem};

use crate::{
    InspectorConfig, InspectorPhase, InspectorProgress, InspectorStreamRequest, Primitive2d,
    Scene2d, Text2d, TextRole, VisualizationContext, VisualizeError, VisualizerBackend,
    VisualizerKind,
};

const PREVIEW_BYTES: usize = 16 * 1024;
const PREVIEW_WRAP_WIDTH: usize = 100;
const COMPATIBILITY_WORK_QUANTUM: usize = 16 * 1024;
const FREQUENCY_ROWS: usize = 12;

pub(crate) fn visualize_cpu(
    generation: &Generation,
    config: InspectorConfig,
    context: VisualizationContext,
) -> Result<Scene2d, VisualizeError> {
    let never_cancelled = || false;
    stream_cpu(
        InspectorStreamRequest {
            generation,
            config,
            context,
            work_quantum: COMPATIBILITY_WORK_QUANTUM,
            is_cancelled: &never_cancelled,
        },
        |_| Ok(()),
    )
}

pub(crate) fn stream_cpu(
    request: InspectorStreamRequest<'_>,
    mut emit: impl FnMut(InspectorProgress) -> Result<(), VisualizeError>,
) -> Result<Scene2d, VisualizeError> {
    if request.work_quantum == 0 {
        return Err(VisualizeError::InvalidConfiguration(
            "inspector work quantum must be greater than zero".into(),
        ));
    }
    ensure_running(request.is_cancelled)?;

    let stats = collect_stats(&request, &mut emit)?;
    let top = top_frequencies(&stats, request.is_cancelled)?;
    let (preview, complete) = bounded_preview(&request, &stats, PREVIEW_BYTES, &mut emit)?;
    let wrapped = wrap_preview(&preview, PREVIEW_WRAP_WIDTH, request.is_cancelled)?;

    let requested_lines = 6usize
        .checked_add(top.len())
        .and_then(|count| count.checked_add(wrapped.len()))
        .ok_or_else(|| resource_error("inspector text rows", None))?;
    let mut lines = Vec::new();
    reserve_vec(&mut lines, requested_lines, "inspector text rows")?;
    lines.push((String::from("Inspector"), 26.0, TextRole::Title));
    lines.push((
        fallible_format(
            format_args!(
                "Iterations: {}    Seed: {}",
                request.context.iterations, request.context.seed
            ),
            96,
            "inspector iteration summary",
        )?,
        16.0,
        TextRole::Body,
    ));
    let calculation = match request.context.elapsed {
        Some(elapsed) => fallible_format(
            format_args!("{} ms", elapsed.as_millis()),
            48,
            "inspector elapsed time",
        )?,
        None => String::from("unknown"),
    };
    lines.push((
        fallible_format(
            format_args!(
                "Backend: {}    Calculation: {}",
                request.context.derivation_backend.unwrap_or("unknown"),
                calculation
            ),
            request
                .context
                .derivation_backend
                .map_or(128, |backend| backend.len().saturating_add(128)),
            "inspector backend summary",
        )?,
        16.0,
        TextRole::Body,
    ));
    lines.push((
        fallible_format(
            format_args!(
                "Modules: {}    Items: {}    Branches: {}    Maximum depth: {}",
                stats.modules, stats.items, stats.branches, stats.max_depth
            ),
            160,
            "inspector statistics summary",
        )?,
        16.0,
        TextRole::Body,
    ));
    lines.push((String::from("Most common modules"), 19.0, TextRole::Heading));
    for (name, count) in top {
        lines.push((
            fallible_format(
                format_args!("{name}: {count}"),
                name.len().saturating_add(32),
                "inspector frequency row",
            )?,
            15.0,
            TextRole::Body,
        ));
    }
    lines.push((
        if stats.items <= 500 && complete {
            String::from("Production")
        } else {
            String::from("Production preview")
        },
        19.0,
        TextRole::Heading,
    ));
    lines.extend(
        wrapped
            .into_iter()
            .map(|line| (line, 14.0, TextRole::Muted)),
    );

    ensure_running(request.is_cancelled)?;
    emit(progress_for(
        InspectorPhase::Complete,
        &stats,
        preview.len(),
    ))?;

    let mut primitives = Vec::new();
    reserve_vec(&mut primitives, lines.len(), "inspector scene primitives")?;
    for (index, (content, size, role)) in lines.into_iter().enumerate() {
        primitives.push(Primitive2d::Text(Text2d {
            position: (28.0, 42.0 + index as f64 * 25.0),
            content,
            size,
            role,
        }));
    }
    Ok(Scene2d {
        primitives,
        background: None,
    })
}

pub(crate) fn visualize_cuda(
    _: &Generation,
    _: InspectorConfig,
    _: VisualizationContext,
) -> Result<Scene2d, VisualizeError> {
    Err(VisualizeError::Unimplemented {
        visualizer: VisualizerKind::Inspector,
        backend: VisualizerBackend::Cuda,
    })
}

#[derive(Default)]
struct Stats<'a> {
    modules: usize,
    items: usize,
    branches: usize,
    max_depth: usize,
    frequencies: HashMap<&'a str, usize>,
}

fn collect_stats<'a>(
    request: &InspectorStreamRequest<'a>,
    emit: &mut impl FnMut(InspectorProgress) -> Result<(), VisualizeError>,
) -> Result<Stats<'a>, VisualizeError> {
    let mut stats = Stats::default();
    let mut pending = Vec::new();
    reserve_vec(&mut pending, 1, "inspector traversal stack")?;
    pending.push((request.generation, 0usize));
    let mut quantum_items = 0usize;

    while let Some((generation, depth)) = pending.pop() {
        for item in generation.items() {
            ensure_running(request.is_cancelled)?;
            stats.items = checked_increment(stats.items, "inspector item counter")?;
            quantum_items = checked_increment(quantum_items, "inspector work counter")?;
            match item {
                GenerationItem::Module(module) => {
                    stats.modules = checked_increment(stats.modules, "inspector module counter")?;
                    let name = module.name.as_str();
                    if let Some(count) = stats.frequencies.get_mut(name) {
                        *count = checked_increment(*count, "inspector module frequency")?;
                    } else {
                        stats.frequencies.try_reserve(1).map_err(|_| {
                            resource_error(
                                "inspector module frequency table",
                                stats.frequencies.len().checked_add(1),
                            )
                        })?;
                        stats.frequencies.insert(name, 1);
                    }
                }
                GenerationItem::Branch(branch) => {
                    stats.branches = checked_increment(stats.branches, "inspector branch counter")?;
                    let branch_depth = depth
                        .checked_add(1)
                        .ok_or_else(|| resource_error("inspector branch depth counter", None))?;
                    stats.max_depth = stats.max_depth.max(branch_depth);
                    reserve_vec(&mut pending, 1, "inspector traversal stack")?;
                    pending.push((branch, branch_depth));
                }
            }

            if quantum_items == request.work_quantum {
                emit(progress_for(InspectorPhase::Statistics, &stats, 0))?;
                quantum_items = 0;
            }
        }
    }
    if quantum_items != 0 || stats.items == 0 {
        emit(progress_for(InspectorPhase::Statistics, &stats, 0))?;
    }
    Ok(stats)
}

fn top_frequencies<'generation>(
    stats: &Stats<'generation>,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<Vec<(&'generation str, usize)>, VisualizeError> {
    let mut top = Vec::new();
    reserve_vec(
        &mut top,
        FREQUENCY_ROWS.saturating_add(1),
        "inspector top frequencies",
    )?;
    for (&name, &count) in &stats.frequencies {
        ensure_running(is_cancelled)?;
        top.push((name, count));
        top.sort_unstable_by(frequency_order);
        if top.len() > FREQUENCY_ROWS {
            top.pop();
        }
    }
    Ok(top)
}

fn frequency_order(left: &(&str, usize), right: &(&str, usize)) -> Ordering {
    right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0))
}

struct PreviewFrame<'a> {
    items: &'a [GenerationItem],
    next: usize,
    close_branch: bool,
}

fn bounded_preview(
    request: &InspectorStreamRequest<'_>,
    stats: &Stats<'_>,
    limit: usize,
    emit: &mut impl FnMut(InspectorProgress) -> Result<(), VisualizeError>,
) -> Result<(String, bool), VisualizeError> {
    let mut output = String::new();
    output
        .try_reserve_exact(limit)
        .map_err(|_| resource_error("inspector production preview", Some(limit)))?;
    let content_limit = limit.saturating_sub('…'.len_utf8());
    let mut writer = LimitedWriter {
        output: &mut output,
        limit: content_limit,
    };
    let mut pending = Vec::new();
    reserve_vec(&mut pending, 1, "inspector preview stack")?;
    pending.push(PreviewFrame {
        items: request.generation.items(),
        next: 0,
        close_branch: false,
    });
    let mut complete = true;
    let mut quantum_items = 0usize;

    while !pending.is_empty() {
        ensure_running(request.is_cancelled)?;
        let finished = pending
            .last()
            .is_some_and(|frame| frame.next == frame.items.len());
        if finished {
            let frame = pending
                .pop()
                .expect("preview stack was checked as non-empty");
            if frame.close_branch && writer.write_str(" ]").is_err() {
                complete = false;
                break;
            }
            continue;
        }

        let (index, item) = {
            let frame = pending
                .last_mut()
                .expect("preview stack was checked as non-empty");
            let index = frame.next;
            frame.next += 1;
            (index, &frame.items[index])
        };
        if index != 0 && writer.write_str(" ").is_err() {
            complete = false;
            break;
        }
        quantum_items = checked_increment(quantum_items, "inspector preview work counter")?;
        match item {
            GenerationItem::Module(module) => {
                if write!(&mut writer, "{module}").is_err() {
                    complete = false;
                    break;
                }
            }
            GenerationItem::Branch(branch) => {
                if writer.write_str("[ ").is_err() {
                    complete = false;
                    break;
                }
                reserve_vec(&mut pending, 1, "inspector preview stack")?;
                pending.push(PreviewFrame {
                    items: branch.items(),
                    next: 0,
                    close_branch: true,
                });
            }
        }
        if quantum_items == request.work_quantum {
            emit(progress_for(
                InspectorPhase::Preview,
                stats,
                writer.output.len(),
            ))?;
            quantum_items = 0;
        }
    }

    let preview_bytes = writer.output.len();
    if !complete && limit >= '…'.len_utf8() {
        output.push('…');
    }
    if quantum_items != 0 || complete {
        emit(progress_for(InspectorPhase::Preview, stats, preview_bytes))?;
    }
    Ok((output, complete))
}

struct LimitedWriter<'a> {
    output: &'a mut String,
    limit: usize,
}

impl fmt::Write for LimitedWriter<'_> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let Some(required) = self.output.len().checked_add(value.len()) else {
            return Err(fmt::Error);
        };
        if required > self.limit {
            return Err(fmt::Error);
        }
        self.output.push_str(value);
        Ok(())
    }
}

fn wrap_preview(
    value: &str,
    width: usize,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<Vec<String>, VisualizeError> {
    let mut lines = Vec::new();
    let estimated_lines = value
        .len()
        .checked_div(width.max(1))
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| resource_error("inspector preview rows", None))?;
    reserve_vec(&mut lines, estimated_lines, "inspector preview rows")?;
    let mut current = String::new();
    current
        .try_reserve(width.min(value.len()))
        .map_err(|_| resource_error("inspector preview row", Some(width.min(value.len()))))?;
    for word in value.split_whitespace() {
        ensure_running(is_cancelled)?;
        let separated_len = current
            .len()
            .checked_add(word.len())
            .and_then(|length| length.checked_add(usize::from(!current.is_empty())))
            .ok_or_else(|| resource_error("inspector preview row", None))?;
        if !current.is_empty() && separated_len > width {
            reserve_vec(&mut lines, 1, "inspector preview rows")?;
            lines.push(current);
            current = String::new();
            current
                .try_reserve(word.len())
                .map_err(|_| resource_error("inspector preview row", Some(word.len())))?;
            current.push_str(word);
        } else {
            current
                .try_reserve(separated_len.saturating_sub(current.len()))
                .map_err(|_| resource_error("inspector preview row", Some(separated_len)))?;
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
    }
    reserve_vec(&mut lines, 1, "inspector preview rows")?;
    lines.push(current);
    Ok(lines)
}

fn progress_for(
    phase: InspectorPhase,
    stats: &Stats<'_>,
    preview_bytes: usize,
) -> InspectorProgress {
    InspectorProgress {
        phase,
        items_processed: stats.items,
        modules: stats.modules,
        branches: stats.branches,
        max_branch_depth: stats.max_depth,
        preview_bytes,
    }
}

fn fallible_format(
    arguments: fmt::Arguments<'_>,
    capacity: usize,
    resource: &'static str,
) -> Result<String, VisualizeError> {
    let mut output = String::new();
    output
        .try_reserve(capacity)
        .map_err(|_| resource_error(resource, Some(capacity)))?;
    output
        .write_fmt(arguments)
        .map_err(|_| resource_error(resource, Some(capacity)))?;
    Ok(output)
}

fn reserve_vec<T>(
    values: &mut Vec<T>,
    additional: usize,
    resource: &'static str,
) -> Result<(), VisualizeError> {
    values
        .try_reserve(additional)
        .map_err(|_| resource_error(resource, values.len().checked_add(additional)))
}

fn checked_increment(value: usize, resource: &'static str) -> Result<usize, VisualizeError> {
    value
        .checked_add(1)
        .ok_or_else(|| resource_error(resource, None))
}

fn ensure_running(is_cancelled: &dyn Fn() -> bool) -> Result<(), VisualizeError> {
    if is_cancelled() {
        Err(VisualizeError::Cancelled)
    } else {
        Ok(())
    }
}

fn resource_error(resource: &'static str, requested: Option<usize>) -> VisualizeError {
    VisualizeError::ResourceExhausted {
        resource,
        requested,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use braken::Module;

    use super::*;

    fn nested_generation(depth: usize) -> Generation {
        let mut generation = Generation(vec![GenerationItem::Module(Module::new("F", vec![]))]);
        for _ in 0..depth {
            generation = Generation(vec![GenerationItem::Branch(generation)]);
        }
        generation
    }

    fn request<'a>(
        generation: &'a Generation,
        work_quantum: usize,
        is_cancelled: &'a dyn Fn() -> bool,
    ) -> InspectorStreamRequest<'a> {
        InspectorStreamRequest {
            generation,
            config: InspectorConfig,
            context: VisualizationContext::default(),
            work_quantum,
            is_cancelled,
        }
    }

    #[test]
    fn deeply_nested_branches_do_not_use_the_call_stack() {
        let generation = nested_generation(8_192);
        let never_cancelled = || false;
        let scene = stream_cpu(request(&generation, 64, &never_cancelled), |_| Ok(())).unwrap();
        assert!(scene.primitives.iter().any(|primitive| {
            matches!(primitive, Primitive2d::Text(text) if text.content.contains("Maximum depth: 8192"))
        }));
    }

    #[test]
    fn incremental_inspection_reports_progress_and_completion() {
        let generation = Generation(vec![
            GenerationItem::Module(Module::new("A", vec![])),
            GenerationItem::Module(Module::new("B", vec![])),
            GenerationItem::Module(Module::new("A", vec![])),
        ]);
        let never_cancelled = || false;
        let mut progress = Vec::new();
        stream_cpu(request(&generation, 1, &never_cancelled), |update| {
            progress.push(update);
            Ok(())
        })
        .unwrap();

        assert_eq!(progress.last().unwrap().phase, InspectorPhase::Complete);
        assert_eq!(progress.last().unwrap().items_processed, 3);
        assert!(progress.windows(2).all(|pair| {
            pair[0].items_processed <= pair[1].items_processed
                && pair[0].preview_bytes <= pair[1].preview_bytes
        }));
    }

    #[test]
    fn cancellation_is_checked_inside_large_traversals() {
        let generation = Generation(
            (0..10_000)
                .map(|_| GenerationItem::Module(Module::new("F", vec![])))
                .collect(),
        );
        let checks = Cell::new(0usize);
        let cancel = || {
            checks.set(checks.get() + 1);
            checks.get() > 100
        };
        let error = stream_cpu(request(&generation, 10_000, &cancel), |_| Ok(())).unwrap_err();
        assert_eq!(error, VisualizeError::Cancelled);
        assert!(checks.get() <= 102);
    }

    #[test]
    fn preview_is_bounded_without_recursive_formatting() {
        let generation = nested_generation(8_192);
        let never_cancelled = || false;
        let stats = Stats {
            items: 8_193,
            ..Stats::default()
        };
        let (preview, complete) = bounded_preview(
            &request(&generation, 64, &never_cancelled),
            &stats,
            128,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert!(!complete);
        assert!(preview.len() <= 128);
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn counter_overflow_is_a_typed_resource_error() {
        assert!(matches!(
            checked_increment(usize::MAX, "test counter"),
            Err(VisualizeError::ResourceExhausted {
                resource: "test counter",
                requested: None,
            })
        ));
    }

    #[test]
    fn zero_work_quantum_is_rejected() {
        let generation = Generation::default();
        let never_cancelled = || false;
        assert!(matches!(
            stream_cpu(request(&generation, 0, &never_cancelled), |_| Ok(())),
            Err(VisualizeError::InvalidConfiguration(_))
        ));
    }
}
