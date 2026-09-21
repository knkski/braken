# Agent guidance

## Scope and precedence

These repository-specific instructions apply throughout the workspace. Follow
the user's request first, preserve unrelated work, and use
[CONTRIBUTING.md](CONTRIBUTING.md) as the canonical source for commands and
feature combinations. [README.md](README.md) describes the public product
surface; API and module rustdoc own exact behavioral contracts.

## Before editing

- Inspect the worktree and do not overwrite changes that are outside the task.
- Identify the processing stage being changed: grammar/derivation,
  visualization, encoding/rendering, or GUI coordination.
- Read the owning module docs and relevant tests before changing an invariant.
- Check whether the code is shared by native and WASM targets or gated behind
  WGPU, CUDA, desktop, web, or Worker features.

## Design guardrails

### Stages and backends

- Keep derivation, visualization, and display backends independent. CUDA or
  WGPU derivation does not imply that visualization or display uses the same
  backend.
- Preserve hardware-first native automatic selection and preflight-only
  fallback. Once backend execution begins, surface its failure instead of
  silently restarting elsewhere. Explicit backend requests never fall back.
- Keep CPU as the full-grammar reference. GPU subsets must reject unsupported
  grammar rather than changing its meaning.
- Seeded results must not depend on CPU worker count, GPU dispatch size, or
  device sharding.

### Resources, progress, and cancellation

- Do not introduce arbitrary default generation ceilings. Caller-selected
  semantic limits, bounded scheduling quanta, and physical device/allocation
  limits are different concepts and must remain distinguishable.
- Use checked size arithmetic, fallible host allocation, device capabilities,
  and typed resource errors. Avoid retry loops that can grow without a bound.
- Cancellation is cooperative at documented safe boundaries. Do not claim that
  already-submitted GPU work can be interrupted.
- Report completed work honestly. Superseded, cancelled, or stale work must not
  become the displayed or cached result.

### GUI responsiveness and rendering

- Never run derivation or large visualization work on the GUI thread. Native
  work uses its coordinator thread; browser work uses the dedicated Web Worker.
- Preserve latest-wins replacement: new requests cancel or drop obsolete work,
  while the last completed scene remains visible. Manual cancellation of
  display refinement remains sticky.
- Keep the shared scene and input architecture across accelerated and software
  renderers. Do not rebuild monolithic indexed geometry for large line scenes.
- Prefer direct CUDA where its native path is supported, then hardware WGPU,
  with bounded CPU/software handling where required. Do not treat a software
  WGPU adapter as hardware acceleration.
- Keep branch-free native CUDA turtle visualization separate from the CPU
  fallback used for structural branches. Other declared visualizer kinds remain
  explicit library stubs until they have real implementations and tests; the
  GUI may present Inspector for those stubs, while the CLI surfaces an explicit
  selection as an error.

### Viewport and controls

- Use `gui/src/camera.rs` as the shared projection and constraint authority for
  every renderer and input path.
- The fitted view is the zoom-out floor. Panning stops at padded drawing edges,
  and pointer or touch anchors remain stable unless an edge constraint wins.
- Camera input is display-only: it must not queue derivation. Reset the camera
  for a successfully replaced system and preserve it for parameter-only changes.
- Preserve desktop wheel/drag behavior. Mobile 2D views leave one finger for
  page scrolling and use two fingers for pan/pinch. Both browser interfaces and
  native spatial views own one-finger taps and rotation drags on the 3D canvas.
- Three-dimensional turtle views are orthographically sphere-fitted and expose
  rotation only: no user zoom or translation. Preset autorotation stops on the
  first owned canvas gesture and restarts only when a 3D preset is selected.
- Resolve three-dimensional display widths, theme-default color, directional
  lighting, and depth cues through the shared spatial contracts so accelerated,
  software, SVG-export, and preset-preview paths do not drift apart. Preserve
  retained source widths and explicit source hues.
- Preserve fractional angle slider values; only the angle minus/plus controls
  snap to adjacent whole degrees. SVG export work remains cancellable.

## Generated files

Do not hand-edit `gui/src/generated_web_presets.rs`,
`gui/src/generated_preset_previews.rs`, or files under
`gui/assets/preset-previews/`. Use the generators and prerequisites documented
in [CONTRIBUTING.md](CONTRIBUTING.md), then run the preset validation suite.
Treat imported fixture attribution as source data that must be preserved.

## Validation routing

Run the commands in [CONTRIBUTING.md](CONTRIBUTING.md) that match the change:

- Shared library or GUI code: native tests and strict Clippy.
- Shared GUI, camera, renderer, or Worker code: native plus WASM checks.
- CUDA code or features: cuda-oxide build/test commands; plain Cargo is not a
  substitute for finalized device artifacts.
- Documentation-only changes: inspect rendered Markdown where possible, verify
  relative links and commands, and run formatting/diff checks.

If required hardware or tooling is unavailable, state exactly which validation
was not run.

## Documentation synchronization

- Public behavior changes update the README summary.
- Exact API behavior changes update rustdoc in the owning source module.
- Build, feature, diagnostic, or generator changes update CONTRIBUTING.
- Agent workflow or non-negotiable design changes update this file.
- Keep volatile constants and implementation timing in their owning modules,
  not in README or AGENTS.
- Prefer links to the authoritative layer over copied prose. Update behavior
  tests alongside invariant changes so documentation is not the only safeguard.
