# Contributing to Braken

Thanks for helping improve Braken. This guide covers the workspace, useful
development commands, and the generated files that need special handling.

## Prerequisites

The repository pins Rust, its components, and the WASM target in
`rust-toolchain.toml`. If you use Nix, `nix develop` opens a shell with the
project tools. Otherwise, install the extra tools needed for the area you are
working on:

- [Just](https://github.com/casey/just) runs maintenance and generation recipes.
- [Trunk](https://trunkrs.dev/) serves and bundles either browser frontend.
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide) builds the optional CUDA
  host and device artifacts. Runtime testing also needs a compatible NVIDIA
  toolkit, driver, and device.

## Workspace and features

| Package | Directory | Purpose |
| --- | --- | --- |
| `braken` | `lib/` | Grammar parsing and derivation; WGPU is enabled by default |
| `braken-viz` | `viz/` | Target-neutral inspector and 2D/3D turtle visualization |
| `braken-cli` | `cli/` | Command-line derivation and TUI/SVG output |
| `braken-gui` | `gui/` | Native and Iced browser applications plus the shared Worker |
| `braken-web` | `web/` | Alternative Yew, HTML, CSS, and Canvas2D browser frontend |

The core `braken` package supports CPU-only builds with
`--no-default-features`, WGPU with the default `wgpu` feature, and native CUDA
with `cuda`. `braken-viz` has an independent `cuda` feature for supported
turtle visualization. `braken-gui` uses `desktop` by default; its other feature
sets are `web`, `web-worker`, and `cuda`.

CUDA derivation, visualization, and GUI display are separate stages. Enabling
CUDA does not require every stage to use it. CPU implements the full grammar;
accelerated backends reject unsupported input rather than changing its meaning.
Browser calculation and visualization run in the dedicated Worker because the
synchronous API cannot await WebGPU and expensive work must stay off the UI
thread.

## Run the applications

Native GUI:

```bash
cargo run -p braken-gui
```

Native GUI with CUDA:

```bash
cargo oxide run --package braken-gui --bin braken-gui \
  --features braken-gui/cuda
```

CLI:

```bash
cargo run -p braken-cli -- \
  --file gui/presets/koch-snowflake.lsys --iterations 4
```

CLI with an explicit CUDA derivation backend:

```bash
cargo oxide run --package braken-cli --bin braken \
  --features braken-cli/cuda -- \
  --backend cuda --file path/to/system.lsys --iterations 8
```

The derivation backend is selected with `--backend auto|cpu|wgpu|cuda`.
Visualization is selected independently with
`--visualizer-backend auto|cpu|cuda`, and `--target auto|tui|svg` chooses the
output.

Iced browser GUI:

```bash
cd gui
trunk serve --release --verbose index.html
```

Yew browser GUI:

```bash
cd web
trunk serve --release --verbose index.html
```

Both browser frontends build the shared `braken-render-worker`; it is not a
standalone native application. GitHub Pages publishes the Yew frontend at the
site root and the Iced frontend under `iced/`.

## Validation

Run the checks relevant to your change. A broad workspace change should pass
the full applicable set.

Baseline native checks:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
git diff --check
```

Native GUI checks:

```bash
cargo test -p braken-gui
cargo clippy -p braken-gui --all-targets -- -D warnings
```

For responsive Iced layout changes, also smoke-test desktop/mobile resizing,
panel toggles, and scrolling with both WGPU and the software display renderer:

```bash
ICED_BACKEND=tiny-skia cargo run -p braken-gui
```

This override selects the display renderer, not the derivation backend. Canvas
clipping compatibility is documented in [canvas_clip.rs](gui/src/canvas_clip.rs).

Iced browser checks:

```bash
cargo check -p braken-gui \
  --target wasm32-unknown-unknown \
  --no-default-features --features web --bin braken-gui
cargo clippy -p braken-gui \
  --target wasm32-unknown-unknown \
  --no-default-features --features web --bin braken-gui -- -D warnings
(cd gui && trunk build index.html --release)
```

Shared browser Worker checks:

```bash
cargo check -p braken-gui \
  --target wasm32-unknown-unknown \
  --no-default-features --features web-worker \
  --bin braken-render-worker
cargo clippy -p braken-gui \
  --target wasm32-unknown-unknown \
  --no-default-features --features web-worker \
  --bin braken-render-worker -- -D warnings
```

Yew browser checks:

```bash
cargo check -p braken-web \
  --target wasm32-unknown-unknown --all-targets
cargo clippy -p braken-web \
  --target wasm32-unknown-unknown --all-targets -- -D warnings
(cd web && trunk build index.html --release)
```

Once dependencies are cached, `--offline` can be added to Cargo browser checks
for a reproducible local run.

### CUDA

Plain Cargo can type-check CUDA host code without producing finalized device
artifacts. Use cuda-oxide for acceptance:

```bash
cargo oxide build --features cuda
cargo oxide test -- -p braken --features cuda
cargo oxide test -- -p braken-viz --features cuda
```

If CUDA hardware is unavailable, say which runtime checks were not run. The
visualization tests skip device assertions when no CUDA context can be created,
so a passing test count by itself confirms compilation rather than runtime
parity.

## Derivation IR fixtures

Focused source grammars, versioned semantic JSON, deterministic disassembly,
and generated review reports live under `lib/tests/fixtures/ir/`. The
[gallery](lib/tests/fixtures/ir/GALLERY.md) presents each case beside its
disassembly, while the [coverage report](lib/tests/fixtures/ir/COVERAGE.md)
summarizes the features exercised by the full fixture corpus.

Check the fixtures after changing the shared derivation IR:

```bash
just ir-fixtures-check
cargo test -p braken --test ir_fixtures --no-default-features
```

Regenerate them only for an intentional format or semantic change:

```bash
just ir-fixtures-update
just ir-fixtures-check
```

Review source, JSON, disassembly, gallery, and coverage changes together. To
inspect one case without writing files:

```bash
cargo run -p braken --example generate_ir_fixtures -- \
  --case 04-boolean-short-circuit
```

Add focused fixtures to `manifest.json` and `cases/`; keep expected executions
as independently reviewed inputs rather than deriving them from generated JSON.

## Generated presets and artwork

Do not hand-edit `gui/src/generated_web_presets.rs`,
`gui/src/generated_preset_previews.rs`, files under
`gui/assets/preset-previews/`, or files under `gui/assets/icons/`.

Re-import the curated external preset corpus with:

```bash
just web-presets
```

The importer expects its source corpus at
`scratch/well-known-lsystems-corpus/lsystem-corpus`. It preserves attribution,
filters unsupported or resource-dependent sources, and emits the checked-in
preset manifest. Run the GUI preset tests after regeneration.

Generate every light and dark preset preview from the current catalog with:

```bash
just preset-icons
```

The generator uses seed `0` and the shared rendering contracts. Check that the
committed previews are current without rewriting them:

```bash
cargo run -p braken-gui --example generate_preset_previews -- --check
```

The Braken preset, application icon, and README logo share the canonical source
`gui/branding/app-icon.lsys`. Regenerate the browser, native, and packaging
assets with:

```bash
just app-icons
```

Check those outputs without rewriting them with:

```bash
cargo run -p braken-gui --example generate_app_icons -- --check
```

When the imported catalog, previews, and icons all change, regenerate them in
that order: `just web-presets`, `just preset-icons`, then `just app-icons`.

## Pull requests

Before opening a pull request:

- Keep the change focused and preserve unrelated work.
- Add or update tests for observable behavior and error cases.
- Update public documentation when capabilities or commands change.
- Regenerate owned files with their recipes and review the resulting diffs.
- Run the applicable checks above and list any hardware-dependent checks you
  could not run.
