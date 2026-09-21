<p align="center">
  <img src="gui/assets/icons/native/app-icon.svg" width="180" alt="Braken frond logo">
</p>

<h1 align="center">Braken</h1>

<p align="center">
  Explore, animate, and render L-systems in 2D and 3D.
</p>

<p align="center">
  <a href="https://github.com/knkski/braken/actions/workflows/ci.yml"><img src="https://github.com/knkski/braken/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-GPL--3.0--only-4c8f57.svg" alt="GPL-3.0-only license"></a>
</p>

<p align="center">
  <strong><a href="https://knkski.github.io/braken/">Try Braken online</a></strong>
</p>

Braken is a playground and toolkit for Lindenmayer systems: compact rewriting
grammars that grow into curves, plants, tilings, and spatial structures. Start
with a curated preset or write a grammar of your own, then move through its
iterations and inspect the result as it develops.

<table>
  <tr>
    <td align="center" width="50%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="gui/assets/preset-previews/3d-hilbert-curve--effb851c6df36953-dark.svg">
        <img src="gui/assets/preset-previews/3d-hilbert-curve--effb851c6df36953-light.svg" alt="3D Hilbert Curve preset" width="320">
      </picture><br>
      <sub><strong>3D Hilbert Curve</strong></sub>
    </td>
    <td align="center" width="50%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="gui/assets/preset-previews/penrose-tiling--a6f2e5c924905faa-dark.svg">
        <img src="gui/assets/preset-previews/penrose-tiling--a6f2e5c924905faa-light.svg" alt="Penrose Tiling preset" width="320">
      </picture><br>
      <sub><strong>Penrose Tiling</strong></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="gui/assets/preset-previews/stochastic-plant--87517210c25cad65-dark.svg">
        <img src="gui/assets/preset-previews/stochastic-plant--87517210c25cad65-light.svg" alt="Stochastic Plant preset" width="320">
      </picture><br>
      <sub><strong>Stochastic Plant</strong></sub>
    </td>
    <td align="center" width="50%">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="gui/assets/preset-previews/colored-triangle-gasket--2a7a3992a737421a-dark.svg">
        <img src="gui/assets/preset-previews/colored-triangle-gasket--2a7a3992a737421a-light.svg" alt="Colored Triangle Gasket preset" width="320">
      </picture><br>
      <sub><strong>Colored Triangle Gasket</strong></sub>
    </td>
  </tr>
</table>

## Highlights

- Browse a searchable gallery of curated 2D and 3D systems.
- Edit grammars and parameters while keeping the last completed scene visible.
- Watch adjacent iterations transform through their actual rewrite lineage.
- Pan and zoom through planar drawings or rotate spatial models interactively.
- Export finished 2D drawings and the current view of 3D scenes as SVG.
- Run the same grammar engine from the native app, browser, CLI, or Rust API.
- Choose CPU, WGPU, or optional CUDA derivation without coupling it to display.

## Get started

The quickest route is the [hosted browser app](https://knkski.github.io/braken/).
The browser and native interfaces keep the preset gallery separate from grammar
editing and adjustments, with desktop controls and status below the drawing.
The desktop grammar editor and Advanced settings open above the bottom bar. On
phones, change iterations with the slider or minus/plus buttons, and drag 3D
models with one finger to rotate them; a tap stops preset autorotation. The
mobile editor keeps a draft: Apply updates the scene, while Back preserves the
draft for later. Floating-point precision and ambiguous-rule policy are
available under Advanced. Iced also keeps its IR inspection and import/export
tools in the editor's IR tab.

To run the native application from source:

```bash
git clone https://github.com/knkski/braken.git
cd braken
cargo run -p braken-gui
```

Install the command-line renderer directly from the checkout:

```bash
cargo install --path cli
braken --file gui/presets/koch-snowflake.lsys \
  --iterations 4 --target svg --output snowflake.svg
```

Braken is currently distributed from source; its Rust packages are not yet
published on crates.io. The pinned toolchain in `rust-toolchain.toml` includes
the components and WASM target used by the workspace.

## A small L-system

This grammar starts with a triangle and replaces every edge with a Koch curve:

```text
# Visualizer: turtle_2d
# Angle: 60

axiom Draw Right Right Draw Right Right Draw;

match Draw then Draw Left Draw Right Right Draw Left Draw;
```

Save it as `snowflake.lsys`, then render it with:

```bash
braken --file snowflake.lsys --iterations 4 \
  --target svg --output snowflake.svg
```

Grammars can also use parameters, expressions, context-sensitive productions,
weighted alternatives, structural branches, color, and three-dimensional
turtle commands. Runnable Rust examples live in [`lib/examples`](lib/examples).

## Components

| Package | What it provides |
| --- | --- |
| `braken` | Grammar parser, derivation engine, and CPU/WGPU/optional CUDA backends |
| `braken-viz` | Inspector, retained 2D and 3D turtle scenes, and output targets |
| `braken-cli` | The `braken` command for deriving and rendering grammars |
| `braken-gui` | Native Iced app, Iced browser build, and shared browser Worker |
| `braken-web` | Alternative Yew browser frontend using the same Worker and scenes |

## Architecture

Braken keeps derivation, visualization, and display independent, so accelerated
rewriting never dictates how a scene is interpreted or rendered. CPU is the
full-grammar reference; GPU backends explicitly reject grammar features they
cannot preserve instead of changing their meaning. Seeded derivations stay
deterministic across worker counts and GPU dispatch layouts. The native
coordinator and browser Worker run expensive work away from the interface and
use latest-wins replacement, leaving the last finished scene visible while
stale work is cancelled or discarded.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
feature combinations, validation commands, generated assets, and optional CUDA
development.

Braken is available under the [GNU General Public License v3.0](LICENSE.md).
