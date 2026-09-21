# List the available maintenance recipes.
_default:
    @just --list

# Regenerate all checked-in application and preset icons.
icons: app-icons preset-icons

# Regenerate the checked-in application icon assets.
app-icons:
    cargo run -p braken-gui --example generate_app_icons

# Regenerate the GUI preset icons and their Rust manifest.
preset-icons:
    cargo run -p braken-gui --example generate_preset_previews

# Re-import the web corpus fixtures, GUI presets, and Rust manifest.
web-presets:
    cargo run -p braken --example import_web_corpus

# Verify curated derivation-IR goldens and generated review reports.
ir-fixtures-check:
    cargo run -p braken --example generate_ir_fixtures -- --check

# Explicitly regenerate derivation-IR goldens for review.
ir-fixtures-update:
    cargo run -p braken --example generate_ir_fixtures -- --update
