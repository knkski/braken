//! Shared GUI boundary for inspecting and importing derivation IR.
//!
//! Both native coordination and the browser Worker call this module. The byte
//! limit bounds display/structured-clone payloads only; it is not a derivation
//! or grammar semantic limit.

use std::error::Error;
use std::fmt;
use std::fmt::Write;

use braken::{
    CompiledGrammar, DerivationSemantics, IrBackendProfile, IrDocument, IrJsonError,
    IrValidationErrors,
};

/// Maximum disassembly or JSON payload retained by GUI tooling.
pub const IR_TOOLING_TEXT_LIMIT: usize = 2 * 1024 * 1024;

/// Bounded text data displayed or exported by a GUI frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrToolingSnapshot {
    pub disassembly: String,
    pub json: Option<String>,
}

/// A validated imported grammar and its bounded inspection data.
#[derive(Debug, Clone)]
pub struct ImportedIr {
    pub grammar: CompiledGrammar,
    pub embedded_source: Option<String>,
    pub tooling: IrToolingSnapshot,
}

#[derive(Debug)]
pub enum IrToolingError {
    Json(IrJsonError),
    Validation(IrValidationErrors),
}

impl fmt::Display for IrToolingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid IR JSON: {error}"),
            Self::Validation(error) => error.fmt(formatter),
        }
    }
}

impl Error for IrToolingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Validation(error) => Some(error),
        }
    }
}

/// Builds the bounded snapshot transferred with a completed GUI render.
pub fn snapshot(grammar: &CompiledGrammar) -> IrToolingSnapshot {
    let mut disassembly = grammar.ir().disassemble();
    truncate_disassembly(&mut disassembly, IR_TOOLING_TEXT_LIMIT);
    let json = grammar
        .to_ir_document()
        .to_json_pretty()
        .ok()
        .filter(|json| json.len() <= IR_TOOLING_TEXT_LIMIT);
    IrToolingSnapshot { disassembly, json }
}

/// Builds a bounded snapshot with compatibility for the exact active semantics.
pub fn snapshot_with_semantics(
    grammar: &CompiledGrammar,
    semantics: DerivationSemantics,
) -> IrToolingSnapshot {
    let mut snapshot = snapshot(grammar);
    let _ = writeln!(
        snapshot.disassembly,
        "\nselected semantics\n  width: {}\n  ambiguous rules: {}",
        semantics.float_width, semantics.ambiguous_rules
    );
    for backend in [
        IrBackendProfile::Cpu,
        IrBackendProfile::CurrentCuda,
        IrBackendProfile::CurrentWgpu,
    ] {
        let report = grammar
            .ir()
            .compatibility_with_semantics(backend, semantics);
        let _ = writeln!(
            snapshot.disassembly,
            "  {backend}: {}",
            if report.compatible {
                "compatible"
            } else {
                "unsupported"
            }
        );
        for diagnostic in report.diagnostics {
            let _ = writeln!(
                snapshot.disassembly,
                "    {:?} [{}] {}",
                diagnostic.severity, diagnostic.code, diagnostic.message
            );
        }
    }
    truncate_disassembly(&mut snapshot.disassembly, IR_TOOLING_TEXT_LIMIT);
    snapshot
}

fn truncate_disassembly(disassembly: &mut String, limit: usize) {
    const NOTICE: &str = "\n\n[IR disassembly truncated by the GUI tooling limit]\n";
    if disassembly.len() <= limit {
        return;
    }
    let notice = if NOTICE.len() <= limit { NOTICE } else { "" };
    let mut boundary = limit.saturating_sub(notice.len());
    while !disassembly.is_char_boundary(boundary) {
        boundary = boundary.saturating_sub(1);
    }
    disassembly.truncate(boundary);
    disassembly.push_str(notice);
}

/// Validates a caller-provided JSON document before it can be activated.
pub fn import_json(json: &str) -> Result<ImportedIr, IrToolingError> {
    let grammar = compile_json(json)?;
    let embedded_source = grammar.to_ir_document().source;
    let tooling = snapshot(&grammar);
    Ok(ImportedIr {
        grammar,
        embedded_source,
        tooling,
    })
}

/// Validates bounded IR JSON without constructing display strings.
pub fn compile_json(json: &str) -> Result<CompiledGrammar, IrToolingError> {
    let document = IrDocument::from_json_with_limit(json, IR_TOOLING_TEXT_LIMIT)
        .map_err(IrToolingError::Json)?;
    CompiledGrammar::from_ir(document).map_err(IrToolingError::Validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use braken::{AmbiguousRulePolicy, FloatWidth};

    #[test]
    fn exported_snapshot_imports_as_the_same_ir() {
        let grammar = CompiledGrammar::parse("axiom A; match A then B;").unwrap();
        let snapshot = snapshot(&grammar);
        let imported = import_json(snapshot.json.as_deref().unwrap()).unwrap();
        assert_eq!(imported.grammar.to_ir_document(), grammar.to_ir_document());
    }

    #[test]
    fn gui_snapshot_and_import_match_the_checked_in_ir_fixture() {
        let source =
            include_str!("../../lib/tests/fixtures/ir/cases/04-boolean-short-circuit.lsys");
        let expected_disassembly =
            include_str!("../../lib/tests/fixtures/ir/expected/04-boolean-short-circuit.ir.txt");
        let expected_json =
            include_str!("../../lib/tests/fixtures/ir/expected/04-boolean-short-circuit.ir.json");

        let grammar = CompiledGrammar::parse(source).unwrap();
        let exported = snapshot(&grammar);
        assert_eq!(exported.disassembly, expected_disassembly);
        assert_eq!(exported.json.as_deref(), Some(expected_json.trim_end()));

        let imported = import_json(expected_json).unwrap();
        assert_eq!(imported.tooling.disassembly, expected_disassembly);
        assert_eq!(imported.grammar.to_ir_document(), grammar.to_ir_document());
    }

    #[test]
    fn oversized_import_is_rejected_before_json_parsing() {
        let json = " ".repeat(IR_TOOLING_TEXT_LIMIT + 1);
        assert!(matches!(
            import_json(&json),
            Err(IrToolingError::Json(IrJsonError::SizeLimitExceeded { .. }))
        ));
    }

    #[test]
    fn semantic_snapshot_exposes_width_scoped_backend_compatibility() {
        let grammar = CompiledGrammar::parse("axiom A; match A then B;").unwrap();
        let snapshot = snapshot_with_semantics(
            &grammar,
            DerivationSemantics {
                float_width: FloatWidth::F32,
                ambiguous_rules: AmbiguousRulePolicy::First,
            },
        );
        assert!(snapshot.disassembly.contains("selected semantics"));
        assert!(snapshot.disassembly.contains("width: f32"));
        assert!(snapshot.disassembly.contains("CUDA: unsupported"));
        assert!(snapshot.disassembly.contains("WGPU: compatible"));
    }

    #[test]
    fn truncated_disassembly_including_notice_stays_within_its_transfer_limit() {
        let mut disassembly = "leaf ❤️ ".repeat(32);
        truncate_disassembly(&mut disassembly, 80);
        assert!(disassembly.len() <= 80);
        assert!(disassembly.ends_with("[IR disassembly truncated by the GUI tooling limit]\n"));
    }
}
