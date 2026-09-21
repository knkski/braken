//! Backend-neutral compiled grammar representation and inspection tooling.
//!
//! [`IrDocument`] is the versioned interchange form. It is intentionally
//! separate from private CUDA and WGPU buffer layouts: imported documents are
//! validated before they can become a [`DerivationIr`]. [`IrView`] provides
//! read-only access to that validated representation, deterministic
//! disassembly, and static backend-compatibility diagnostics.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Write as _};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::grammar::{
    BinaryOp, Binding, ContextFilter, Document, Expr, Grammar, Identifier, Item, ModuleExpr,
    ModulePattern, PatternArgument, PatternItem, PatternWord, Production, UnaryOp, Word, WordItem,
};

/// Current JSON/interchange container version.
pub const IR_FORMAT_VERSION: u32 = 1;
/// Current derivation-language semantic version.
pub const IR_SEMANTICS_VERSION: u32 = 2;
/// Format discriminator used by JSON interchange documents.
pub const IR_FORMAT_NAME: &str = "braken-ir";

macro_rules! id_type {
    ($name:ident, $label:literal) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, concat!($label, "{}"), self.0)
            }
        }
    };
}

id_type!(SymbolId, "s");
id_type!(GlobalId, "g");
id_type!(ProductionId, "p");
id_type!(BindingSlot, "$");
id_type!(ProgramId, "e");

/// Owned, versioned grammar IR suitable for JSON interchange.
///
/// Its fields are public so external tools can transform documents. Such a
/// document is data, not executable input, until [`Self::validate`] succeeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrDocument {
    pub format: String,
    pub format_version: u32,
    pub semantics_version: u32,
    pub source: Option<String>,
    pub source_fingerprint: Option<String>,
    pub symbols: Vec<IrSymbol>,
    pub globals: Vec<IrGlobal>,
    pub context_filter: Option<IrContextFilter>,
    pub axiom: IrWord,
    pub productions: Vec<IrProduction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrSymbol {
    pub name: String,
    pub arity: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrGlobal {
    pub name: String,
    pub value: IrExpr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrContextFilter {
    Ignore(Vec<String>),
    Only(Vec<String>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrWord {
    pub items: Vec<IrWordItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrWordItem {
    Module(IrModuleExpr),
    Branch(IrWord),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrModuleExpr {
    pub symbol: SymbolId,
    pub arguments: Vec<IrExpr>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrPatternWord {
    pub items: Vec<IrPatternItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrPatternItem {
    Module(IrModulePattern),
    Branch(IrPatternWord),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrModulePattern {
    pub symbol: SymbolId,
    pub arguments: Vec<IrPatternArgument>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrPatternArgument {
    Bind(BindingSlot),
    Wildcard,
    Literal(#[serde(with = "hex_u64")] u64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrProduction {
    pub bindings: Vec<String>,
    pub center: IrModulePattern,
    pub left: Option<IrPatternWord>,
    pub right: Option<IrPatternWord>,
    pub condition: Option<IrExpr>,
    pub weight: Option<IrExpr>,
    pub successor: IrWord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrExpr {
    Number(#[serde(with = "hex_u64")] u64),
    Bool(bool),
    Global(GlobalId),
    Binding(BindingSlot),
    Call {
        name: String,
        arguments: Vec<IrExpr>,
    },
    Unary {
        op: IrUnaryOp,
        operand: Box<IrExpr>,
    },
    Binary {
        op: IrBinaryOp,
        left: Box<IrExpr>,
        right: Box<IrExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrUnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrBinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}

mod hex_u64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("0x{value:016x}"))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        let digits = encoded.strip_prefix("0x").ok_or_else(|| {
            serde::de::Error::custom("binary64 bits must be a 0x-prefixed hexadecimal string")
        })?;
        if digits.len() != 16 {
            return Err(serde::de::Error::custom(
                "binary64 bits must contain exactly 16 hexadecimal digits",
            ));
        }
        u64::from_str_radix(digits, 16).map_err(serde::de::Error::custom)
    }
}

/// A validated IR plus derived bytecode and capability metadata.
#[derive(Debug, Clone)]
pub struct DerivationIr(Arc<ValidatedIr>);

#[derive(Debug)]
struct ValidatedIr {
    document: IrDocument,
    programs: Vec<IrProgram>,
    production_groups: Vec<IrProductionGroup>,
    requirements: IrRequirements,
}

/// Borrowed, read-only access to a [`DerivationIr`].
#[derive(Debug, Clone, Copy)]
pub struct IrView<'a>(&'a ValidatedIr);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrProgram {
    pub id: ProgramId,
    pub owner: IrProgramOwner,
    pub instructions: Vec<IrInstruction>,
    pub max_stack: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrProductionGroup {
    pub symbol: SymbolId,
    pub productions: Vec<ProductionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IrProgramOwner {
    Global(GlobalId),
    AxiomArgument {
        module: u32,
        argument: u32,
    },
    Condition(ProductionId),
    Weight(ProductionId),
    SuccessorArgument {
        production: ProductionId,
        module: u32,
        argument: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrInstruction {
    PushNumber(u64),
    PushBool(bool),
    LoadGlobal(GlobalId),
    LoadBinding(BindingSlot),
    Call { name: String, arity: u32 },
    Unary(IrUnaryOp),
    Binary(IrBinaryOp),
    JumpIfFalse(u32),
    JumpIfTrue(u32),
    Pop,
    Return,
}

/// Static features and bounded scratch requirements found in an IR document.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IrRequirements {
    pub symbol_count: u32,
    pub global_count: u32,
    pub production_count: u32,
    pub parameterized_modules: bool,
    pub contextual_productions: bool,
    pub structural_contexts: bool,
    pub context_filter: bool,
    pub dynamic_conditions: bool,
    pub dynamic_weights: bool,
    pub weighted_productions: bool,
    pub structural_branches: bool,
    pub maximum_bindings: u32,
    pub maximum_expression_stack: u32,
    pub maximum_context_depth: u32,
    pub builtins: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBackendProfile {
    Cpu,
    CurrentCuda,
    CurrentWgpu,
}

impl fmt::Display for IrBackendProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "CPU",
            Self::CurrentCuda => "CUDA",
            Self::CurrentWgpu => "WGPU",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrDiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrDiagnostic {
    pub severity: IrDiagnosticSeverity,
    pub code: &'static str,
    pub message: String,
    pub entity: Option<String>,
    pub backend: Option<IrBackendProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrCompatibilityReport {
    pub backend: IrBackendProfile,
    pub compatible: bool,
    pub diagnostics: Vec<IrDiagnostic>,
}

/// Exact result of one CPU-reference rewrite initiated from IR tooling.
#[derive(Debug, Clone, PartialEq)]
pub struct IrRewriteTrace {
    pub generation: crate::grammar::Generation,
    pub stats: crate::grammar::StepStats,
    pub lineage: crate::grammar::RewriteLineage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrValidationErrors(pub Vec<IrDiagnostic>);

#[derive(Debug)]
pub enum IrJsonError {
    SizeLimitExceeded { actual: usize, limit: usize },
    InvalidJson(serde_json::Error),
}

impl fmt::Display for IrJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeLimitExceeded { actual, limit } => {
                write!(
                    formatter,
                    "IR JSON size {actual} exceeds configured limit {limit}"
                )
            }
            Self::InvalidJson(error) => error.fmt(formatter),
        }
    }
}

impl Error for IrJsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidJson(error) => Some(error),
            Self::SizeLimitExceeded { .. } => None,
        }
    }
}

impl fmt::Display for IrValidationErrors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "IR validation failed with {} error(s):",
            self.0.len()
        )?;
        for diagnostic in &self.0 {
            writeln!(formatter, "- [{}] {}", diagnostic.code, diagnostic.message)?;
        }
        Ok(())
    }
}

impl Error for IrValidationErrors {}

/// Incremental constructor for an [`IrDocument`]. Validation remains mandatory.
#[derive(Debug, Clone)]
pub struct IrBuilder {
    document: IrDocument,
}

impl IrBuilder {
    pub fn new() -> Self {
        Self {
            document: IrDocument {
                format: String::from(IR_FORMAT_NAME),
                format_version: IR_FORMAT_VERSION,
                semantics_version: IR_SEMANTICS_VERSION,
                source: None,
                source_fingerprint: None,
                symbols: Vec::new(),
                globals: Vec::new(),
                context_filter: None,
                axiom: IrWord::default(),
                productions: Vec::new(),
            },
        }
    }

    pub fn source(&mut self, source: impl Into<String>) -> &mut Self {
        let source = source.into();
        self.document.source_fingerprint = Some(source_fingerprint(&source));
        self.document.source = Some(source);
        self
    }

    pub fn intern_symbol(&mut self, name: impl Into<String>, arity: u32) -> SymbolId {
        let name = name.into();
        if let Some(index) = self
            .document
            .symbols
            .iter()
            .position(|symbol| symbol.name == name && symbol.arity == arity)
        {
            return SymbolId(u32::try_from(index).unwrap_or(u32::MAX));
        }
        let id = SymbolId(u32::try_from(self.document.symbols.len()).unwrap_or(u32::MAX));
        self.document.symbols.push(IrSymbol { name, arity });
        id
    }

    pub fn push_global(&mut self, name: impl Into<String>, value: IrExpr) -> GlobalId {
        let id = GlobalId(u32::try_from(self.document.globals.len()).unwrap_or(u32::MAX));
        self.document.globals.push(IrGlobal {
            name: name.into(),
            value,
        });
        id
    }

    pub fn context_filter(&mut self, filter: Option<IrContextFilter>) -> &mut Self {
        self.document.context_filter = filter;
        self
    }

    pub fn axiom(&mut self, axiom: IrWord) -> &mut Self {
        self.document.axiom = axiom;
        self
    }

    pub fn push_production(&mut self, production: IrProduction) -> ProductionId {
        let id = ProductionId(u32::try_from(self.document.productions.len()).unwrap_or(u32::MAX));
        self.document.productions.push(production);
        id
    }

    pub fn document(self) -> IrDocument {
        self.document
    }

    pub fn finish(self) -> Result<DerivationIr, IrValidationErrors> {
        self.document.validate()
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IrDocument {
    /// Compiles a semantically validated grammar into its interchange form.
    pub fn from_grammar(grammar: &Grammar) -> Self {
        compile_grammar(grammar, None)
    }

    /// Deserializes untrusted JSON. Call [`Self::validate`] before use.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Deserializes JSON only when the caller-selected byte limit permits it.
    pub fn from_json_with_limit(json: &str, limit: usize) -> Result<Self, IrJsonError> {
        if json.len() > limit {
            return Err(IrJsonError::SizeLimitExceeded {
                actual: json.len(),
                limit,
            });
        }
        Self::from_json(json).map_err(IrJsonError::InvalidJson)
    }

    /// Serializes exact numeric bits and stable source order as formatted JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Checks all IDs and invariants and derives executable expression bytecode.
    pub fn validate(self) -> Result<DerivationIr, IrValidationErrors> {
        let diagnostics = validate_document(&self);
        if diagnostics.is_empty() {
            let programs = compile_programs(&self);
            let production_groups = collect_production_groups(&self);
            let requirements = collect_requirements(&self, &programs);
            let ir = DerivationIr(Arc::new(ValidatedIr {
                document: self,
                programs,
                production_groups,
                requirements,
            }));
            // Reuse the language validator as a second, semantic gate. The IR
            // checks above make raising IDs safe; this pass preserves source
            // grammar rules such as positive literal weights.
            ir.to_grammar()?;
            Ok(ir)
        } else {
            Err(IrValidationErrors(diagnostics))
        }
    }
}

impl DerivationIr {
    pub(crate) fn from_grammar_source(grammar: &Grammar, source: Option<&str>) -> Self {
        compile_grammar(grammar, source)
            .validate()
            .expect("validated grammar must compile to valid derivation IR")
    }

    pub fn view(&self) -> IrView<'_> {
        IrView(&self.0)
    }

    pub fn document(&self) -> &IrDocument {
        &self.0.document
    }

    pub fn into_document(self) -> IrDocument {
        self.0.document.clone()
    }

    pub fn disassemble(&self) -> String {
        self.view().disassemble()
    }

    pub fn compatibility(&self, backend: IrBackendProfile) -> IrCompatibilityReport {
        self.view().compatibility(backend)
    }

    /// Reports compatibility for an exact numeric-width and selection profile.
    pub fn compatibility_with_semantics(
        &self,
        backend: IrBackendProfile,
        semantics: crate::execution::DerivationSemantics,
    ) -> IrCompatibilityReport {
        self.view().compatibility_with_semantics(backend, semantics)
    }

    /// Rewrites an externally retained generation once with CPU-reference
    /// semantics and returns exact output lineage.
    pub fn trace_rewrite(
        &self,
        generation: &crate::grammar::Generation,
        generation_index: u64,
        seed: u64,
        ambiguous_rules: crate::grammar::AmbiguousRulePolicy,
    ) -> Result<IrRewriteTrace, crate::grammar::ExecutionError> {
        self.trace_rewrite_with_semantics(
            generation,
            generation_index,
            seed,
            crate::execution::DerivationSemantics {
                float_width: crate::grammar::FloatWidth::F64,
                ambiguous_rules,
            },
        )
    }

    /// Rewrites an externally retained generation once with the selected
    /// numeric-width and ambiguity semantics and returns exact output lineage.
    pub fn trace_rewrite_with_semantics(
        &self,
        generation: &crate::grammar::Generation,
        generation_index: u64,
        seed: u64,
        semantics: crate::execution::DerivationSemantics,
    ) -> Result<IrRewriteTrace, crate::grammar::ExecutionError> {
        self.trace_rewrite_with_semantics_and_control(
            generation,
            generation_index,
            seed,
            semantics,
            crate::grammar::ExecutionLimits::default(),
            || false,
            |_| {},
        )
    }

    /// Cancellable, limit-aware form of [`Self::trace_rewrite`].
    #[allow(clippy::too_many_arguments)]
    pub fn trace_rewrite_with_control(
        &self,
        generation: &crate::grammar::Generation,
        generation_index: u64,
        seed: u64,
        ambiguous_rules: crate::grammar::AmbiguousRulePolicy,
        limits: crate::grammar::ExecutionLimits,
        is_cancelled: impl FnMut() -> bool,
        on_progress: impl FnMut(crate::grammar::CpuStepProgress),
    ) -> Result<IrRewriteTrace, crate::grammar::ExecutionError> {
        self.trace_rewrite_with_semantics_and_control(
            generation,
            generation_index,
            seed,
            crate::execution::DerivationSemantics {
                float_width: crate::grammar::FloatWidth::F64,
                ambiguous_rules,
            },
            limits,
            is_cancelled,
            on_progress,
        )
    }

    /// Cancellable, limit-aware form of
    /// [`Self::trace_rewrite_with_semantics`].
    #[allow(clippy::too_many_arguments)]
    pub fn trace_rewrite_with_semantics_and_control(
        &self,
        generation: &crate::grammar::Generation,
        generation_index: u64,
        seed: u64,
        semantics: crate::execution::DerivationSemantics,
        limits: crate::grammar::ExecutionLimits,
        is_cancelled: impl FnMut() -> bool,
        on_progress: impl FnMut(crate::grammar::CpuStepProgress),
    ) -> Result<IrRewriteTrace, crate::grammar::ExecutionError> {
        let program = crate::grammar::CpuBackend::new()
            .float_width(semantics.float_width)
            .ambiguous_rules(semantics.ambiguous_rules)
            .limits(limits)
            .compile_ir(self)?;
        let (generation, stats, lineage) = program.trace_rewrite_with_control(
            generation,
            generation_index,
            seed,
            is_cancelled,
            on_progress,
        )?;
        Ok(IrRewriteTrace {
            generation,
            stats,
            lineage,
        })
    }
}

impl<'a> IrView<'a> {
    pub fn document(self) -> &'a IrDocument {
        &self.0.document
    }

    pub fn programs(self) -> &'a [IrProgram] {
        &self.0.programs
    }

    pub fn production_groups(self) -> &'a [IrProductionGroup] {
        &self.0.production_groups
    }

    pub fn symbol(self, id: SymbolId) -> Option<&'a IrSymbol> {
        usize::try_from(id.0)
            .ok()
            .and_then(|index| self.0.document.symbols.get(index))
    }

    pub fn global(self, id: GlobalId) -> Option<&'a IrGlobal> {
        usize::try_from(id.0)
            .ok()
            .and_then(|index| self.0.document.globals.get(index))
    }

    pub fn production(self, id: ProductionId) -> Option<&'a IrProduction> {
        usize::try_from(id.0)
            .ok()
            .and_then(|index| self.0.document.productions.get(index))
    }

    pub fn program(self, id: ProgramId) -> Option<&'a IrProgram> {
        usize::try_from(id.0)
            .ok()
            .and_then(|index| self.0.programs.get(index))
    }

    /// Executes one expression program with CPU-reference numeric semantics.
    pub fn evaluate_program(
        self,
        id: ProgramId,
        globals: &[crate::grammar::Value],
        bindings: &[crate::grammar::Value],
    ) -> Result<crate::grammar::Value, crate::grammar::ExecutionError> {
        self.evaluate_program_with_width(id, globals, bindings, crate::grammar::FloatWidth::F64)
    }

    /// Executes one expression program using the selected numeric width.
    pub fn evaluate_program_with_width(
        self,
        id: ProgramId,
        globals: &[crate::grammar::Value],
        bindings: &[crate::grammar::Value],
        float_width: crate::grammar::FloatWidth,
    ) -> Result<crate::grammar::Value, crate::grammar::ExecutionError> {
        let program = self.program(id).ok_or_else(|| {
            crate::grammar::ExecutionError::new(format!("unknown IR expression program {id}"))
        })?;
        evaluate_program(program, globals, bindings, float_width)
    }

    /// Evaluates global bindings in dependency order with CPU-reference semantics.
    pub fn evaluate_globals(
        self,
    ) -> Result<Vec<crate::grammar::Value>, crate::grammar::ExecutionError> {
        self.evaluate_globals_with_width(crate::grammar::FloatWidth::F64)
    }

    /// Evaluates global bindings in dependency order using the selected numeric width.
    pub fn evaluate_globals_with_width(
        self,
        float_width: crate::grammar::FloatWidth,
    ) -> Result<Vec<crate::grammar::Value>, crate::grammar::ExecutionError> {
        use crate::grammar::{ExecutionError, Value};

        let count = self.0.document.globals.len();
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|error| ExecutionError::resource("IR global values", error))?;
        values.resize(count, Value::Number(0.0));
        let mut resolved = Vec::new();
        resolved
            .try_reserve_exact(count)
            .map_err(|error| ExecutionError::resource("IR global resolution flags", error))?;
        resolved.resize(count, false);
        let mut remaining = count;
        while remaining > 0 {
            let mut made_progress = false;
            for index in 0..count {
                if resolved[index] {
                    continue;
                }
                let program = &self.0.programs[index];
                let dependencies_ready = program.instructions.iter().all(|instruction| {
                    let IrInstruction::LoadGlobal(id) = instruction else {
                        return true;
                    };
                    usize::try_from(id.0)
                        .ok()
                        .and_then(|dependency| resolved.get(dependency))
                        .is_some_and(|resolved| *resolved)
                });
                if !dependencies_ready {
                    continue;
                }
                values[index] = evaluate_program(program, &values, &[], float_width)?;
                resolved[index] = true;
                remaining -= 1;
                made_progress = true;
            }
            if !made_progress {
                let names = resolved
                    .iter()
                    .enumerate()
                    .filter(|(_, resolved)| !**resolved)
                    .map(|(index, _)| self.0.document.globals[index].name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(ExecutionError::new(format!(
                    "cyclic or unresolved IR global bindings: {names}"
                )));
            }
        }
        Ok(values)
    }

    pub fn requirements(self) -> &'a IrRequirements {
        &self.0.requirements
    }

    pub fn disassemble(self) -> String {
        disassemble(self.0)
    }

    /// Reports grammar-feature compatibility using each backend's native width.
    ///
    /// CPU and CUDA use f64; WGPU uses f32. Use
    /// [`Self::compatibility_with_semantics`] when comparing backends within one
    /// exact semantic profile.
    pub fn compatibility(self, backend: IrBackendProfile) -> IrCompatibilityReport {
        compatibility(&self.0.requirements, backend)
    }

    /// Reports compatibility for an exact numeric-width and selection profile.
    pub fn compatibility_with_semantics(
        self,
        backend: IrBackendProfile,
        semantics: crate::execution::DerivationSemantics,
    ) -> IrCompatibilityReport {
        compatibility_with_semantics(&self.0.requirements, backend, semantics)
    }
}

fn evaluate_program(
    program: &IrProgram,
    globals: &[crate::grammar::Value],
    bindings: &[crate::grammar::Value],
    float_width: crate::grammar::FloatWidth,
) -> Result<crate::grammar::Value, crate::grammar::ExecutionError> {
    use crate::grammar::{ExecutionError, Value};

    let capacity = usize::try_from(program.max_stack)
        .map_err(|error| ExecutionError::resource("IR expression stack", error))?;
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(capacity)
        .map_err(|error| ExecutionError::resource("IR expression stack", error))?;
    let mut instruction_pointer = 0_usize;
    loop {
        let instruction = program
            .instructions
            .get(instruction_pointer)
            .ok_or_else(|| {
                ExecutionError::new(format!(
                    "IR expression {} reached the end without returning",
                    program.id
                ))
            })?;
        match instruction {
            IrInstruction::PushNumber(bits) => stack.push(Value::Number(match float_width {
                crate::grammar::FloatWidth::F32 => f64::from(f64::from_bits(*bits) as f32),
                crate::grammar::FloatWidth::F64 => f64::from_bits(*bits),
            })),
            IrInstruction::PushBool(value) => stack.push(Value::Bool(*value)),
            IrInstruction::LoadGlobal(id) => stack.push(
                *usize::try_from(id.0)
                    .ok()
                    .and_then(|index| globals.get(index))
                    .ok_or_else(|| ExecutionError::new(format!("unavailable global {id}")))?,
            ),
            IrInstruction::LoadBinding(slot) => stack.push(
                *usize::try_from(slot.0)
                    .ok()
                    .and_then(|index| bindings.get(index))
                    .ok_or_else(|| ExecutionError::new(format!("unavailable binding {slot}")))?,
            ),
            IrInstruction::Call { name, arity } => {
                let arity = usize::try_from(*arity)
                    .map_err(|error| ExecutionError::new(error.to_string()))?;
                let start = stack.len().checked_sub(arity).ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its stack before calling `{name}`",
                        program.id
                    ))
                })?;
                let value = crate::grammar::evaluate_builtin_with_width(
                    &Identifier::new(name.clone()),
                    &stack[start..],
                    float_width,
                )?;
                stack.truncate(start);
                stack.push(value);
            }
            IrInstruction::Unary(op) => {
                let operand = stack.pop().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its unary operand",
                        program.id
                    ))
                })?;
                stack.push(crate::grammar::evaluate_unary_with_width(
                    raise_unary_op(*op),
                    operand,
                    float_width,
                )?);
            }
            IrInstruction::Binary(op) => {
                let right = stack.pop().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its right operand",
                        program.id
                    ))
                })?;
                let left = stack.pop().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its left operand",
                        program.id
                    ))
                })?;
                stack.push(crate::grammar::evaluate_binary_with_width(
                    raise_binary_op(*op),
                    left,
                    right,
                    float_width,
                )?);
            }
            IrInstruction::JumpIfFalse(target) => {
                let condition = stack.last().copied().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its branch condition",
                        program.id
                    ))
                })?;
                if !condition.as_bool()? {
                    instruction_pointer = checked_jump_target(program, *target)?;
                    continue;
                }
            }
            IrInstruction::JumpIfTrue(target) => {
                let condition = stack.last().copied().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its branch condition",
                        program.id
                    ))
                })?;
                if condition.as_bool()? {
                    instruction_pointer = checked_jump_target(program, *target)?;
                    continue;
                }
            }
            IrInstruction::Pop => {
                stack.pop().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} exhausted its stack before pop",
                        program.id
                    ))
                })?;
            }
            IrInstruction::Return => {
                let result = stack.pop().ok_or_else(|| {
                    ExecutionError::new(format!(
                        "IR expression {} returned without a value",
                        program.id
                    ))
                })?;
                if !stack.is_empty() {
                    return Err(ExecutionError::new(format!(
                        "IR expression {} returned with {} extra stack value(s)",
                        program.id,
                        stack.len()
                    )));
                }
                return Ok(result);
            }
        }
        instruction_pointer = instruction_pointer.saturating_add(1);
    }
}

fn checked_jump_target(
    program: &IrProgram,
    target: u32,
) -> Result<usize, crate::grammar::ExecutionError> {
    let target = usize::try_from(target)
        .map_err(|error| crate::grammar::ExecutionError::new(error.to_string()))?;
    if target >= program.instructions.len() {
        Err(crate::grammar::ExecutionError::new(format!(
            "IR expression {} has out-of-range jump target {target}",
            program.id
        )))
    } else {
        Ok(target)
    }
}

fn raise_unary_op(op: IrUnaryOp) -> UnaryOp {
    match op {
        IrUnaryOp::Negate => UnaryOp::Negate,
        IrUnaryOp::Not => UnaryOp::Not,
    }
}

fn raise_binary_op(op: IrBinaryOp) -> BinaryOp {
    match op {
        IrBinaryOp::Add => BinaryOp::Add,
        IrBinaryOp::Subtract => BinaryOp::Subtract,
        IrBinaryOp::Multiply => BinaryOp::Multiply,
        IrBinaryOp::Divide => BinaryOp::Divide,
        IrBinaryOp::Power => BinaryOp::Power,
        IrBinaryOp::Equal => BinaryOp::Equal,
        IrBinaryOp::NotEqual => BinaryOp::NotEqual,
        IrBinaryOp::Less => BinaryOp::Less,
        IrBinaryOp::LessEqual => BinaryOp::LessEqual,
        IrBinaryOp::Greater => BinaryOp::Greater,
        IrBinaryOp::GreaterEqual => BinaryOp::GreaterEqual,
        IrBinaryOp::And => BinaryOp::And,
        IrBinaryOp::Or => BinaryOp::Or,
    }
}

fn collect_production_groups(document: &IrDocument) -> Vec<IrProductionGroup> {
    let mut groups = BTreeMap::<SymbolId, Vec<ProductionId>>::new();
    for (index, production) in document.productions.iter().enumerate() {
        groups
            .entry(production.center.symbol)
            .or_default()
            .push(ProductionId(u32::try_from(index).unwrap_or(u32::MAX)));
    }
    groups
        .into_iter()
        .map(|(symbol, productions)| IrProductionGroup {
            symbol,
            productions,
        })
        .collect()
}

fn compile_grammar(grammar: &Grammar, source: Option<&str>) -> IrDocument {
    let mut symbol_keys = BTreeSet::new();
    collect_word_symbols(&grammar.axiom, &mut symbol_keys);
    for production in &grammar.productions {
        symbol_keys.insert((
            production.center.name.as_str().to_owned(),
            production.center.arguments.len(),
        ));
        if let Some(left) = &production.left {
            collect_pattern_symbols(left, &mut symbol_keys);
        }
        if let Some(right) = &production.right {
            collect_pattern_symbols(right, &mut symbol_keys);
        }
        collect_word_symbols(&production.successor, &mut symbol_keys);
    }
    let symbols = symbol_keys
        .iter()
        .map(|(name, arity)| IrSymbol {
            name: name.clone(),
            arity: u32::try_from(*arity).unwrap_or(u32::MAX),
        })
        .collect::<Vec<_>>();
    let symbol_ids = symbol_keys
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, SymbolId(u32::try_from(index).unwrap_or(u32::MAX))))
        .collect::<BTreeMap<_, _>>();
    let global_ids = grammar
        .bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            (
                binding.name.as_str().to_owned(),
                GlobalId(u32::try_from(index).unwrap_or(u32::MAX)),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let globals = grammar
        .bindings
        .iter()
        .map(|binding| IrGlobal {
            name: binding.name.as_str().to_owned(),
            value: lower_expr(&binding.value, &global_ids, &BTreeMap::new()),
        })
        .collect();
    let axiom = lower_word(&grammar.axiom, &symbol_ids, &global_ids, &BTreeMap::new());
    let productions = grammar
        .productions
        .iter()
        .map(|production| lower_production(production, &symbol_ids, &global_ids))
        .collect();
    let source = source.map(str::to_owned);
    let source_fingerprint = source.as_deref().map(source_fingerprint);
    IrDocument {
        format: String::from(IR_FORMAT_NAME),
        format_version: IR_FORMAT_VERSION,
        semantics_version: IR_SEMANTICS_VERSION,
        source,
        source_fingerprint,
        symbols,
        globals,
        context_filter: grammar.context_filter.as_ref().map(|filter| match filter {
            ContextFilter::Ignore(names) => {
                IrContextFilter::Ignore(names.iter().map(|name| name.as_str().to_owned()).collect())
            }
            ContextFilter::Only(names) => {
                IrContextFilter::Only(names.iter().map(|name| name.as_str().to_owned()).collect())
            }
        }),
        axiom,
        productions,
    }
}

fn collect_word_symbols(word: &Word, output: &mut BTreeSet<(String, usize)>) {
    for item in &word.0 {
        match item {
            WordItem::Module(module) => {
                output.insert((module.name.as_str().to_owned(), module.arguments.len()));
            }
            WordItem::Branch(branch) => collect_word_symbols(branch, output),
        }
    }
}

fn collect_pattern_symbols(word: &PatternWord, output: &mut BTreeSet<(String, usize)>) {
    for item in &word.0 {
        match item {
            PatternItem::Module(module) => {
                output.insert((module.name.as_str().to_owned(), module.arguments.len()));
            }
            PatternItem::Branch(branch) => collect_pattern_symbols(branch, output),
        }
    }
}

fn lower_production(
    production: &Production,
    symbols: &BTreeMap<(String, usize), SymbolId>,
    globals: &BTreeMap<String, GlobalId>,
) -> IrProduction {
    let mut binding_names = Vec::new();
    collect_module_bindings(&production.center, &mut binding_names);
    if let Some(left) = &production.left {
        collect_word_bindings(left, &mut binding_names);
    }
    if let Some(right) = &production.right {
        collect_word_bindings(right, &mut binding_names);
    }
    let bindings = binding_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            (
                name.clone(),
                BindingSlot(u32::try_from(index).unwrap_or(u32::MAX)),
            )
        })
        .collect::<BTreeMap<_, _>>();
    IrProduction {
        bindings: binding_names,
        center: lower_module_pattern(&production.center, symbols, &bindings),
        left: production
            .left
            .as_ref()
            .map(|word| lower_pattern_word(word, symbols, &bindings)),
        right: production
            .right
            .as_ref()
            .map(|word| lower_pattern_word(word, symbols, &bindings)),
        condition: production
            .condition
            .as_ref()
            .map(|expr| lower_expr(expr, globals, &bindings)),
        weight: production
            .weight
            .as_ref()
            .map(|expr| lower_expr(expr, globals, &bindings)),
        successor: lower_word(&production.successor, symbols, globals, &bindings),
    }
}

fn collect_module_bindings(pattern: &ModulePattern, output: &mut Vec<String>) {
    output.extend(pattern.arguments.iter().filter_map(|argument| {
        if let PatternArgument::Bind(name) = argument {
            Some(name.as_str().to_owned())
        } else {
            None
        }
    }));
}

fn collect_word_bindings(word: &PatternWord, output: &mut Vec<String>) {
    for item in &word.0 {
        match item {
            PatternItem::Module(module) => collect_module_bindings(module, output),
            PatternItem::Branch(branch) => collect_word_bindings(branch, output),
        }
    }
}

fn lower_word(
    word: &Word,
    symbols: &BTreeMap<(String, usize), SymbolId>,
    globals: &BTreeMap<String, GlobalId>,
    bindings: &BTreeMap<String, BindingSlot>,
) -> IrWord {
    IrWord {
        items: word
            .0
            .iter()
            .map(|item| match item {
                WordItem::Module(module) => IrWordItem::Module(IrModuleExpr {
                    symbol: symbol_id(symbols, &module.name, module.arguments.len()),
                    arguments: module
                        .arguments
                        .iter()
                        .map(|expr| lower_expr(expr, globals, bindings))
                        .collect(),
                }),
                WordItem::Branch(branch) => {
                    IrWordItem::Branch(lower_word(branch, symbols, globals, bindings))
                }
            })
            .collect(),
    }
}

fn lower_pattern_word(
    word: &PatternWord,
    symbols: &BTreeMap<(String, usize), SymbolId>,
    bindings: &BTreeMap<String, BindingSlot>,
) -> IrPatternWord {
    IrPatternWord {
        items: word
            .0
            .iter()
            .map(|item| match item {
                PatternItem::Module(module) => {
                    IrPatternItem::Module(lower_module_pattern(module, symbols, bindings))
                }
                PatternItem::Branch(branch) => {
                    IrPatternItem::Branch(lower_pattern_word(branch, symbols, bindings))
                }
            })
            .collect(),
    }
}

fn lower_module_pattern(
    pattern: &ModulePattern,
    symbols: &BTreeMap<(String, usize), SymbolId>,
    bindings: &BTreeMap<String, BindingSlot>,
) -> IrModulePattern {
    IrModulePattern {
        symbol: symbol_id(symbols, &pattern.name, pattern.arguments.len()),
        arguments: pattern
            .arguments
            .iter()
            .map(|argument| match argument {
                PatternArgument::Bind(name) => IrPatternArgument::Bind(
                    *bindings
                        .get(name.as_str())
                        .expect("validated binding must have an IR slot"),
                ),
                PatternArgument::Wildcard => IrPatternArgument::Wildcard,
                PatternArgument::Literal(value) => IrPatternArgument::Literal(value.to_bits()),
            })
            .collect(),
    }
}

fn symbol_id(
    symbols: &BTreeMap<(String, usize), SymbolId>,
    name: &Identifier,
    arity: usize,
) -> SymbolId {
    symbols[&(name.as_str().to_owned(), arity)]
}

fn lower_expr(
    expression: &Expr,
    globals: &BTreeMap<String, GlobalId>,
    bindings: &BTreeMap<String, BindingSlot>,
) -> IrExpr {
    match expression {
        Expr::Number(value) => IrExpr::Number(value.to_bits()),
        Expr::Bool(value) => IrExpr::Bool(*value),
        Expr::Name(name) => bindings.get(name.as_str()).map_or_else(
            || {
                IrExpr::Global(
                    *globals
                        .get(name.as_str())
                        .expect("validated name must resolve to a global or local"),
                )
            },
            |slot| IrExpr::Binding(*slot),
        ),
        Expr::Call { name, arguments } => IrExpr::Call {
            name: name.as_str().to_owned(),
            arguments: arguments
                .iter()
                .map(|argument| lower_expr(argument, globals, bindings))
                .collect(),
        },
        Expr::Unary { op, operand } => IrExpr::Unary {
            op: (*op).into(),
            operand: Box::new(lower_expr(operand, globals, bindings)),
        },
        Expr::Binary { op, left, right } => IrExpr::Binary {
            op: (*op).into(),
            left: Box::new(lower_expr(left, globals, bindings)),
            right: Box::new(lower_expr(right, globals, bindings)),
        },
    }
}

impl From<UnaryOp> for IrUnaryOp {
    fn from(value: UnaryOp) -> Self {
        match value {
            UnaryOp::Negate => Self::Negate,
            UnaryOp::Not => Self::Not,
        }
    }
}

impl From<BinaryOp> for IrBinaryOp {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::Add => Self::Add,
            BinaryOp::Subtract => Self::Subtract,
            BinaryOp::Multiply => Self::Multiply,
            BinaryOp::Divide => Self::Divide,
            BinaryOp::Power => Self::Power,
            BinaryOp::Equal => Self::Equal,
            BinaryOp::NotEqual => Self::NotEqual,
            BinaryOp::Less => Self::Less,
            BinaryOp::LessEqual => Self::LessEqual,
            BinaryOp::Greater => Self::Greater,
            BinaryOp::GreaterEqual => Self::GreaterEqual,
            BinaryOp::And => Self::And,
            BinaryOp::Or => Self::Or,
        }
    }
}

fn validate_document(document: &IrDocument) -> Vec<IrDiagnostic> {
    let mut diagnostics = Vec::new();
    if document.format != IR_FORMAT_NAME {
        validation_error(
            &mut diagnostics,
            "format-name",
            format!(
                "IR format `{}` is unsupported; expected `{IR_FORMAT_NAME}`",
                document.format
            ),
            None,
        );
    }
    if document.format_version != IR_FORMAT_VERSION {
        validation_error(
            &mut diagnostics,
            "format-version",
            format!(
                "IR format version {} is unsupported; expected {IR_FORMAT_VERSION}",
                document.format_version
            ),
            None,
        );
    }
    if document.semantics_version != IR_SEMANTICS_VERSION {
        validation_error(
            &mut diagnostics,
            "semantics-version",
            format!(
                "IR semantics version {} is unsupported; expected {IR_SEMANTICS_VERSION}",
                document.semantics_version
            ),
            None,
        );
    }
    match (&document.source, &document.source_fingerprint) {
        (Some(source), Some(fingerprint)) if *fingerprint != source_fingerprint(source) => {
            validation_error(
                &mut diagnostics,
                "source-fingerprint",
                "source fingerprint does not match the embedded source",
                None,
            );
        }
        (None, Some(_)) => validation_error(
            &mut diagnostics,
            "source-fingerprint",
            "a source fingerprint cannot be present without embedded source",
            None,
        ),
        _ => {}
    }
    check_u32_len(document.symbols.len(), "symbols", &mut diagnostics, None);
    check_u32_len(document.globals.len(), "globals", &mut diagnostics, None);
    check_u32_len(
        document.productions.len(),
        "productions",
        &mut diagnostics,
        None,
    );

    let mut symbols = BTreeSet::new();
    for (index, symbol) in document.symbols.iter().enumerate() {
        let entity = Some(format!("symbol s{index}"));
        if symbol.name.is_empty() {
            validation_error(
                &mut diagnostics,
                "empty-symbol",
                "symbol names cannot be empty",
                entity.clone(),
            );
        }
        if !symbols.insert((symbol.name.as_str(), symbol.arity)) {
            validation_error(
                &mut diagnostics,
                "duplicate-symbol",
                format!(
                    "symbol `{}` with arity {} is declared more than once",
                    symbol.name, symbol.arity
                ),
                entity,
            );
        }
    }

    let mut globals = BTreeSet::new();
    for (index, global) in document.globals.iter().enumerate() {
        let entity = Some(format!("global g{index}"));
        if global.name.is_empty() || !globals.insert(global.name.as_str()) {
            validation_error(
                &mut diagnostics,
                "invalid-global",
                format!("global name `{}` is empty or duplicated", global.name),
                entity.clone(),
            );
        }
        validate_expr(
            &global.value,
            document.globals.len(),
            0,
            false,
            &mut diagnostics,
            entity,
        );
    }

    validate_word(
        &document.axiom,
        document,
        0,
        false,
        &mut diagnostics,
        Some(String::from("axiom")),
    );
    for (index, production) in document.productions.iter().enumerate() {
        let entity = Some(format!("production p{index}"));
        check_u32_len(
            production.bindings.len(),
            "bindings",
            &mut diagnostics,
            entity.clone(),
        );
        let mut names = BTreeSet::new();
        for name in &production.bindings {
            if name.is_empty() || !names.insert(name.as_str()) {
                validation_error(
                    &mut diagnostics,
                    "invalid-binding",
                    format!("binding name `{name}` is empty or duplicated"),
                    entity.clone(),
                );
            }
        }
        let mut binding_uses = vec![0_u32; production.bindings.len()];
        validate_module_pattern(
            &production.center,
            document,
            &mut binding_uses,
            &mut diagnostics,
            entity.clone(),
        );
        if let Some(left) = &production.left {
            validate_pattern_word(
                left,
                document,
                &mut binding_uses,
                &mut diagnostics,
                entity.clone(),
            );
        }
        if let Some(right) = &production.right {
            validate_pattern_word(
                right,
                document,
                &mut binding_uses,
                &mut diagnostics,
                entity.clone(),
            );
        }
        for (slot, uses) in binding_uses.into_iter().enumerate() {
            if uses != 1 {
                validation_error(
                    &mut diagnostics,
                    "binding-definition",
                    format!(
                        "binding ${slot} must be defined exactly once, but is used {uses} times"
                    ),
                    entity.clone(),
                );
            }
        }
        if let Some(condition) = &production.condition {
            validate_expr(
                condition,
                document.globals.len(),
                production.bindings.len(),
                true,
                &mut diagnostics,
                entity.clone(),
            );
        }
        if let Some(weight) = &production.weight {
            validate_expr(
                weight,
                document.globals.len(),
                production.bindings.len(),
                true,
                &mut diagnostics,
                entity.clone(),
            );
        }
        validate_word(
            &production.successor,
            document,
            production.bindings.len(),
            true,
            &mut diagnostics,
            entity,
        );
    }
    diagnostics
}

fn validation_error(
    diagnostics: &mut Vec<IrDiagnostic>,
    code: &'static str,
    message: impl Into<String>,
    entity: Option<String>,
) {
    diagnostics.push(IrDiagnostic {
        severity: IrDiagnosticSeverity::Error,
        code,
        message: message.into(),
        entity,
        backend: None,
    });
}

fn check_u32_len(
    length: usize,
    what: &str,
    diagnostics: &mut Vec<IrDiagnostic>,
    entity: Option<String>,
) {
    if u32::try_from(length).is_err() {
        validation_error(
            diagnostics,
            "index-overflow",
            format!("{what} count {length} does not fit the portable u32 index space"),
            entity,
        );
    }
}

fn validate_expr(
    expression: &IrExpr,
    globals: usize,
    bindings: usize,
    bindings_allowed: bool,
    diagnostics: &mut Vec<IrDiagnostic>,
    entity: Option<String>,
) {
    match expression {
        IrExpr::Number(_) | IrExpr::Bool(_) => {}
        IrExpr::Global(id) => {
            if usize::try_from(id.0).map_or(true, |index| index >= globals) {
                validation_error(
                    diagnostics,
                    "global-reference",
                    format!("global reference {id} is out of range"),
                    entity,
                );
            }
        }
        IrExpr::Binding(slot) => {
            if !bindings_allowed || usize::try_from(slot.0).map_or(true, |index| index >= bindings)
            {
                validation_error(
                    diagnostics,
                    "binding-reference",
                    format!("binding reference {slot} is unavailable in this expression"),
                    entity,
                );
            }
        }
        IrExpr::Call { name, arguments } => {
            if name.is_empty() {
                validation_error(
                    diagnostics,
                    "empty-builtin",
                    "builtin names cannot be empty",
                    entity.clone(),
                );
            }
            check_u32_len(
                arguments.len(),
                "call arguments",
                diagnostics,
                entity.clone(),
            );
            for argument in arguments {
                validate_expr(
                    argument,
                    globals,
                    bindings,
                    bindings_allowed,
                    diagnostics,
                    entity.clone(),
                );
            }
        }
        IrExpr::Unary { operand, .. } => validate_expr(
            operand,
            globals,
            bindings,
            bindings_allowed,
            diagnostics,
            entity,
        ),
        IrExpr::Binary { left, right, .. } => {
            validate_expr(
                left,
                globals,
                bindings,
                bindings_allowed,
                diagnostics,
                entity.clone(),
            );
            validate_expr(
                right,
                globals,
                bindings,
                bindings_allowed,
                diagnostics,
                entity,
            );
        }
    }
}

fn validate_word(
    word: &IrWord,
    document: &IrDocument,
    bindings: usize,
    bindings_allowed: bool,
    diagnostics: &mut Vec<IrDiagnostic>,
    entity: Option<String>,
) {
    check_u32_len(word.items.len(), "word items", diagnostics, entity.clone());
    for item in &word.items {
        match item {
            IrWordItem::Module(module) => {
                validate_symbol_arity(
                    module.symbol,
                    module.arguments.len(),
                    document,
                    diagnostics,
                    entity.clone(),
                );
                for argument in &module.arguments {
                    validate_expr(
                        argument,
                        document.globals.len(),
                        bindings,
                        bindings_allowed,
                        diagnostics,
                        entity.clone(),
                    );
                }
            }
            IrWordItem::Branch(branch) => validate_word(
                branch,
                document,
                bindings,
                bindings_allowed,
                diagnostics,
                entity.clone(),
            ),
        }
    }
}

fn validate_pattern_word(
    word: &IrPatternWord,
    document: &IrDocument,
    binding_uses: &mut [u32],
    diagnostics: &mut Vec<IrDiagnostic>,
    entity: Option<String>,
) {
    check_u32_len(
        word.items.len(),
        "pattern items",
        diagnostics,
        entity.clone(),
    );
    for item in &word.items {
        match item {
            IrPatternItem::Module(module) => {
                validate_module_pattern(module, document, binding_uses, diagnostics, entity.clone())
            }
            IrPatternItem::Branch(branch) => {
                validate_pattern_word(branch, document, binding_uses, diagnostics, entity.clone())
            }
        }
    }
}

fn validate_module_pattern(
    pattern: &IrModulePattern,
    document: &IrDocument,
    binding_uses: &mut [u32],
    diagnostics: &mut Vec<IrDiagnostic>,
    entity: Option<String>,
) {
    validate_symbol_arity(
        pattern.symbol,
        pattern.arguments.len(),
        document,
        diagnostics,
        entity.clone(),
    );
    for argument in &pattern.arguments {
        if let IrPatternArgument::Bind(slot) = argument {
            let Ok(index) = usize::try_from(slot.0) else {
                validation_error(
                    diagnostics,
                    "binding-definition",
                    format!("binding definition {slot} is out of range"),
                    entity.clone(),
                );
                continue;
            };
            if let Some(uses) = binding_uses.get_mut(index) {
                *uses = uses.saturating_add(1);
            } else {
                validation_error(
                    diagnostics,
                    "binding-definition",
                    format!("binding definition {slot} is out of range"),
                    entity.clone(),
                );
            }
        }
    }
}

fn validate_symbol_arity(
    id: SymbolId,
    actual_arity: usize,
    document: &IrDocument,
    diagnostics: &mut Vec<IrDiagnostic>,
    entity: Option<String>,
) {
    let symbol = usize::try_from(id.0)
        .ok()
        .and_then(|index| document.symbols.get(index));
    match symbol {
        None => validation_error(
            diagnostics,
            "symbol-reference",
            format!("symbol reference {id} is out of range"),
            entity,
        ),
        Some(symbol) if usize::try_from(symbol.arity).ok() != Some(actual_arity) => {
            validation_error(
                diagnostics,
                "symbol-arity",
                format!(
                    "{id} declares arity {} but is used with {actual_arity} argument(s)",
                    symbol.arity
                ),
                entity,
            );
        }
        Some(_) => {}
    }
}

fn compile_programs(document: &IrDocument) -> Vec<IrProgram> {
    let mut programs = Vec::new();
    for (index, global) in document.globals.iter().enumerate() {
        push_program(
            &mut programs,
            IrProgramOwner::Global(GlobalId(u32::try_from(index).unwrap_or(u32::MAX))),
            &global.value,
        );
    }
    let mut module = 0_u32;
    visit_word_expressions(&document.axiom, &mut module, |module, argument, expr| {
        push_program(
            &mut programs,
            IrProgramOwner::AxiomArgument { module, argument },
            expr,
        );
    });
    for (index, production) in document.productions.iter().enumerate() {
        let production_id = ProductionId(u32::try_from(index).unwrap_or(u32::MAX));
        if let Some(condition) = &production.condition {
            push_program(
                &mut programs,
                IrProgramOwner::Condition(production_id),
                condition,
            );
        }
        if let Some(weight) = &production.weight {
            push_program(&mut programs, IrProgramOwner::Weight(production_id), weight);
        }
        let mut module = 0_u32;
        visit_word_expressions(
            &production.successor,
            &mut module,
            |module, argument, expr| {
                push_program(
                    &mut programs,
                    IrProgramOwner::SuccessorArgument {
                        production: production_id,
                        module,
                        argument,
                    },
                    expr,
                );
            },
        );
    }
    programs
}

fn visit_word_expressions(
    word: &IrWord,
    module: &mut u32,
    mut visitor: impl FnMut(u32, u32, &IrExpr),
) {
    fn visit(word: &IrWord, module: &mut u32, visitor: &mut impl FnMut(u32, u32, &IrExpr)) {
        for item in &word.items {
            match item {
                IrWordItem::Module(item) => {
                    let current = *module;
                    *module = module.saturating_add(1);
                    for (argument, expression) in item.arguments.iter().enumerate() {
                        visitor(
                            current,
                            u32::try_from(argument).unwrap_or(u32::MAX),
                            expression,
                        );
                    }
                }
                IrWordItem::Branch(branch) => visit(branch, module, visitor),
            }
        }
    }
    visit(word, module, &mut visitor);
}

fn push_program(programs: &mut Vec<IrProgram>, owner: IrProgramOwner, expression: &IrExpr) {
    let mut instructions = Vec::new();
    let (_, max_stack) = compile_expr_program(expression, 0, &mut instructions);
    instructions.push(IrInstruction::Return);
    programs.push(IrProgram {
        id: ProgramId(u32::try_from(programs.len()).unwrap_or(u32::MAX)),
        owner,
        instructions,
        max_stack,
    });
}

/// Compiles `expression` with `depth` values already on the stack and returns
/// the final depth and maximum depth observed.
fn compile_expr_program(
    expression: &IrExpr,
    depth: u32,
    instructions: &mut Vec<IrInstruction>,
) -> (u32, u32) {
    match expression {
        IrExpr::Number(bits) => {
            instructions.push(IrInstruction::PushNumber(*bits));
            (depth.saturating_add(1), depth.saturating_add(1))
        }
        IrExpr::Bool(value) => {
            instructions.push(IrInstruction::PushBool(*value));
            (depth.saturating_add(1), depth.saturating_add(1))
        }
        IrExpr::Global(id) => {
            instructions.push(IrInstruction::LoadGlobal(*id));
            (depth.saturating_add(1), depth.saturating_add(1))
        }
        IrExpr::Binding(slot) => {
            instructions.push(IrInstruction::LoadBinding(*slot));
            (depth.saturating_add(1), depth.saturating_add(1))
        }
        IrExpr::Call { name, arguments } => {
            let mut current = depth;
            let mut maximum = depth;
            for argument in arguments {
                let (next, argument_maximum) =
                    compile_expr_program(argument, current, instructions);
                current = next;
                maximum = maximum.max(argument_maximum);
            }
            instructions.push(IrInstruction::Call {
                name: name.clone(),
                arity: u32::try_from(arguments.len()).unwrap_or(u32::MAX),
            });
            let final_depth = current
                .saturating_sub(u32::try_from(arguments.len()).unwrap_or(u32::MAX))
                .saturating_add(1);
            (final_depth, maximum.max(final_depth))
        }
        IrExpr::Unary { op, operand } => {
            let (final_depth, maximum) = compile_expr_program(operand, depth, instructions);
            instructions.push(IrInstruction::Unary(*op));
            (final_depth, maximum)
        }
        IrExpr::Binary { op, left, right } if matches!(op, IrBinaryOp::And | IrBinaryOp::Or) => {
            let (left_depth, left_maximum) = compile_expr_program(left, depth, instructions);
            let jump_index = instructions.len();
            instructions.push(if *op == IrBinaryOp::And {
                IrInstruction::JumpIfFalse(0)
            } else {
                IrInstruction::JumpIfTrue(0)
            });
            instructions.push(IrInstruction::Pop);
            let (right_depth, right_maximum) = compile_expr_program(right, depth, instructions);
            let target = u32::try_from(instructions.len()).unwrap_or(u32::MAX);
            instructions[jump_index] = if *op == IrBinaryOp::And {
                IrInstruction::JumpIfFalse(target)
            } else {
                IrInstruction::JumpIfTrue(target)
            };
            debug_assert_eq!(left_depth, right_depth);
            (right_depth, left_maximum.max(right_maximum))
        }
        IrExpr::Binary { op, left, right } => {
            let (left_depth, left_maximum) = compile_expr_program(left, depth, instructions);
            let (right_depth, right_maximum) =
                compile_expr_program(right, left_depth, instructions);
            instructions.push(IrInstruction::Binary(*op));
            (
                right_depth.saturating_sub(1),
                left_maximum.max(right_maximum),
            )
        }
    }
}

fn collect_requirements(document: &IrDocument, programs: &[IrProgram]) -> IrRequirements {
    let mut requirements = IrRequirements {
        symbol_count: u32::try_from(document.symbols.len()).unwrap_or(u32::MAX),
        global_count: u32::try_from(document.globals.len()).unwrap_or(u32::MAX),
        production_count: u32::try_from(document.productions.len()).unwrap_or(u32::MAX),
        context_filter: document.context_filter.is_some(),
        maximum_expression_stack: programs
            .iter()
            .map(|program| program.max_stack)
            .max()
            .unwrap_or(0),
        ..IrRequirements::default()
    };
    inspect_word_requirements(&document.axiom, &mut requirements);
    for global in &document.globals {
        inspect_expr_requirements(&global.value, &mut requirements);
    }
    for production in &document.productions {
        requirements.maximum_bindings = requirements
            .maximum_bindings
            .max(u32::try_from(production.bindings.len()).unwrap_or(u32::MAX));
        requirements.parameterized_modules |= !production.center.arguments.is_empty();
        requirements.contextual_productions |=
            production.left.is_some() || production.right.is_some();
        if let Some(left) = &production.left {
            inspect_pattern_requirements(left, 1, &mut requirements);
        }
        if let Some(right) = &production.right {
            inspect_pattern_requirements(right, 1, &mut requirements);
        }
        if let Some(condition) = &production.condition {
            requirements.dynamic_conditions |= expr_uses_binding(condition);
            inspect_expr_requirements(condition, &mut requirements);
        }
        if let Some(weight) = &production.weight {
            requirements.weighted_productions = true;
            requirements.dynamic_weights |= expr_uses_binding(weight);
            inspect_expr_requirements(weight, &mut requirements);
        }
        inspect_word_requirements(&production.successor, &mut requirements);
    }
    requirements
}

fn inspect_word_requirements(word: &IrWord, requirements: &mut IrRequirements) {
    for item in &word.items {
        match item {
            IrWordItem::Module(module) => {
                requirements.parameterized_modules |= !module.arguments.is_empty();
                for argument in &module.arguments {
                    inspect_expr_requirements(argument, requirements);
                }
            }
            IrWordItem::Branch(branch) => {
                requirements.structural_branches = true;
                inspect_word_requirements(branch, requirements);
            }
        }
    }
}

fn inspect_pattern_requirements(
    word: &IrPatternWord,
    depth: u32,
    requirements: &mut IrRequirements,
) {
    requirements.maximum_context_depth = requirements.maximum_context_depth.max(depth);
    for item in &word.items {
        match item {
            IrPatternItem::Module(module) => {
                requirements.parameterized_modules |= !module.arguments.is_empty();
            }
            IrPatternItem::Branch(branch) => {
                requirements.structural_contexts = true;
                inspect_pattern_requirements(branch, depth.saturating_add(1), requirements);
            }
        }
    }
}

fn inspect_expr_requirements(expression: &IrExpr, requirements: &mut IrRequirements) {
    match expression {
        IrExpr::Call { name, arguments } => {
            requirements.builtins.insert(name.clone());
            for argument in arguments {
                inspect_expr_requirements(argument, requirements);
            }
        }
        IrExpr::Unary { operand, .. } => inspect_expr_requirements(operand, requirements),
        IrExpr::Binary { left, right, .. } => {
            inspect_expr_requirements(left, requirements);
            inspect_expr_requirements(right, requirements);
        }
        IrExpr::Number(_) | IrExpr::Bool(_) | IrExpr::Global(_) | IrExpr::Binding(_) => {}
    }
}

fn expr_uses_binding(expression: &IrExpr) -> bool {
    match expression {
        IrExpr::Binding(_) => true,
        IrExpr::Call { arguments, .. } => arguments.iter().any(expr_uses_binding),
        IrExpr::Unary { operand, .. } => expr_uses_binding(operand),
        IrExpr::Binary { left, right, .. } => expr_uses_binding(left) || expr_uses_binding(right),
        IrExpr::Number(_) | IrExpr::Bool(_) | IrExpr::Global(_) => false,
    }
}

fn compatibility(
    requirements: &IrRequirements,
    backend: IrBackendProfile,
) -> IrCompatibilityReport {
    let mut diagnostics = Vec::new();
    if backend == IrBackendProfile::Cpu {
        return IrCompatibilityReport {
            backend,
            compatible: true,
            diagnostics,
        };
    }
    let mut unsupported = |code: &'static str, message: &'static str| {
        diagnostics.push(IrDiagnostic {
            severity: IrDiagnosticSeverity::Error,
            code,
            message: message.to_owned(),
            entity: None,
            backend: Some(backend),
        });
    };
    if requirements.parameterized_modules {
        unsupported(
            "gpu-parameters",
            "the current GPU derivation compiler accepts only argument-free modules",
        );
    }
    if requirements.contextual_productions {
        unsupported(
            "gpu-context",
            "the current GPU derivation compiler accepts only context-free productions",
        );
    }
    if requirements.context_filter {
        unsupported(
            "gpu-context-filter",
            "the current GPU derivation compiler does not implement ignore/only filters",
        );
    }
    if requirements.dynamic_conditions {
        unsupported(
            "gpu-dynamic-condition",
            "the current GPU derivation compiler requires conditions to be independent of module bindings",
        );
    }
    if requirements.dynamic_weights {
        unsupported(
            "gpu-dynamic-weight",
            "the current GPU derivation compiler requires weights to be independent of module bindings",
        );
    }
    IrCompatibilityReport {
        backend,
        compatible: !diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == IrDiagnosticSeverity::Error),
        diagnostics,
    }
}

fn compatibility_with_semantics(
    requirements: &IrRequirements,
    backend: IrBackendProfile,
    semantics: crate::execution::DerivationSemantics,
) -> IrCompatibilityReport {
    use crate::grammar::FloatWidth;

    let mut report = compatibility(requirements, backend);
    let unsupported = match backend {
        IrBackendProfile::Cpu => None,
        IrBackendProfile::CurrentCuda if semantics.float_width != FloatWidth::F64 => Some((
            "backend-float-width",
            format!(
                "the current CUDA derivation kernels implement f64, not {}",
                semantics.float_width
            ),
        )),
        IrBackendProfile::CurrentWgpu if semantics.float_width != FloatWidth::F32 => Some((
            "backend-float-width",
            format!(
                "the current WGPU derivation kernels implement f32, not {}",
                semantics.float_width
            ),
        )),
        _ => None,
    };
    if let Some((code, message)) = unsupported {
        report.diagnostics.push(IrDiagnostic {
            severity: IrDiagnosticSeverity::Error,
            code,
            message,
            entity: None,
            backend: Some(backend),
        });
    }
    report.compatible = !report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == IrDiagnosticSeverity::Error);
    report
}

fn disassemble(ir: &ValidatedIr) -> String {
    let document = &ir.document;
    let requirements = &ir.requirements;
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Braken IR v{} / semantics v{}",
        document.format_version, document.semantics_version
    );
    let _ = writeln!(
        output,
        "{} symbols · {} globals · {} productions · {} expression programs",
        document.symbols.len(),
        document.globals.len(),
        document.productions.len(),
        ir.programs.len()
    );
    let _ = writeln!(
        output,
        "maximum expression stack {} · maximum bindings {} · maximum context depth {}",
        requirements.maximum_expression_stack,
        requirements.maximum_bindings,
        requirements.maximum_context_depth
    );
    if let Some(fingerprint) = &document.source_fingerprint {
        let _ = writeln!(output, "source fingerprint {fingerprint}");
    }
    output.push_str("\nsymbols\n");
    for (index, symbol) in document.symbols.iter().enumerate() {
        let _ = writeln!(output, "  s{index} = {} / {}", symbol.name, symbol.arity);
    }
    if !document.globals.is_empty() {
        output.push_str("\nglobals\n");
        for (index, global) in document.globals.iter().enumerate() {
            let _ = writeln!(output, "  g{index} = {}", global.name);
        }
    }
    output.push_str("\naxiom\n  ");
    format_word(&mut output, &document.axiom, document);
    output.push('\n');
    if let Some(filter) = &document.context_filter {
        let (kind, names) = match filter {
            IrContextFilter::Ignore(names) => ("ignore", names),
            IrContextFilter::Only(names) => ("only", names),
        };
        let _ = writeln!(output, "\ncontext filter\n  {kind} {}", names.join(" "));
    }
    output.push_str("\nproductions\n");
    for (index, production) in document.productions.iter().enumerate() {
        let _ = write!(output, "  p{index}: ");
        format_module_pattern(&mut output, &production.center, production, document);
        if let Some(left) = &production.left {
            output.push_str(" left ");
            format_pattern_word(&mut output, left, production, document);
        }
        if let Some(right) = &production.right {
            output.push_str(" right ");
            format_pattern_word(&mut output, right, production, document);
        }
        if production.condition.is_some() {
            output.push_str(" when e");
            if let Some(program) = ir.programs.iter().find(|program| {
                program.owner == IrProgramOwner::Condition(ProductionId(index as u32))
            }) {
                let _ = write!(output, "{}", program.id.0);
            } else {
                output.push('?');
            }
        }
        if production.weight.is_some() {
            output.push_str(" weight e");
            if let Some(program) = ir
                .programs
                .iter()
                .find(|program| program.owner == IrProgramOwner::Weight(ProductionId(index as u32)))
            {
                let _ = write!(output, "{}", program.id.0);
            } else {
                output.push('?');
            }
        }
        output.push_str(" then ");
        if production.successor.items.is_empty() {
            output.push_str("nothing");
        } else {
            format_word(&mut output, &production.successor, document);
        }
        output.push('\n');
        if !production.bindings.is_empty() {
            output.push_str("    bindings:");
            for (slot, name) in production.bindings.iter().enumerate() {
                let _ = write!(output, " ${slot}={name}");
            }
            output.push('\n');
        }
    }
    output.push_str("\nexpression programs\n");
    for program in &ir.programs {
        let _ = writeln!(
            output,
            "  {} {:?} (stack {})",
            program.id, program.owner, program.max_stack
        );
        for (offset, instruction) in program.instructions.iter().enumerate() {
            let _ = writeln!(
                output,
                "    {offset:04} {}",
                display_instruction(instruction)
            );
        }
    }
    output.push_str("\nbackend compatibility\n");
    for backend in [
        IrBackendProfile::Cpu,
        IrBackendProfile::CurrentCuda,
        IrBackendProfile::CurrentWgpu,
    ] {
        let report = compatibility(requirements, backend);
        let _ = writeln!(
            output,
            "  {backend}: {}",
            if report.compatible {
                "compatible"
            } else {
                "unsupported"
            }
        );
        for diagnostic in report.diagnostics {
            let _ = writeln!(
                output,
                "    {:?} [{}] {}",
                diagnostic.severity, diagnostic.code, diagnostic.message
            );
        }
    }
    output
}

fn display_instruction(instruction: &IrInstruction) -> String {
    match instruction {
        IrInstruction::PushNumber(bits) => {
            format!("push.number 0x{bits:016x} ; {:?}", f64::from_bits(*bits))
        }
        IrInstruction::PushBool(value) => format!("push.bool {value}"),
        IrInstruction::LoadGlobal(id) => format!("load.global {id}"),
        IrInstruction::LoadBinding(slot) => format!("load.binding {slot}"),
        IrInstruction::Call { name, arity } => format!("call {name}/{arity}"),
        IrInstruction::Unary(op) => format!("unary {op:?}"),
        IrInstruction::Binary(op) => format!("binary {op:?}"),
        IrInstruction::JumpIfFalse(target) => format!("jump_if_false {target:04}"),
        IrInstruction::JumpIfTrue(target) => format!("jump_if_true {target:04}"),
        IrInstruction::Pop => String::from("pop"),
        IrInstruction::Return => String::from("return"),
    }
}

fn format_word(output: &mut String, word: &IrWord, document: &IrDocument) {
    for (index, item) in word.items.iter().enumerate() {
        if index > 0 {
            output.push(' ');
        }
        match item {
            IrWordItem::Module(module) => {
                let symbol = &document.symbols[module.symbol.0 as usize];
                output.push_str(&symbol.name);
                if !module.arguments.is_empty() {
                    output.push('(');
                    for (argument, _) in module.arguments.iter().enumerate() {
                        if argument > 0 {
                            output.push_str(", ");
                        }
                        output.push_str("expr");
                    }
                    output.push(')');
                }
            }
            IrWordItem::Branch(branch) => {
                output.push('[');
                format_word(output, branch, document);
                output.push(']');
            }
        }
    }
}

fn format_pattern_word(
    output: &mut String,
    word: &IrPatternWord,
    production: &IrProduction,
    document: &IrDocument,
) {
    for (index, item) in word.items.iter().enumerate() {
        if index > 0 {
            output.push(' ');
        }
        match item {
            IrPatternItem::Module(module) => {
                format_module_pattern(output, module, production, document);
            }
            IrPatternItem::Branch(branch) => {
                output.push('[');
                format_pattern_word(output, branch, production, document);
                output.push(']');
            }
        }
    }
}

fn format_module_pattern(
    output: &mut String,
    module: &IrModulePattern,
    production: &IrProduction,
    document: &IrDocument,
) {
    let symbol = &document.symbols[module.symbol.0 as usize];
    output.push_str(&symbol.name);
    if module.arguments.is_empty() {
        return;
    }
    output.push('(');
    for (index, argument) in module.arguments.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        match argument {
            IrPatternArgument::Bind(slot) => output.push_str(
                production
                    .bindings
                    .get(slot.0 as usize)
                    .map_or("<invalid>", String::as_str),
            ),
            IrPatternArgument::Wildcard => output.push('_'),
            IrPatternArgument::Literal(bits) => {
                let _ = write!(output, "{:?}", f64::from_bits(*bits));
            }
        }
    }
    output.push(')');
}

fn source_fingerprint(source: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in source.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

impl DerivationIr {
    pub(crate) fn to_grammar(&self) -> Result<Grammar, IrValidationErrors> {
        let document = &self.0.document;
        let mut items = document
            .globals
            .iter()
            .map(|global| {
                Item::Let(Binding {
                    name: Identifier::new(global.name.clone()),
                    value: raise_expr(&global.value, document, &[]),
                })
            })
            .collect::<Vec<_>>();
        if let Some(filter) = &document.context_filter {
            items.push(match filter {
                IrContextFilter::Ignore(names) => Item::Ignore(
                    names
                        .iter()
                        .map(|name| Identifier::new(name.clone()))
                        .collect(),
                ),
                IrContextFilter::Only(names) => Item::Only(
                    names
                        .iter()
                        .map(|name| Identifier::new(name.clone()))
                        .collect(),
                ),
            });
        }
        items.push(Item::Axiom(raise_word(&document.axiom, document, &[])));
        items.extend(document.productions.iter().map(|production| {
            let bindings = production
                .bindings
                .iter()
                .map(|name| Identifier::new(name.clone()))
                .collect::<Vec<_>>();
            Item::Production(Production {
                center: raise_module_pattern(&production.center, document, &bindings),
                left: production
                    .left
                    .as_ref()
                    .map(|word| raise_pattern_word(word, document, &bindings)),
                right: production
                    .right
                    .as_ref()
                    .map(|word| raise_pattern_word(word, document, &bindings)),
                condition: production
                    .condition
                    .as_ref()
                    .map(|expr| raise_expr(expr, document, &bindings)),
                weight: production
                    .weight
                    .as_ref()
                    .map(|expr| raise_expr(expr, document, &bindings)),
                successor: raise_word(&production.successor, document, &bindings),
            })
        }));
        Document { items }.validate().map_err(|errors| {
            IrValidationErrors(
                errors
                    .0
                    .into_iter()
                    .map(|error| IrDiagnostic {
                        severity: IrDiagnosticSeverity::Error,
                        code: "grammar-validation",
                        message: error.message,
                        entity: None,
                        backend: None,
                    })
                    .collect(),
            )
        })
    }
}

fn raise_word(word: &IrWord, document: &IrDocument, bindings: &[Identifier]) -> Word {
    Word(
        word.items
            .iter()
            .map(|item| match item {
                IrWordItem::Module(module) => WordItem::Module(ModuleExpr {
                    name: Identifier::new(document.symbols[module.symbol.0 as usize].name.clone()),
                    arguments: module
                        .arguments
                        .iter()
                        .map(|expr| raise_expr(expr, document, bindings))
                        .collect(),
                }),
                IrWordItem::Branch(branch) => {
                    WordItem::Branch(raise_word(branch, document, bindings))
                }
            })
            .collect(),
    )
}

fn raise_pattern_word(
    word: &IrPatternWord,
    document: &IrDocument,
    bindings: &[Identifier],
) -> PatternWord {
    PatternWord(
        word.items
            .iter()
            .map(|item| match item {
                IrPatternItem::Module(module) => {
                    PatternItem::Module(raise_module_pattern(module, document, bindings))
                }
                IrPatternItem::Branch(branch) => {
                    PatternItem::Branch(raise_pattern_word(branch, document, bindings))
                }
            })
            .collect(),
    )
}

fn raise_module_pattern(
    module: &IrModulePattern,
    document: &IrDocument,
    bindings: &[Identifier],
) -> ModulePattern {
    ModulePattern {
        name: Identifier::new(document.symbols[module.symbol.0 as usize].name.clone()),
        arguments: module
            .arguments
            .iter()
            .map(|argument| match argument {
                IrPatternArgument::Bind(slot) => {
                    PatternArgument::Bind(bindings[slot.0 as usize].clone())
                }
                IrPatternArgument::Wildcard => PatternArgument::Wildcard,
                IrPatternArgument::Literal(bits) => PatternArgument::Literal(f64::from_bits(*bits)),
            })
            .collect(),
    }
}

fn raise_expr(expression: &IrExpr, document: &IrDocument, bindings: &[Identifier]) -> Expr {
    match expression {
        IrExpr::Number(bits) => Expr::Number(f64::from_bits(*bits)),
        IrExpr::Bool(value) => Expr::Bool(*value),
        IrExpr::Global(id) => Expr::Name(Identifier::new(
            document.globals[id.0 as usize].name.clone(),
        )),
        IrExpr::Binding(slot) => Expr::Name(bindings[slot.0 as usize].clone()),
        IrExpr::Call { name, arguments } => Expr::Call {
            name: Identifier::new(name.clone()),
            arguments: arguments
                .iter()
                .map(|argument| raise_expr(argument, document, bindings))
                .collect(),
        },
        IrExpr::Unary { op, operand } => Expr::Unary {
            op: match op {
                IrUnaryOp::Negate => UnaryOp::Negate,
                IrUnaryOp::Not => UnaryOp::Not,
            },
            operand: Box::new(raise_expr(operand, document, bindings)),
        },
        IrExpr::Binary { op, left, right } => Expr::Binary {
            op: match op {
                IrBinaryOp::Add => BinaryOp::Add,
                IrBinaryOp::Subtract => BinaryOp::Subtract,
                IrBinaryOp::Multiply => BinaryOp::Multiply,
                IrBinaryOp::Divide => BinaryOp::Divide,
                IrBinaryOp::Power => BinaryOp::Power,
                IrBinaryOp::Equal => BinaryOp::Equal,
                IrBinaryOp::NotEqual => BinaryOp::NotEqual,
                IrBinaryOp::Less => BinaryOp::Less,
                IrBinaryOp::LessEqual => BinaryOp::LessEqual,
                IrBinaryOp::Greater => BinaryOp::Greater,
                IrBinaryOp::GreaterEqual => BinaryOp::GreaterEqual,
                IrBinaryOp::And => BinaryOp::And,
                IrBinaryOp::Or => BinaryOp::Or,
            },
            left: Box::new(raise_expr(left, document, bindings)),
            right: Box::new(raise_expr(right, document, bindings)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grammar() -> Grammar {
        Grammar::parse(
            r#"
                let STEP = 2;
                only F;
                axiom F(1) [ Turn F(2) ];
                match F(x) left F(l) right F(r)
                    when x > 0 and r > l
                    weight STEP + x
                    then F(x + STEP) [ F(r) ];
            "#,
        )
        .unwrap()
    }

    #[test]
    fn grammar_round_trips_through_validated_ir() {
        let grammar = grammar();
        let ir = IrDocument::from_grammar(&grammar).validate().unwrap();
        assert_eq!(ir.to_grammar().unwrap(), grammar);
        assert!(ir.view().requirements().parameterized_modules);
        assert!(ir.view().requirements().contextual_productions);
        assert!(ir.view().requirements().structural_branches);
        assert!(ir.view().requirements().dynamic_conditions);
        assert!(ir.view().requirements().dynamic_weights);
    }

    #[test]
    fn json_round_trip_preserves_exact_number_bits() {
        let document = IrDocument::from_grammar(&grammar());
        let json = document.to_json_pretty().unwrap();
        assert!(json.contains("0x3ff0000000000000"));
        let decoded = IrDocument::from_json(&json).unwrap();
        assert_eq!(decoded, document);
        decoded.validate().unwrap();
    }

    #[test]
    fn short_circuit_bytecode_keeps_the_left_value() {
        let ir = IrDocument::from_grammar(&grammar()).validate().unwrap();
        let condition = ir
            .view()
            .programs()
            .iter()
            .find(|program| matches!(program.owner, IrProgramOwner::Condition(_)))
            .unwrap();
        assert!(
            condition
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, IrInstruction::JumpIfFalse(_)))
        );
        assert_eq!(condition.instructions.last(), Some(&IrInstruction::Return));
        assert_eq!(
            ir.view()
                .evaluate_program(
                    condition.id,
                    &[crate::grammar::Value::Number(2.0)],
                    &[
                        crate::grammar::Value::Number(1.0),
                        crate::grammar::Value::Number(2.0),
                        crate::grammar::Value::Number(3.0),
                    ],
                )
                .unwrap(),
            crate::grammar::Value::Bool(true)
        );
    }

    #[test]
    fn bytecode_evaluator_preserves_short_circuit_errors() {
        let grammar =
            Grammar::parse("axiom A; match A when false and unknown_function() then B;").unwrap();
        let ir = IrDocument::from_grammar(&grammar).validate().unwrap();
        let condition = ir
            .view()
            .programs()
            .iter()
            .find(|program| matches!(program.owner, IrProgramOwner::Condition(_)))
            .unwrap();
        assert_eq!(
            ir.view().evaluate_program(condition.id, &[], &[]).unwrap(),
            crate::grammar::Value::Bool(false)
        );
    }

    #[test]
    fn bytecode_evaluator_resolves_forward_global_dependencies() {
        let grammar = Grammar::parse("let A = B + 1; let B = sqrt(9); axiom F(A);").unwrap();
        let ir = IrDocument::from_grammar(&grammar).validate().unwrap();
        assert_eq!(
            ir.view().evaluate_globals().unwrap(),
            [
                crate::grammar::Value::Number(4.0),
                crate::grammar::Value::Number(3.0),
            ]
        );
    }

    #[test]
    fn compatibility_reports_the_current_gpu_subset() {
        let ir = IrDocument::from_grammar(&grammar()).validate().unwrap();
        assert!(
            ir.compatibility(IrBackendProfile::Cpu).compatible,
            "CPU is the full-grammar reference"
        );
        let cuda = ir.compatibility(IrBackendProfile::CurrentCuda);
        assert!(!cuda.compatible);
        assert!(
            cuda.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "gpu-parameters")
        );
        assert!(
            cuda.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "gpu-context")
        );
    }

    #[test]
    fn compatibility_accepts_the_declared_flat_gpu_subset() {
        let grammar = Grammar::parse(
            "let W = 2; axiom A [ B ]; match A weight W then A B; match A weight 1 then B;",
        )
        .unwrap();
        let ir = IrDocument::from_grammar(&grammar).validate().unwrap();
        assert!(ir.compatibility(IrBackendProfile::CurrentCuda).compatible);
        assert!(ir.compatibility(IrBackendProfile::CurrentWgpu).compatible);
    }

    #[test]
    fn compatibility_is_scoped_to_numeric_width_and_ambiguity_policy() {
        use crate::execution::DerivationSemantics;
        use crate::grammar::{AmbiguousRulePolicy, FloatWidth};

        let grammar = Grammar::parse("axiom A; match A then B;").unwrap();
        let ir = IrDocument::from_grammar(&grammar).validate().unwrap();
        let f32_uniform = DerivationSemantics {
            float_width: FloatWidth::F32,
            ambiguous_rules: AmbiguousRulePolicy::Uniform,
        };
        let f64_uniform = DerivationSemantics {
            float_width: FloatWidth::F64,
            ambiguous_rules: AmbiguousRulePolicy::Uniform,
        };
        assert!(
            ir.compatibility_with_semantics(IrBackendProfile::CurrentWgpu, f32_uniform)
                .compatible
        );
        assert!(
            ir.view()
                .compatibility_with_semantics(IrBackendProfile::CurrentCuda, f64_uniform)
                .compatible
        );
        assert!(
            !ir.view()
                .compatibility_with_semantics(IrBackendProfile::CurrentWgpu, f64_uniform)
                .compatible
        );
        let first = DerivationSemantics {
            ambiguous_rules: AmbiguousRulePolicy::First,
            ..f64_uniform
        };
        assert!(
            ir.view()
                .compatibility_with_semantics(IrBackendProfile::Cpu, first)
                .compatible
        );
        assert!(
            ir.view()
                .compatibility_with_semantics(IrBackendProfile::CurrentCuda, first)
                .compatible
        );
    }

    #[test]
    fn semantic_trace_uses_the_selected_numeric_width() {
        use crate::execution::DerivationSemantics;
        use crate::grammar::{AmbiguousRulePolicy, CpuBackend, FloatWidth};

        let grammar = Grammar::parse("axiom A(16777216); match A(x) then A(x + 1);").unwrap();
        let ir = IrDocument::from_grammar(&grammar).validate().unwrap();
        let input = CpuBackend::new()
            .float_width(FloatWidth::F32)
            .compile_ir(&ir)
            .unwrap()
            .axiom()
            .clone();
        let semantics = DerivationSemantics {
            float_width: FloatWidth::F32,
            ambiguous_rules: AmbiguousRulePolicy::First,
        };

        let trace = ir
            .trace_rewrite_with_semantics(&input, 0, 7, semantics)
            .unwrap();
        let expected = CpuBackend::new()
            .float_width(FloatWidth::F32)
            .ambiguous_rules(AmbiguousRulePolicy::First)
            .compile_ir(&ir)
            .unwrap()
            .run_with_seed(1, 7)
            .unwrap();

        assert_eq!(trace.generation, expected);
        assert_eq!(trace.generation.to_string(), "A(16777216)");
        assert_eq!(trace.lineage.successor_modules_per_input(), [1]);
    }

    #[test]
    fn invalid_imported_ids_are_rejected_before_conversion() {
        let mut document = IrDocument::from_grammar(&grammar());
        document.axiom.items.push(IrWordItem::Module(IrModuleExpr {
            symbol: SymbolId(u32::MAX),
            arguments: Vec::new(),
        }));
        let errors = document.validate().unwrap_err();
        assert!(
            errors
                .0
                .iter()
                .any(|diagnostic| diagnostic.code == "symbol-reference")
        );
    }

    #[test]
    fn imported_ir_must_also_pass_language_semantics() {
        let mut document =
            IrDocument::from_grammar(&Grammar::parse("axiom A; match A weight 1 then B;").unwrap());
        document.productions[0].weight = Some(IrExpr::Number((-1.0_f64).to_bits()));
        let errors = document.validate().unwrap_err();
        assert!(
            errors
                .0
                .iter()
                .any(|diagnostic| diagnostic.code == "grammar-validation")
        );
    }

    #[test]
    fn disassembly_is_deterministic_and_reports_backends() {
        let ir = IrDocument::from_grammar(&grammar()).validate().unwrap();
        let first = ir.disassemble();
        assert_eq!(first, ir.disassemble());
        assert!(first.contains("expression programs"));
        assert!(first.contains("backend compatibility"));
        assert!(first.contains("load.binding"));
    }

    #[test]
    fn imported_ir_remains_executable() {
        let source = "axiom A(1); match A(x) when x < 3 then A(x + 1);";
        let original = crate::execution::CompiledGrammar::parse(source).unwrap();
        let json = original.to_ir_document().to_json_pretty().unwrap();
        let imported =
            crate::execution::CompiledGrammar::from_ir(IrDocument::from_json(&json).unwrap())
                .unwrap();
        let original_generation = original.grammar().compile_cpu().unwrap().run(2).unwrap();
        let imported_generation = imported.grammar().compile_cpu().unwrap().run(2).unwrap();
        assert_eq!(imported_generation, original_generation);
    }

    #[test]
    fn ir_trace_reports_exact_successor_lineage() {
        let grammar =
            crate::execution::CompiledGrammar::parse("axiom A; match A then B C;").unwrap();
        let program = crate::grammar::CpuBackend::new()
            .compile_ir(grammar.derivation_ir())
            .unwrap();
        let initial = program.run(0).unwrap();
        let trace = grammar
            .derivation_ir()
            .trace_rewrite(&initial, 0, 0, crate::grammar::AmbiguousRulePolicy::Uniform)
            .unwrap();
        assert_eq!(trace.generation.to_string(), "B C");
        assert_eq!(trace.lineage.successor_modules_per_input(), &[2]);
    }

    #[test]
    fn builder_and_typed_queries_expose_validated_entities() {
        let mut builder = IrBuilder::new();
        let a = builder.intern_symbol("A", 0);
        builder.axiom(IrWord {
            items: vec![IrWordItem::Module(IrModuleExpr {
                symbol: a,
                arguments: Vec::new(),
            })],
        });
        builder.push_production(IrProduction {
            bindings: Vec::new(),
            center: IrModulePattern {
                symbol: a,
                arguments: Vec::new(),
            },
            left: None,
            right: None,
            condition: None,
            weight: None,
            successor: IrWord::default(),
        });
        let ir = builder.finish().unwrap();
        assert_eq!(ir.view().symbol(a).unwrap().name, "A");
        assert!(ir.view().production(ProductionId(0)).is_some());
        assert_eq!(
            ir.view().production_groups()[0].productions,
            [ProductionId(0)]
        );
    }
}
