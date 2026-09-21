//! Parser and validated AST for a small, ABOP-inspired L-system language.
//!
//! The supported source shape is:
//!
//! ```text
//! # comments run to the end of the line
//! let NAME = expression;
//! ignore Module...;       # or: only Module...;
//! axiom word;
//!
//! match predecessor
//!     [left context]
//!     [right context]
//!     [when condition]
//!     [weight expression]
//! then successor;
//! ```
//!
//! Production clauses have a fixed order. `nothing` is the empty successor.
//! Reserved words can be used as identifiers by adding one leading backtick,
//! for example `` `left `` or `` `nothing ``.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    str::FromStr,
};

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, line_ending, multispace1, not_line_ending, satisfy},
    combinator::{all_consuming, map, map_res, not, opt, peek, recognize, value, verify},
    error::{self as nom_error, ContextError, ErrorKind, FromExternalError, ParseError},
    multi::{fold_many0, many0, many1, separated_list0},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated},
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Names that have syntactic meaning unless escaped with a leading backtick.
pub const RESERVED_WORDS: &[&str] = &[
    "let", "ignore", "only", "axiom", "match", "left", "right", "when", "weight", "then",
    "nothing", "and", "or", "not", "true", "false",
];

/// A parsed file before semantic validation and normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub items: Vec<Item>,
}

impl Document {
    /// Parse source text without running semantic validation.
    pub fn parse(source: &str) -> Result<Self, SyntaxError> {
        parse_document(source)
    }

    /// Validate and normalize the parsed items into an executable grammar AST.
    pub fn validate(self) -> Result<Grammar, ValidationErrors> {
        Grammar::try_from(self)
    }
}

impl FromStr for Document {
    type Err = SyntaxError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Self::parse(source)
    }
}

/// One top-level declaration or production.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Let(Binding),
    Ignore(Vec<Identifier>),
    Only(Vec<Identifier>),
    Axiom(Word),
    Production(Production),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: Identifier,
    pub value: Expr,
}

/// A semantically validated, normalized grammar.
#[derive(Debug, Clone, PartialEq)]
pub struct Grammar {
    pub bindings: Vec<Binding>,
    pub context_filter: Option<ContextFilter>,
    pub axiom: Word,
    pub productions: Vec<Production>,
}

impl Grammar {
    /// Parse and validate a complete grammar.
    pub fn parse(source: &str) -> Result<Self, GrammarError> {
        let document = Document::parse(source).map_err(GrammarError::Syntax)?;
        document.validate().map_err(GrammarError::Validation)
    }
}

impl FromStr for Grammar {
    type Err = GrammarError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Self::parse(source)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextFilter {
    Ignore(Vec<Identifier>),
    Only(Vec<Identifier>),
}

/// An identifier after keyword escaping has been removed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Identifier(String);

impl Identifier {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for Identifier {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for Identifier {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Production {
    pub center: ModulePattern,
    pub left: Option<PatternWord>,
    pub right: Option<PatternWord>,
    pub condition: Option<Expr>,
    pub weight: Option<Expr>,
    pub successor: Word,
}

/// An ordered sequence of modules and structural branches.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Word(pub Vec<WordItem>);

impl Word {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &WordItem> {
        self.0.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WordItem {
    Module(ModuleExpr),
    Branch(Word),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleExpr {
    pub name: Identifier,
    pub arguments: Vec<Expr>,
}

/// A context pattern can include modules and structural branches.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PatternWord(pub Vec<PatternItem>);

impl PatternWord {
    pub fn iter(&self) -> impl Iterator<Item = &PatternItem> {
        self.0.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternItem {
    Module(ModulePattern),
    Branch(PatternWord),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModulePattern {
    pub name: Identifier,
    pub arguments: Vec<PatternArgument>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternArgument {
    Bind(Identifier),
    Wildcard,
    Literal(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Bool(bool),
    Name(Identifier),
    Call {
        name: Identifier,
        arguments: Vec<Expr>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
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

/// A source syntax failure with an owned location and line snippet.
#[derive(Debug, Clone)]
pub struct SyntaxError {
    pub offset: usize,
    pub line: usize,
    pub column: usize,
    pub context: Vec<&'static str>,
    pub kind: ErrorKind,
    pub line_text: String,
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "syntax error at {}:{} ({:?})",
            self.line, self.column, self.kind,
        )?;

        if !self.context.is_empty() {
            write!(f, " while parsing {}", self.context.join(" -> "))?;
        }

        if !self.line_text.is_empty() {
            write!(
                f,
                "\n{}\n{}^",
                self.line_text,
                " ".repeat(self.column.saturating_sub(1))
            )?;
        }

        Ok(())
    }
}

impl Error for SyntaxError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub message: String,
}

impl ValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationErrors(pub Vec<ValidationError>);

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "grammar validation failed with {} error(s):",
            self.0.len()
        )?;
        for error in &self.0 {
            writeln!(f, "- {error}")?;
        }
        Ok(())
    }
}

impl Error for ValidationErrors {}

#[derive(Debug, Clone)]
pub enum GrammarError {
    Syntax(SyntaxError),
    Validation(ValidationErrors),
}

impl fmt::Display for GrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(error) => error.fmt(f),
            Self::Validation(error) => error.fmt(f),
        }
    }
}

impl Error for GrammarError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Syntax(error) => Some(error),
            Self::Validation(error) => Some(error),
        }
    }
}

impl TryFrom<Document> for Grammar {
    type Error = ValidationErrors;

    fn try_from(document: Document) -> Result<Self, Self::Error> {
        let mut errors = Vec::new();
        let mut bindings = Vec::new();
        let mut binding_names = BTreeSet::new();
        let mut context_filter: Option<ContextFilter> = None;
        let mut axiom: Option<Word> = None;
        let mut productions = Vec::new();

        for item in document.items {
            match item {
                Item::Let(binding) => {
                    if !binding_names.insert(binding.name.clone()) {
                        errors.push(ValidationError::new(format!(
                            "duplicate binding `{}`",
                            binding.name,
                        )));
                    }
                    bindings.push(binding);
                }
                Item::Ignore(names) => {
                    if context_filter.is_some() {
                        errors.push(ValidationError::new(
                            "only one `ignore` or `only` declaration is permitted",
                        ));
                    } else {
                        context_filter = Some(ContextFilter::Ignore(names));
                    }
                }
                Item::Only(names) => {
                    if context_filter.is_some() {
                        errors.push(ValidationError::new(
                            "only one `ignore` or `only` declaration is permitted",
                        ));
                    } else {
                        context_filter = Some(ContextFilter::Only(names));
                    }
                }
                Item::Axiom(word) => {
                    if axiom.replace(word).is_some() {
                        errors.push(ValidationError::new(
                            "a grammar must contain exactly one `axiom` declaration",
                        ));
                    }
                }
                Item::Production(production) => productions.push(production),
            }
        }

        let Some(axiom) = axiom else {
            errors.push(ValidationError::new(
                "a grammar must contain exactly one `axiom` declaration",
            ));
            return Err(ValidationErrors(errors));
        };

        let globals: BTreeSet<_> = bindings
            .iter()
            .map(|binding| binding.name.clone())
            .collect();

        validate_binding_references(&bindings, &globals, &mut errors);
        validate_axiom_references(&axiom, &globals, &mut errors);
        // ABOP grammars occasionally overload a module name by arity (for example
        // `G(s,r)` and `G(s,r,t)` in the rose-leaf model). The executable identity
        // is therefore the pair (name, arity), not the bare name.

        for (index, production) in productions.iter().enumerate() {
            validate_production(index, production, &globals, &mut errors);
        }

        if errors.is_empty() {
            Ok(Self {
                bindings,
                context_filter,
                axiom,
                productions,
            })
        } else {
            Err(ValidationErrors(errors))
        }
    }
}

fn validate_binding_references(
    bindings: &[Binding],
    globals: &BTreeSet<Identifier>,
    errors: &mut Vec<ValidationError>,
) {
    for binding in bindings {
        let mut references = BTreeSet::new();
        collect_expr_names(&binding.value, &mut references);
        for reference in references {
            if !globals.contains(&reference) {
                errors.push(ValidationError::new(format!(
                    "binding `{}` references unknown name `{reference}`",
                    binding.name,
                )));
            }
        }
    }
}

fn validate_axiom_references(
    axiom: &Word,
    globals: &BTreeSet<Identifier>,
    errors: &mut Vec<ValidationError>,
) {
    fn visit(word: &Word, globals: &BTreeSet<Identifier>, errors: &mut Vec<ValidationError>) {
        for item in &word.0 {
            match item {
                WordItem::Module(module) => {
                    for argument in &module.arguments {
                        let mut names = BTreeSet::new();
                        collect_expr_names(argument, &mut names);
                        for name in names {
                            if !globals.contains(&name) {
                                errors.push(ValidationError::new(format!(
                                    "axiom references unknown name `{name}`",
                                )));
                            }
                        }
                    }
                }
                WordItem::Branch(branch) => visit(branch, globals, errors),
            }
        }
    }

    visit(axiom, globals, errors);
}

fn validate_production(
    index: usize,
    production: &Production,
    globals: &BTreeSet<Identifier>,
    errors: &mut Vec<ValidationError>,
) {
    let rule_number = index + 1;
    let mut locals = BTreeSet::new();

    collect_pattern_bindings(
        std::iter::once(&production.center),
        rule_number,
        &mut locals,
        errors,
    );

    if let Some(left) = &production.left {
        let patterns = pattern_modules(left);
        collect_pattern_bindings(patterns, rule_number, &mut locals, errors);
    }

    if let Some(right) = &production.right {
        let patterns = pattern_modules(right);
        collect_pattern_bindings(patterns, rule_number, &mut locals, errors);
    }

    if let Some(condition) = &production.condition {
        validate_expr_scope(
            condition,
            rule_number,
            "condition",
            &locals,
            globals,
            errors,
        );
    }

    if let Some(weight) = &production.weight {
        validate_expr_scope(weight, rule_number, "weight", &locals, globals, errors);

        if matches!(weight, Expr::Number(value) if *value <= 0.0) {
            errors.push(ValidationError::new(format!(
                "production {rule_number} has a non-positive literal weight",
            )));
        }
    }

    validate_word_expr_scope(&production.successor, rule_number, &locals, globals, errors);
}

fn collect_pattern_bindings<'a>(
    patterns: impl IntoIterator<Item = &'a ModulePattern>,
    rule_number: usize,
    locals: &mut BTreeSet<Identifier>,
    errors: &mut Vec<ValidationError>,
) {
    for pattern in patterns {
        for argument in &pattern.arguments {
            if let PatternArgument::Bind(name) = argument
                && !locals.insert(name.clone())
            {
                errors.push(ValidationError::new(format!(
                    "production {rule_number} binds `{name}` more than once",
                )));
            }
        }
    }
}

fn pattern_modules(word: &PatternWord) -> Vec<&ModulePattern> {
    fn visit<'a>(word: &'a PatternWord, output: &mut Vec<&'a ModulePattern>) {
        for item in &word.0 {
            match item {
                PatternItem::Module(module) => output.push(module),
                PatternItem::Branch(branch) => visit(branch, output),
            }
        }
    }

    let mut output = Vec::new();
    visit(word, &mut output);
    output
}

fn validate_expr_scope(
    expression: &Expr,
    rule_number: usize,
    role: &str,
    locals: &BTreeSet<Identifier>,
    globals: &BTreeSet<Identifier>,
    errors: &mut Vec<ValidationError>,
) {
    let mut names = BTreeSet::new();
    collect_expr_names(expression, &mut names);
    for name in names {
        if !locals.contains(&name) && !globals.contains(&name) {
            errors.push(ValidationError::new(format!(
                "production {rule_number} {role} references unbound name `{name}`",
            )));
        }
    }
}

fn validate_word_expr_scope(
    word: &Word,
    rule_number: usize,
    locals: &BTreeSet<Identifier>,
    globals: &BTreeSet<Identifier>,
    errors: &mut Vec<ValidationError>,
) {
    for item in &word.0 {
        match item {
            WordItem::Module(module) => {
                for argument in &module.arguments {
                    validate_expr_scope(
                        argument,
                        rule_number,
                        "successor",
                        locals,
                        globals,
                        errors,
                    );
                }
            }
            WordItem::Branch(branch) => {
                validate_word_expr_scope(branch, rule_number, locals, globals, errors);
            }
        }
    }
}

fn collect_expr_names(expression: &Expr, output: &mut BTreeSet<Identifier>) {
    match expression {
        Expr::Number(_) | Expr::Bool(_) => {}
        Expr::Name(name) => {
            output.insert(name.clone());
        }
        Expr::Call { arguments, .. } => {
            for argument in arguments {
                collect_expr_names(argument, output);
            }
        }
        Expr::Unary { operand, .. } => collect_expr_names(operand, output),
        Expr::Binary { left, right, .. } => {
            collect_expr_names(left, output);
            collect_expr_names(right, output);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// nom parser
// -------------------------------------------------------------------------------------------------

type ParseResult<'a, T> = IResult<&'a str, T, NomError<'a>>;

#[derive(Debug, Clone)]
struct NomError<'a> {
    input: &'a str,
    kind: ErrorKind,
    contexts: Vec<&'static str>,
}

impl<'a> NomError<'a> {
    fn new(input: &'a str, kind: ErrorKind) -> Self {
        Self {
            input,
            kind,
            contexts: Vec::new(),
        }
    }

    fn farther(self, other: Self) -> Self {
        if self.input.len() < other.input.len() {
            self
        } else if other.input.len() < self.input.len() {
            other
        } else {
            let mut chosen = self;
            for context in other.contexts {
                if !chosen.contexts.contains(&context) {
                    chosen.contexts.push(context);
                }
            }
            chosen
        }
    }
}

impl<'a> ParseError<&'a str> for NomError<'a> {
    fn from_error_kind(input: &'a str, kind: ErrorKind) -> Self {
        Self::new(input, kind)
    }

    fn append(input: &'a str, kind: ErrorKind, other: Self) -> Self {
        other.farther(Self::new(input, kind))
    }

    fn from_char(input: &'a str, _character: char) -> Self {
        Self::new(input, ErrorKind::Char)
    }

    fn or(self, other: Self) -> Self {
        self.farther(other)
    }
}

impl<'a> ContextError<&'a str> for NomError<'a> {
    fn add_context(input: &'a str, context: &'static str, mut other: Self) -> Self {
        if input.len() >= other.input.len() && !other.contexts.contains(&context) {
            other.contexts.push(context);
        }
        other
    }
}

impl<'a, E> FromExternalError<&'a str, E> for NomError<'a> {
    fn from_external_error(input: &'a str, kind: ErrorKind, _error: E) -> Self {
        Self::new(input, kind)
    }
}

pub fn parse_document(source: &str) -> Result<Document, SyntaxError> {
    match all_consuming(document_parser).parse(source) {
        Ok((_remaining, document)) => Ok(document),
        Err(nom::Err::Error(error) | nom::Err::Failure(error)) => {
            Err(to_syntax_error(source, error))
        }
        Err(nom::Err::Incomplete(_)) => unreachable!("complete parsers never return Incomplete"),
    }
}

fn to_syntax_error(source: &str, error: NomError<'_>) -> SyntaxError {
    let offset = source.len().saturating_sub(error.input.len());
    let line = source[..offset]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1;
    let line_start = source[..offset]
        .rfind('\n')
        .map_or(0, |index| index.saturating_add(1));
    let line_end = source[offset..]
        .find('\n')
        .map_or(source.len(), |index| offset + index);
    let column = source[line_start..offset].chars().count() + 1;
    let line_text = source[line_start..line_end].to_owned();

    let mut context = error.contexts;
    context.reverse();
    context.dedup();

    SyntaxError {
        offset,
        line,
        column,
        context,
        kind: error.kind,
        line_text,
    }
}

fn document_parser(mut input: &str) -> ParseResult<'_, Document> {
    let (next, _) = trivia(input)?;
    input = next;
    let mut items = Vec::new();

    while !input.is_empty() {
        let before = input.len();
        let (next, item) = nom_error::context("top-level item", item).parse(input)?;
        if next.len() >= before {
            return Err(nom::Err::Failure(NomError::new(input, ErrorKind::Many0)));
        }
        items.push(item);
        input = next;
    }

    Ok((input, Document { items }))
}

fn item(input: &str) -> ParseResult<'_, Item> {
    alt((
        map(binding, Item::Let),
        map(ignore_declaration, Item::Ignore),
        map(only_declaration, Item::Only),
        map(axiom_declaration, Item::Axiom),
        map(production, Item::Production),
    ))
    .parse(input)
}

fn binding(input: &str) -> ParseResult<'_, Binding> {
    nom_error::context(
        "let declaration",
        (
            keyword("let"),
            identifier,
            punctuation('='),
            expression,
            punctuation(';'),
        ),
    )
    .map(|(_, name, _, value, _)| Binding { name, value })
    .parse(input)
}

fn ignore_declaration(input: &str) -> ParseResult<'_, Vec<Identifier>> {
    nom_error::context(
        "ignore declaration",
        (keyword("ignore"), many1(identifier), punctuation(';')),
    )
    .map(|(_, names, _)| names)
    .parse(input)
}

fn only_declaration(input: &str) -> ParseResult<'_, Vec<Identifier>> {
    nom_error::context(
        "only declaration",
        (keyword("only"), many1(identifier), punctuation(';')),
    )
    .map(|(_, names, _)| names)
    .parse(input)
}

fn axiom_declaration(input: &str) -> ParseResult<'_, Word> {
    nom_error::context("axiom", (keyword("axiom"), word, punctuation(';')))
        .map(|(_, word, _)| word)
        .parse(input)
}

fn production(input: &str) -> ParseResult<'_, Production> {
    nom_error::context(
        "production",
        (
            preceded(keyword("match"), module_pattern),
            opt(preceded(keyword("left"), pattern_word)),
            opt(preceded(keyword("right"), pattern_word)),
            opt(preceded(keyword("when"), expression)),
            opt(preceded(keyword("weight"), expression)),
            preceded(keyword("then"), successor),
            punctuation(';'),
        ),
    )
    .map(
        |(center, left, right, condition, weight, successor, _)| Production {
            center,
            left,
            right,
            condition,
            weight,
            successor,
        },
    )
    .parse(input)
}

fn successor(input: &str) -> ParseResult<'_, Word> {
    alt((value(Word::default(), keyword("nothing")), word)).parse(input)
}

fn word(input: &str) -> ParseResult<'_, Word> {
    map(many1(word_item), Word).parse(input)
}

fn word_item(input: &str) -> ParseResult<'_, WordItem> {
    alt((
        map(
            delimited(punctuation('['), many0(word_item), punctuation(']')),
            |items| WordItem::Branch(Word(items)),
        ),
        map(module_expression, WordItem::Module),
    ))
    .parse(input)
}

fn module_expression(input: &str) -> ParseResult<'_, ModuleExpr> {
    let (input, name) = identifier(input)?;
    let (input, arguments) = opt(delimited(
        punctuation('('),
        separated_list0(punctuation(','), expression),
        punctuation(')'),
    ))
    .parse(input)?;

    Ok((
        input,
        ModuleExpr {
            name,
            arguments: arguments.unwrap_or_default(),
        },
    ))
}

fn pattern_word(input: &str) -> ParseResult<'_, PatternWord> {
    map(many1(pattern_item), PatternWord).parse(input)
}

fn pattern_item(input: &str) -> ParseResult<'_, PatternItem> {
    alt((
        map(
            delimited(punctuation('['), many0(pattern_item), punctuation(']')),
            |items| PatternItem::Branch(PatternWord(items)),
        ),
        map(module_pattern, PatternItem::Module),
    ))
    .parse(input)
}

fn module_pattern(input: &str) -> ParseResult<'_, ModulePattern> {
    let (input, name) = identifier(input)?;
    let (input, arguments) = opt(delimited(
        punctuation('('),
        separated_list0(punctuation(','), pattern_argument),
        punctuation(')'),
    ))
    .parse(input)?;

    Ok((
        input,
        ModulePattern {
            name,
            arguments: arguments.unwrap_or_default(),
        },
    ))
}

fn pattern_argument(input: &str) -> ParseResult<'_, PatternArgument> {
    alt((
        value(
            PatternArgument::Wildcard,
            lexeme(terminated(
                char('_'),
                not(peek(satisfy(is_identifier_continue))),
            )),
        ),
        map(number, PatternArgument::Literal),
        map(identifier, PatternArgument::Bind),
    ))
    .parse(input)
}

fn expression(input: &str) -> ParseResult<'_, Expr> {
    or_expression(input)
}

fn or_expression(input: &str) -> ParseResult<'_, Expr> {
    let (input, first) = and_expression(input)?;

    fold_many0(
        preceded(keyword("or"), and_expression),
        move || first.clone(),
        |left, right| Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(left),
            right: Box::new(right),
        },
    )
    .parse(input)
}

fn and_expression(input: &str) -> ParseResult<'_, Expr> {
    let (input, first) = not_expression(input)?;

    fold_many0(
        preceded(keyword("and"), not_expression),
        move || first.clone(),
        |left, right| Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(left),
            right: Box::new(right),
        },
    )
    .parse(input)
}

fn not_expression(input: &str) -> ParseResult<'_, Expr> {
    alt((
        map(preceded(keyword("not"), not_expression), |operand| {
            Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            }
        }),
        comparison_expression,
    ))
    .parse(input)
}

fn comparison_expression(input: &str) -> ParseResult<'_, Expr> {
    let (input, left) = additive_expression(input)?;
    let (input, comparison) = opt(pair(comparison_operator, additive_expression)).parse(input)?;

    match comparison {
        None => Ok((input, left)),
        Some((op, right)) => Ok((
            input,
            Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        )),
    }
}

fn comparison_operator(input: &str) -> ParseResult<'_, BinaryOp> {
    alt((
        value(BinaryOp::LessEqual, symbol("<=")),
        value(BinaryOp::GreaterEqual, symbol(">=")),
        value(BinaryOp::NotEqual, symbol("!=")),
        value(BinaryOp::Equal, symbol("=")),
        value(BinaryOp::Less, symbol("<")),
        value(BinaryOp::Greater, symbol(">")),
    ))
    .parse(input)
}

fn additive_expression(input: &str) -> ParseResult<'_, Expr> {
    let (input, first) = multiplicative_expression(input)?;

    fold_many0(
        pair(
            alt((
                value(BinaryOp::Add, punctuation('+')),
                value(BinaryOp::Subtract, punctuation('-')),
            )),
            multiplicative_expression,
        ),
        move || first.clone(),
        |left, (op, right)| Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
    )
    .parse(input)
}

fn multiplicative_expression(input: &str) -> ParseResult<'_, Expr> {
    let (input, first) = unary_expression(input)?;

    fold_many0(
        pair(
            alt((
                value(BinaryOp::Multiply, punctuation('*')),
                value(BinaryOp::Divide, punctuation('/')),
            )),
            unary_expression,
        ),
        move || first.clone(),
        |left, (op, right)| Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
    )
    .parse(input)
}

fn unary_expression(input: &str) -> ParseResult<'_, Expr> {
    alt((
        map(preceded(punctuation('-'), unary_expression), |operand| {
            Expr::Unary {
                op: UnaryOp::Negate,
                operand: Box::new(operand),
            }
        }),
        power_expression,
    ))
    .parse(input)
}

fn power_expression(input: &str) -> ParseResult<'_, Expr> {
    let (input, left) = atom_expression(input)?;
    let (input, right) = opt(preceded(punctuation('^'), unary_expression)).parse(input)?;

    match right {
        Some(right) => Ok((
            input,
            Expr::Binary {
                op: BinaryOp::Power,
                left: Box::new(left),
                right: Box::new(right),
            },
        )),
        None => Ok((input, left)),
    }
}

fn atom_expression(input: &str) -> ParseResult<'_, Expr> {
    alt((
        value(Expr::Bool(true), keyword("true")),
        value(Expr::Bool(false), keyword("false")),
        map(number, Expr::Number),
        delimited(punctuation('('), expression, punctuation(')')),
        name_or_call,
    ))
    .parse(input)
}

fn name_or_call(input: &str) -> ParseResult<'_, Expr> {
    let (input, name) = identifier(input)?;
    let (input, arguments) = opt(delimited(
        punctuation('('),
        separated_list0(punctuation(','), expression),
        punctuation(')'),
    ))
    .parse(input)?;

    match arguments {
        Some(arguments) => Ok((input, Expr::Call { name, arguments })),
        None => Ok((input, Expr::Name(name))),
    }
}

fn number(input: &str) -> ParseResult<'_, f64> {
    map_res(lexeme(recognize_float), str::parse::<f64>).parse(input)
}

fn line_comment(input: &str) -> ParseResult<'_, ()> {
    value((), (char('#'), not_line_ending, opt(line_ending))).parse(input)
}

fn trivia(input: &str) -> ParseResult<'_, ()> {
    value((), many0(alt((value((), multispace1), line_comment)))).parse(input)
}

fn lexeme<'a, O, P>(parser: P) -> impl Parser<&'a str, Output = O, Error = NomError<'a>>
where
    P: Parser<&'a str, Output = O, Error = NomError<'a>>,
{
    delimited(trivia, parser, trivia)
}

fn symbol<'a>(
    expected: &'static str,
) -> impl Parser<&'a str, Output = &'a str, Error = NomError<'a>> {
    lexeme(tag(expected))
}

fn punctuation<'a>(expected: char) -> impl Parser<&'a str, Output = char, Error = NomError<'a>> {
    lexeme(char(expected))
}

fn keyword<'a>(expected: &'static str) -> impl Parser<&'a str, Output = (), Error = NomError<'a>> {
    value(
        (),
        lexeme(terminated(
            tag(expected),
            not(peek(satisfy(is_identifier_continue))),
        )),
    )
}

fn identifier(input: &str) -> ParseResult<'_, Identifier> {
    lexeme(alt((
        map(preceded(char('`'), raw_identifier), Identifier::from),
        map(
            verify(raw_identifier, |name: &str| !RESERVED_WORDS.contains(&name)),
            Identifier::from,
        ),
    )))
    .parse(input)
}

fn raw_identifier(input: &str) -> ParseResult<'_, &str> {
    recognize(pair(
        satisfy(is_identifier_start),
        take_while(is_identifier_continue),
    ))
    .parse(input)
}

fn is_identifier_start(character: char) -> bool {
    character == '_' || character.is_ascii_alphabetic()
}

fn is_identifier_continue(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_basic_dol_system() {
        let grammar = Grammar::parse(
            r#"
                # ABOP's introductory system.
                axiom b;

                match a then a b;
                match b then a;
            "#,
        )
        .expect("grammar should parse");

        assert_eq!(grammar.productions.len(), 2);
        assert_eq!(grammar.axiom.0.len(), 1);
    }

    #[test]
    fn parses_parametric_branches_and_power() {
        let grammar = Grammar::parse(
            r#"
                let R = 1.456;
                let H = (R * R) ^ 0.5;

                axiom A(1);

                match A(s)
                then Forward(s)
                     [ TurnLeft A(s / R) ]
                     [ TurnRight A(s / R) ];
            "#,
        )
        .expect("grammar should parse");

        assert_eq!(grammar.bindings.len(), 2);
        assert_eq!(grammar.productions.len(), 1);
    }

    #[test]
    fn parses_and_validates_anabaena() {
        let grammar = Grammar::parse(
            r#"
                let CH = 900;
                let CT = 0.4;
                let ST = 3.9;

                only F;

                axiom Turn(-90) F(0,0,CH) F(4,1,CH) F(0,0,CH);

                match F(s,t,c)
                    when t = 1 and s >= 6
                then F(s / 3 * 2, 2, c) Move(1) F(s / 3, 1, c);

                match F(s,t,c)
                    when t = 2 and s >= 6
                then F(s / 3, 2, c) Move(1) F(s / 3 * 2, 1, c);

                match F(s,t,c)
                    left F(_,_,k)
                    right F(_,_,r)
                    when s > ST or c > CT
                then F(s + 0.1, t, c + 0.25 * (k + r - 3 * c));

                match F(s,t,c)
                    left F(_,_,_)
                    right F(_,_,_)
                then F(0,0,CH) H(1);

                match H(s)
                    when s < 3
                then H(s * 1.1);
            "#,
        )
        .expect("grammar should parse");

        assert_eq!(grammar.bindings.len(), 3);
        assert_eq!(grammar.productions.len(), 5);
        assert_eq!(
            grammar.context_filter,
            Some(ContextFilter::Only(vec!["F".into()]))
        );
    }

    #[test]
    fn leading_backtick_escapes_keywords() {
        let grammar = Grammar::parse(
            r#"
                axiom `match;
                match `match left `left then `nothing;
            "#,
        )
        .expect("escaped keywords should be identifiers");

        let production = &grammar.productions[0];
        assert_eq!(production.center.name.as_str(), "match");
        assert_eq!(production.successor.0.len(), 1);
    }

    #[test]
    fn nothing_is_distinct_from_escaped_nothing() {
        let deleted = Grammar::parse(
            r#"
                axiom A;
                match A then nothing;
            "#,
        )
        .expect("deletion should parse");
        assert!(deleted.productions[0].successor.is_empty());

        let literal = Grammar::parse(
            r#"
                axiom A;
                match A then `nothing;
            "#,
        )
        .expect("escaped module should parse");
        assert!(!literal.productions[0].successor.is_empty());
    }

    #[test]
    fn fixed_clause_order_is_enforced() {
        let error = Document::parse(
            r#"
                axiom A;
                match A when true left B then C;
            "#,
        )
        .expect_err("left after when should be rejected");

        assert!(error.line >= 2);
    }

    #[test]
    fn rejects_ignore_and_only_together() {
        let error = Grammar::parse(
            r#"
                ignore A;
                only B;
                axiom A;
            "#,
        )
        .expect_err("the filters are mutually exclusive");

        let GrammarError::Validation(errors) = error else {
            panic!("expected a validation error");
        };
        assert!(
            errors
                .0
                .iter()
                .any(|error| error.message.contains("only one"))
        );
    }

    #[test]
    fn rejects_unbound_successor_variables() {
        let error = Grammar::parse(
            r#"
                axiom A(1);
                match A(x) then A(y);
            "#,
        )
        .expect_err("y is unbound");

        let GrammarError::Validation(errors) = error else {
            panic!("expected a validation error");
        };
        assert!(
            errors
                .0
                .iter()
                .any(|error| error.message.contains("unbound name `y`"))
        );
    }

    #[test]
    fn equality_uses_single_equals() {
        let grammar = Grammar::parse(
            r#"
                axiom A(1);
                match A(x) when x = 1 then B;
            "#,
        )
        .expect("single-equals comparison should parse");

        assert!(matches!(
            grammar.productions[0].condition.as_ref(),
            Some(Expr::Binary {
                op: BinaryOp::Equal,
                ..
            })
        ));
    }
}

// -------------------------------------------------------------------------------------------------
// CPU execution backend
// -------------------------------------------------------------------------------------------------

/// A concrete scalar value stored in a generated module or expression environment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Number(f64),
    Bool(bool),
}

/// Floating-point width used by derivation expressions and stored module arguments.
///
/// An `F32` execution rounds literals and every numeric operation to binary32,
/// then widens the exact binary32 value when exposing it through [`Value`].
/// This keeps the public generation representation stable while providing a
/// real same-width reference for device backends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FloatWidth {
    F32,
    #[default]
    F64,
}

impl fmt::Display for FloatWidth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

fn round_number(value: f64, width: FloatWidth) -> f64 {
    match width {
        FloatWidth::F32 => f64::from(value as f32),
        FloatWidth::F64 => value,
    }
}

impl Value {
    pub fn as_number(self) -> Result<f64, ExecutionError> {
        match self {
            Self::Number(value) => Ok(value),
            Self::Bool(_) => Err(ExecutionError::new("expected a number, found a boolean")),
        }
    }

    pub fn as_bool(self) -> Result<bool, ExecutionError> {
        match self {
            Self::Bool(value) => Ok(value),
            // ABOP expressions conventionally use zero for false and non-zero for true.
            Self::Number(value) => Ok(value != 0.0),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(value) => {
                if value.fract() == 0.0 {
                    write!(f, "{value:.0}")
                } else {
                    write!(f, "{value}")
                }
            }
            Self::Bool(value) => write!(f, "{value}"),
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

/// A concrete module in a generated L-system word.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub name: Identifier,
    pub arguments: Vec<Value>,
}

impl Module {
    pub fn new(name: impl Into<Identifier>, arguments: Vec<Value>) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.arguments.is_empty() {
            write!(f, "(")?;
            for (index, argument) in self.arguments.iter().enumerate() {
                if index != 0 {
                    write!(f, ",")?;
                }
                write!(f, "{argument}")?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// A concrete generation. Branches are structural and remain nested.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Generation(pub Vec<GenerationItem>);

impl Generation {
    pub fn items(&self) -> &[GenerationItem] {
        &self.0
    }

    /// Returns the owned top-level items without recursively dropping nested branches.
    pub fn into_items(mut self) -> Vec<GenerationItem> {
        std::mem::take(&mut self.0)
    }

    pub fn module_count(&self) -> usize {
        let mut modules = 0usize;
        let mut pending = vec![self];
        while let Some(word) = pending.pop() {
            for item in &word.0 {
                match item {
                    GenerationItem::Module(_) => modules = modules.saturating_add(1),
                    GenerationItem::Branch(branch) => pending.push(branch),
                }
            }
        }
        modules
    }

    pub fn item_count(&self) -> usize {
        let mut items = 0usize;
        let mut pending = vec![self];
        while let Some(word) = pending.pop() {
            for item in &word.0 {
                items = items.saturating_add(1);
                if let GenerationItem::Branch(branch) = item {
                    pending.push(branch);
                }
            }
        }
        items
    }

    pub fn max_branch_depth(&self) -> usize {
        let mut maximum = 0usize;
        let mut pending = vec![(self, 0usize)];
        while let Some((word, depth)) = pending.pop() {
            maximum = maximum.max(depth);
            for item in &word.0 {
                if let GenerationItem::Branch(branch) = item {
                    pending.push((branch, depth.saturating_add(1)));
                }
            }
        }
        maximum
    }

    pub fn to_source_string(&self) -> String {
        self.to_string()
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        // The default destructor follows the branch tree recursively and can overflow the call
        // stack for a valid deeply nested generation. Empty each branch before it is dropped.
        let mut pending = Vec::<Generation>::new();
        let mut items = std::mem::take(&mut self.0);
        loop {
            for item in items.drain(..) {
                if let GenerationItem::Branch(branch) = item {
                    if pending.try_reserve(1).is_ok() {
                        pending.push(branch);
                    } else {
                        // Destructors cannot report allocation failure. Leaking only this branch
                        // is preferable to aborting or recursively overflowing during cleanup.
                        std::mem::forget(branch);
                    }
                }
            }
            let Some(mut branch) = pending.pop() else {
                break;
            };
            items = std::mem::take(&mut branch.0);
            // `branch` now drops with an empty item list.
        }
    }
}

impl fmt::Display for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn write_word(word: &Generation, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut first = true;
            for item in &word.0 {
                if !first {
                    write!(f, " ")?;
                }
                first = false;
                match item {
                    GenerationItem::Module(module) => write!(f, "{module}")?,
                    GenerationItem::Branch(branch) => {
                        write!(f, "[ ")?;
                        write_word(branch, f)?;
                        write!(f, " ]")?;
                    }
                }
            }
            Ok(())
        }
        write_word(self, f)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationItem {
    Module(Module),
    Branch(Generation),
}

/// Exact parentage for one completed rewrite.
///
/// Entries are ordered like predecessor modules in a depth-first traversal.
/// Each value is the number of modules emitted by that predecessor's selected
/// successor. Structural branch nodes do not receive entries: modules inside
/// an existing branch are rewritten independently, while branches introduced
/// by a successor are included in that successor's module count.
///
/// This compact representation is sufficient to recover contiguous successor
/// ranges, including empty successors, without retaining production choices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteLineage {
    successor_modules_per_input: Vec<usize>,
}

impl RewriteLineage {
    #[cfg(feature = "wgpu")]
    pub(crate) fn from_successor_module_counts(counts: Vec<usize>) -> Result<Self, ExecutionError> {
        counts.iter().try_fold(0usize, |total, count| {
            total.checked_add(*count).ok_or_else(|| {
                ExecutionError::resource("rewrite lineage module count", "usize overflow")
            })
        })?;
        Ok(Self {
            successor_modules_per_input: counts,
        })
    }

    /// Successor module counts in depth-first predecessor-module order.
    pub fn successor_modules_per_input(&self) -> &[usize] {
        &self.successor_modules_per_input
    }

    /// Number of predecessor modules represented by this rewrite.
    pub fn input_modules(&self) -> usize {
        self.successor_modules_per_input.len()
    }

    /// Checked number of successor modules represented by this rewrite.
    pub fn output_modules(&self) -> Option<usize> {
        self.successor_modules_per_input
            .iter()
            .try_fold(0usize, |total, count| total.checked_add(*count))
    }
}

/// How to handle multiple applicable unweighted productions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AmbiguousRulePolicy {
    /// Select the first applicable production in source order.
    First,
    /// Return an execution error.
    Error,
    /// Select uniformly. This is useful for explicitly opting into an interpretation of partial systems.
    #[default]
    Uniform,
}

impl fmt::Display for AmbiguousRulePolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::First => "First matching rule",
            Self::Error => "Report ambiguity",
            Self::Uniform => "Uniform random choice",
        })
    }
}

/// Caller-selected semantic limits for CPU execution.
///
/// These are output-validity rules, not worker chunk sizes. The unbounded
/// default still permits typed allocation and address-space failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionLimits {
    /// Maximum number of modules in one generation.
    pub max_modules: usize,
    /// Maximum number of modules plus structural branch items.
    pub max_items: usize,
    /// Maximum structural branch nesting depth.
    pub max_branch_depth: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_modules: usize::MAX,
            max_items: usize::MAX,
            max_branch_depth: usize::MAX,
        }
    }
}

impl ExecutionLimits {
    /// No policy limits. Allocator, address-space, and backend constraints still apply.
    pub const fn unbounded() -> Self {
        Self {
            max_modules: usize::MAX,
            max_items: usize::MAX,
            max_branch_depth: usize::MAX,
        }
    }
}

/// Compiler and execution policy for the full-grammar CPU backend.
///
/// Native execution uses a dedicated Rayon pool and all available logical CPUs
/// by default. Wasm execution uses the same deterministic semantics without
/// native threads. Bounded parallel chunks are scheduling units, not generation
/// limits, and cancellation leaves [`CpuState`] unchanged.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend {
    /// Policy for multiple applicable unweighted productions.
    pub ambiguous_rules: AmbiguousRulePolicy,
    /// Numeric semantics used for literals, expressions, arguments, and weights.
    pub float_width: FloatWidth,
    /// Semantic limits applied to the axiom and every generated result.
    pub limits: ExecutionLimits,
    /// Native worker count. `None` lets Rayon use all available logical CPUs.
    pub worker_threads: Option<usize>,
}

impl CpuBackend {
    /// Creates the default full-grammar CPU policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects how ambiguous unweighted productions are resolved.
    pub fn ambiguous_rules(mut self, policy: AmbiguousRulePolicy) -> Self {
        self.ambiguous_rules = policy;
        self
    }

    /// Selects the floating-point semantics used by this program.
    pub fn float_width(mut self, width: FloatWidth) -> Self {
        self.float_width = width;
        self
    }

    /// Applies caller-selected semantic generation limits.
    pub fn limits(mut self, limits: ExecutionLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Uses exactly this many native Rayon workers, with a minimum of one.
    ///
    /// This setting has no effect on deterministic production choices.
    pub fn worker_threads(mut self, worker_threads: usize) -> Self {
        self.worker_threads = Some(worker_threads.max(1));
        self
    }

    /// Compiles a validated grammar for repeated CPU execution.
    pub fn compile(&self, grammar: &Grammar) -> Result<CpuProgram, ExecutionError> {
        CpuProgram::compile(grammar, self.clone())
    }

    /// Compiles validated backend-neutral IR for repeated CPU execution.
    ///
    /// Expression programs for globals, axioms, conditions, weights, and
    /// successors execute from the IR bytecode. Structural matching continues
    /// to use the CPU reference engine's validated pattern representation.
    pub fn compile_ir(&self, ir: &crate::ir::DerivationIr) -> Result<CpuProgram, ExecutionError> {
        CpuProgram::compile_ir(ir, self.clone())
    }
}

impl Grammar {
    /// Compiles this grammar with the default CPU policy.
    pub fn compile_cpu(&self) -> Result<CpuProgram, ExecutionError> {
        CpuBackend::default().compile(self)
    }

    /// Compiles this grammar with an explicit CPU policy.
    pub fn compile_cpu_with(&self, backend: CpuBackend) -> Result<CpuProgram, ExecutionError> {
        backend.compile(self)
    }
}

/// Immutable compiled CPU program reusable across seeds and executions.
#[derive(Debug, Clone)]
pub struct CpuProgram {
    globals: BTreeMap<Identifier, Value>,
    context_filter: CompiledContextFilter,
    axiom: Generation,
    productions: Vec<Production>,
    dispatch: BTreeMap<Identifier, Vec<usize>>,
    wildcard_dispatch: Vec<usize>,
    ambiguous_rules: AmbiguousRulePolicy,
    float_width: FloatWidth,
    limits: ExecutionLimits,
    ir: Option<CpuIrRuntime>,
    #[cfg(not(target_arch = "wasm32"))]
    worker_pool: std::sync::Arc<rayon::ThreadPool>,
}

#[derive(Debug, Clone)]
struct CpuIrRuntime {
    ir: crate::ir::DerivationIr,
    globals: Vec<Value>,
    programs: BTreeMap<crate::ir::IrProgramOwner, crate::ir::ProgramId>,
    float_width: FloatWidth,
}

impl CpuProgram {
    fn compile(grammar: &Grammar, backend: CpuBackend) -> Result<Self, ExecutionError> {
        Self::compile_internal(grammar, backend, None)
    }

    fn compile_ir(
        ir: &crate::ir::DerivationIr,
        backend: CpuBackend,
    ) -> Result<Self, ExecutionError> {
        let float_width = backend.float_width;
        let grammar = ir
            .to_grammar()
            .map_err(|error| ExecutionError::new(error.to_string()))?;
        let globals = ir.view().evaluate_globals_with_width(float_width)?;
        let programs = ir
            .view()
            .programs()
            .iter()
            .map(|program| (program.owner.clone(), program.id))
            .collect();
        Self::compile_internal(
            &grammar,
            backend,
            Some(CpuIrRuntime {
                ir: ir.clone(),
                globals,
                programs,
                float_width,
            }),
        )
    }

    fn compile_internal(
        grammar: &Grammar,
        backend: CpuBackend,
        ir: Option<CpuIrRuntime>,
    ) -> Result<Self, ExecutionError> {
        let globals = if let Some(runtime) = &ir {
            runtime
                .ir
                .view()
                .document()
                .globals
                .iter()
                .zip(runtime.globals.iter().copied())
                .map(|(global, value)| (Identifier::new(global.name.clone()), value))
                .collect()
        } else {
            resolve_bindings_with_width(&grammar.bindings, backend.float_width)?
        };
        let axiom = if let Some(runtime) = &ir {
            evaluate_ir_word(runtime, &runtime.ir.view().document().axiom, None, &[])?
        } else {
            evaluate_word_with_width(
                &grammar.axiom,
                &globals,
                &BTreeMap::new(),
                backend.float_width,
            )?
        };
        check_generation_limits(&axiom, backend.limits)?;

        let context_filter = CompiledContextFilter::from_ast(grammar.context_filter.as_ref());
        let mut dispatch: BTreeMap<Identifier, Vec<usize>> = BTreeMap::new();
        let mut wildcard_dispatch = Vec::new();

        #[cfg(not(target_arch = "wasm32"))]
        let worker_pool = {
            let mut builder =
                rayon::ThreadPoolBuilder::new().thread_name(|index| format!("lsystem-cpu-{index}"));
            if let Some(worker_threads) = backend.worker_threads {
                builder = builder.num_threads(worker_threads);
            }
            std::sync::Arc::new(
                builder
                    .build()
                    .map_err(|error| ExecutionError::resource("native CPU worker pool", error))?,
            )
        };

        for (index, production) in grammar.productions.iter().enumerate() {
            if production.center.name.as_str() == "_" {
                wildcard_dispatch.push(index);
            } else {
                dispatch
                    .entry(production.center.name.clone())
                    .or_default()
                    .push(index);
            }
        }

        Ok(Self {
            globals,
            context_filter,
            axiom,
            productions: grammar.productions.clone(),
            dispatch,
            wildcard_dispatch,
            ambiguous_rules: backend.ambiguous_rules,
            float_width: backend.float_width,
            limits: backend.limits,
            ir,
            #[cfg(not(target_arch = "wasm32"))]
            worker_pool,
        })
    }

    /// Evaluated global bindings captured by this program.
    pub fn globals(&self) -> &BTreeMap<Identifier, Value> {
        &self.globals
    }

    /// Evaluated axiom from which new states start.
    pub fn axiom(&self) -> &Generation {
        &self.axiom
    }

    /// Starts an incremental execution with seed zero.
    pub fn start(&self) -> CpuState {
        self.start_with_seed(0)
    }

    /// Starts an incremental execution with a stable stochastic seed.
    pub fn start_with_seed(&self, seed: u64) -> CpuState {
        CpuState {
            program: std::sync::Arc::new(self.clone()),
            generation: self.axiom.clone(),
            generation_index: 0,
            seed,
        }
    }

    /// Rewrites an externally retained generation once and reports exact
    /// parentage for the selected successors.
    ///
    /// `generation_index` is the number of rewrites that produced `generation`
    /// from this program's axiom. Supplying that index and the original seed is
    /// required for the same stochastic selection as an incremental run. The
    /// input is borrowed and remains unchanged on success, cancellation, or
    /// failure.
    pub fn trace_rewrite_with_control(
        &self,
        generation: &Generation,
        generation_index: u64,
        seed: u64,
        mut is_cancelled: impl FnMut() -> bool,
        mut on_progress: impl FnMut(CpuStepProgress),
    ) -> Result<(Generation, StepStats, RewriteLineage), ExecutionError> {
        let mut observer = CpuStepObserver::new(&mut is_cancelled, &mut on_progress);
        let input = generation_metrics_with_control(
            generation,
            self.limits,
            CpuStepPhase::InspectingInput,
            &mut observer,
        )?;
        let index = GenerationIndex::build(generation, &self.context_filter, input, &mut observer)?;
        let (next, lineage) = rewrite_indexed_generation(
            &index,
            self,
            seed,
            generation_index,
            input.items,
            true,
            &mut observer,
        )?;
        let output = generation_metrics_with_control(
            &next,
            self.limits,
            CpuStepPhase::ValidatingOutput,
            &mut observer,
        )?;
        Ok((
            next,
            StepStats {
                generation: generation_index.saturating_add(1),
                input_modules: input.modules,
                output_modules: output.modules,
                input_items: input.items,
                output_items: output.items,
            },
            lineage.expect("lineage capture was requested for the completed rewrite"),
        ))
    }

    /// Derives `iterations` generations with seed zero.
    pub fn run(&self, iterations: usize) -> Result<Generation, ExecutionError> {
        let mut state = self.start();
        state.advance(iterations)?;
        Ok(state.into_generation())
    }

    /// Derives `iterations` generations with a stable stochastic seed.
    pub fn run_with_seed(
        &self,
        iterations: usize,
        seed: u64,
    ) -> Result<Generation, ExecutionError> {
        let mut state = self.start_with_seed(seed);
        state.advance(iterations)?;
        Ok(state.into_generation())
    }
}

/// Incremental CPU derivation state.
///
/// Each step is transactional: cancellation, evaluation failure, or a resource
/// error preserves the previous generation, generation index, and random key.
#[derive(Debug, Clone)]
pub struct CpuState {
    program: std::sync::Arc<CpuProgram>,
    generation: Generation,
    generation_index: u64,
    seed: u64,
}

impl CpuState {
    /// Returns the last successfully completed generation.
    pub fn generation(&self) -> &Generation {
        &self.generation
    }

    /// Number of successful rewrites after the axiom.
    pub fn generation_index(&self) -> u64 {
        self.generation_index
    }

    /// Consumes the state and returns its last successful generation.
    pub fn into_generation(self) -> Generation {
        self.generation
    }

    /// Rewrites one generation without progress reporting or cancellation.
    pub fn step(&mut self) -> Result<StepStats, ExecutionError> {
        self.step_with_control(|| false, |_| {})
    }

    /// Rewrites one generation with cooperative cancellation and intra-step progress.
    ///
    /// Cancellation is checked while inspecting, indexing, and rewriting the input, so a
    /// caller does not have to wait for an unusually large generation to finish. The state,
    /// including its random-number stream, is unchanged when the operation is cancelled or
    /// fails.
    pub fn step_with_control(
        &mut self,
        mut is_cancelled: impl FnMut() -> bool,
        mut on_progress: impl FnMut(CpuStepProgress),
    ) -> Result<StepStats, ExecutionError> {
        self.step_impl(&mut is_cancelled, &mut on_progress, false)
            .map(|(stats, _)| stats)
    }

    /// Rewrites one generation and returns the exact selected-successor
    /// lineage used to produce it.
    ///
    /// This is the opt-in form for consumers that animate or otherwise track
    /// modules across adjacent generations. It has the same transactional and
    /// cooperative-cancellation guarantees as [`Self::step_with_control`].
    pub fn step_with_lineage_and_control(
        &mut self,
        mut is_cancelled: impl FnMut() -> bool,
        mut on_progress: impl FnMut(CpuStepProgress),
    ) -> Result<(StepStats, RewriteLineage), ExecutionError> {
        let (stats, lineage) = self.step_impl(&mut is_cancelled, &mut on_progress, true)?;
        Ok((
            stats,
            lineage.expect("lineage capture was requested for the completed rewrite"),
        ))
    }

    fn step_impl(
        &mut self,
        is_cancelled: &mut dyn FnMut() -> bool,
        on_progress: &mut dyn FnMut(CpuStepProgress),
        capture_lineage: bool,
    ) -> Result<(StepStats, Option<RewriteLineage>), ExecutionError> {
        let old = std::mem::take(&mut self.generation);
        let mut observer = CpuStepObserver::new(is_cancelled, on_progress);

        // Keep the immutable index scoped to the rewrite. This makes the update transactional:
        // the old generation can be restored if selection, evaluation, or limit checking fails.
        let result = (|| {
            let input = generation_metrics_with_control(
                &old,
                self.program.limits,
                CpuStepPhase::InspectingInput,
                &mut observer,
            )?;
            let index =
                GenerationIndex::build(&old, &self.program.context_filter, input, &mut observer)?;
            let (next, lineage) = rewrite_indexed_generation(
                &index,
                &self.program,
                self.seed,
                self.generation_index,
                input.items,
                capture_lineage,
                &mut observer,
            )?;
            let output = generation_metrics_with_control(
                &next,
                self.program.limits,
                CpuStepPhase::ValidatingOutput,
                &mut observer,
            )?;
            Ok((next, lineage, input, output))
        })();

        match result {
            Ok((next, lineage, input, output)) => {
                self.generation = next;
                self.generation_index = self.generation_index.saturating_add(1);
                Ok((
                    StepStats {
                        generation: self.generation_index,
                        input_modules: input.modules,
                        output_modules: output.modules,
                        input_items: input.items,
                        output_items: output.items,
                    },
                    lineage,
                ))
            }
            Err(error) => {
                self.generation = old;
                Err(error)
            }
        }
    }

    /// Rewrites `iterations` generations sequentially.
    ///
    /// Each individual step remains transactional.
    pub fn advance(&mut self, iterations: usize) -> Result<(), ExecutionError> {
        for _ in 0..iterations {
            self.step()?;
        }
        Ok(())
    }
}

/// Exact counts around one completed CPU or CUDA rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepStats {
    /// Generation index after the rewrite.
    pub generation: u64,
    /// Module count before the rewrite.
    pub input_modules: usize,
    /// Module count after the rewrite.
    pub output_modules: usize,
    /// Module-plus-branch-item count before the rewrite.
    pub input_items: usize,
    /// Module-plus-branch-item count after the rewrite.
    pub output_items: usize,
}

/// Phase of a bounded CPU rewrite progress update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuStepPhase {
    /// Traverse and validate the current generation.
    InspectingInput,
    /// Build the context lookup structure used by matching rules.
    Indexing,
    /// Select and evaluate one successor for each input module.
    SelectingProductions,
    /// Assemble selected successors and structural branches in source order.
    Rewriting,
    /// Traverse and validate the candidate output before committing it.
    ValidatingOutput,
}

/// Progress within one CPU rewrite phase.
///
/// Counts restart when the phase changes. `total_items` is absent until the
/// backend can know the phase total without an additional full traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStepProgress {
    /// Rewrite phase currently running.
    pub phase: CpuStepPhase,
    /// Completed work units within this phase.
    pub completed_items: usize,
    /// Total work units for this phase, once known.
    pub total_items: Option<usize>,
}

const CPU_PROGRESS_INTERVAL: usize = 4_096;

struct CpuStepObserver<'a> {
    is_cancelled: &'a mut dyn FnMut() -> bool,
    on_progress: &'a mut dyn FnMut(CpuStepProgress),
    last_reported: usize,
    last_phase: Option<CpuStepPhase>,
}

impl<'a> CpuStepObserver<'a> {
    fn new(
        is_cancelled: &'a mut dyn FnMut() -> bool,
        on_progress: &'a mut dyn FnMut(CpuStepProgress),
    ) -> Self {
        Self {
            is_cancelled,
            on_progress,
            last_reported: 0,
            last_phase: None,
        }
    }

    fn checkpoint(
        &mut self,
        phase: CpuStepPhase,
        completed_items: usize,
        total_items: Option<usize>,
        force_report: bool,
    ) -> Result<(), ExecutionError> {
        if (self.is_cancelled)() {
            return Err(ExecutionError::cancelled());
        }
        let phase_changed = self.last_phase != Some(phase);
        if force_report
            || phase_changed
            || completed_items.saturating_sub(self.last_reported) >= CPU_PROGRESS_INTERVAL
        {
            (self.on_progress)(CpuStepProgress {
                phase,
                completed_items,
                total_items,
            });
            self.last_reported = completed_items;
            self.last_phase = Some(phase);
        }
        Ok(())
    }

    fn check_cancelled(&mut self) -> Result<(), ExecutionError> {
        if (self.is_cancelled)() {
            Err(ExecutionError::cancelled())
        } else {
            Ok(())
        }
    }
}

/// Machine-readable category of a CPU execution failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionErrorKind {
    /// An expression or production could not be evaluated.
    Evaluation,
    /// A caller-selected semantic limit was exceeded.
    LimitExceeded {
        resource: &'static str,
        actual: usize,
        limit: usize,
    },
    /// Host allocation or address-space capacity was exhausted.
    ResourceExhausted { resource: &'static str },
    /// Cooperative cancellation was observed.
    Cancelled,
}

/// Typed CPU execution failure with a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionError {
    /// Human-readable diagnostic suitable for display.
    pub message: String,
    kind: ExecutionErrorKind,
}

impl ExecutionError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ExecutionErrorKind::Evaluation,
        }
    }

    fn limit(resource: &'static str, actual: usize, limit: usize) -> Self {
        Self {
            message: format!("{resource} count {actual} exceeds configured limit {limit}"),
            kind: ExecutionErrorKind::LimitExceeded {
                resource,
                actual,
                limit,
            },
        }
    }

    pub(crate) fn resource(resource: &'static str, error: impl fmt::Display) -> Self {
        Self {
            message: format!("unable to allocate {resource}: {error}"),
            kind: ExecutionErrorKind::ResourceExhausted { resource },
        }
    }

    fn cancelled() -> Self {
        Self {
            message: "calculation cancelled".to_string(),
            kind: ExecutionErrorKind::Cancelled,
        }
    }

    /// Returns the stable machine-readable error category.
    pub fn kind(&self) -> &ExecutionErrorKind {
        &self.kind
    }

    /// Reports whether this failure was cooperative cancellation.
    pub fn is_cancelled(&self) -> bool {
        matches!(self.kind, ExecutionErrorKind::Cancelled)
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ExecutionError {}

#[derive(Debug, Clone)]
enum CompiledContextFilter {
    All,
    Ignore(BTreeSet<Identifier>),
    Only(BTreeSet<Identifier>),
}

impl CompiledContextFilter {
    fn from_ast(filter: Option<&ContextFilter>) -> Self {
        match filter {
            None => Self::All,
            Some(ContextFilter::Ignore(names)) => Self::Ignore(names.iter().cloned().collect()),
            Some(ContextFilter::Only(names)) => Self::Only(names.iter().cloned().collect()),
        }
    }

    fn is_visible(&self, name: &Identifier) -> bool {
        match self {
            Self::All => true,
            Self::Ignore(names) => !names.contains(name),
            Self::Only(names) => names.contains(name),
        }
    }
}

fn resolve_bindings_with_width(
    bindings: &[Binding],
    float_width: FloatWidth,
) -> Result<BTreeMap<Identifier, Value>, ExecutionError> {
    let mut resolved = BTreeMap::new();
    let mut pending: Vec<&Binding> = bindings.iter().collect();

    while !pending.is_empty() {
        let mut next_pending = Vec::new();
        let mut made_progress = false;

        for binding in pending {
            match evaluate_expr_with_width(&binding.value, &resolved, &BTreeMap::new(), float_width)
            {
                Ok(value) => {
                    resolved.insert(binding.name.clone(), value);
                    made_progress = true;
                }
                Err(error) if error.message.starts_with("unknown name `") => {
                    next_pending.push(binding);
                }
                Err(error) => {
                    return Err(ExecutionError::new(format!(
                        "failed to evaluate binding `{}`: {}",
                        binding.name, error,
                    )));
                }
            }
        }

        if !made_progress {
            let names = next_pending
                .iter()
                .map(|binding| binding.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ExecutionError::new(format!(
                "cyclic or unresolved `let` bindings: {names}",
            )));
        }

        pending = next_pending;
    }

    Ok(resolved)
}

#[cfg(any(all(feature = "cuda", not(target_arch = "wasm32")), feature = "wgpu"))]
pub(crate) fn resolve_global_bindings_with_width(
    bindings: &[Binding],
    float_width: FloatWidth,
) -> Result<BTreeMap<Identifier, Value>, ExecutionError> {
    resolve_bindings_with_width(bindings, float_width)
}

#[cfg(any(all(feature = "cuda", not(target_arch = "wasm32")), feature = "wgpu"))]
pub(crate) fn evaluate_constant_expression_with_width(
    expression: &Expr,
    globals: &BTreeMap<Identifier, Value>,
    float_width: FloatWidth,
) -> Result<Value, ExecutionError> {
    evaluate_expr_with_width(expression, globals, &BTreeMap::new(), float_width)
}

type Environment = BTreeMap<Identifier, Value>;

fn evaluate_expr_with_width(
    expression: &Expr,
    globals: &Environment,
    locals: &Environment,
    float_width: FloatWidth,
) -> Result<Value, ExecutionError> {
    match expression {
        Expr::Number(value) => Ok(Value::Number(round_number(*value, float_width))),
        Expr::Bool(value) => Ok(Value::Bool(*value)),
        Expr::Name(name) => locals
            .get(name)
            .or_else(|| globals.get(name))
            .copied()
            .ok_or_else(|| ExecutionError::new(format!("unknown name `{name}`"))),
        Expr::Call { name, arguments } => {
            let values = arguments
                .iter()
                .map(|argument| evaluate_expr_with_width(argument, globals, locals, float_width))
                .collect::<Result<Vec<_>, _>>()?;
            evaluate_builtin_with_width(name, &values, float_width)
        }
        Expr::Unary { op, operand } => {
            let operand = evaluate_expr_with_width(operand, globals, locals, float_width)?;
            evaluate_unary_with_width(*op, operand, float_width)
        }
        Expr::Binary { op, left, right } => {
            match op {
                BinaryOp::And => {
                    let left =
                        evaluate_expr_with_width(left, globals, locals, float_width)?.as_bool()?;
                    if !left {
                        return Ok(Value::Bool(false));
                    }
                    return Ok(Value::Bool(
                        evaluate_expr_with_width(right, globals, locals, float_width)?.as_bool()?,
                    ));
                }
                BinaryOp::Or => {
                    let left =
                        evaluate_expr_with_width(left, globals, locals, float_width)?.as_bool()?;
                    if left {
                        return Ok(Value::Bool(true));
                    }
                    return Ok(Value::Bool(
                        evaluate_expr_with_width(right, globals, locals, float_width)?.as_bool()?,
                    ));
                }
                _ => {}
            }

            let left = evaluate_expr_with_width(left, globals, locals, float_width)?;
            let right = evaluate_expr_with_width(right, globals, locals, float_width)?;
            evaluate_binary_with_width(*op, left, right, float_width)
        }
    }
}

pub(crate) fn evaluate_unary_with_width(
    op: UnaryOp,
    operand: Value,
    float_width: FloatWidth,
) -> Result<Value, ExecutionError> {
    match op {
        UnaryOp::Negate => Ok(Value::Number(round_number(
            -operand.as_number()?,
            float_width,
        ))),
        UnaryOp::Not => Ok(Value::Bool(!operand.as_bool()?)),
    }
}

pub(crate) fn evaluate_binary_with_width(
    op: BinaryOp,
    left: Value,
    right: Value,
    float_width: FloatWidth,
) -> Result<Value, ExecutionError> {
    let numeric = |operation: fn(f64, f64) -> f64, operation_f32: fn(f32, f32) -> f32| {
        let left = left.as_number()?;
        let right = right.as_number()?;
        Ok(Value::Number(match float_width {
            FloatWidth::F32 => f64::from(operation_f32(left as f32, right as f32)),
            FloatWidth::F64 => operation(left, right),
        }))
    };
    let compare = |operation: fn(f64, f64) -> bool, operation_f32: fn(f32, f32) -> bool| {
        let left = left.as_number()?;
        let right = right.as_number()?;
        Ok(Value::Bool(match float_width {
            FloatWidth::F32 => operation_f32(left as f32, right as f32),
            FloatWidth::F64 => operation(left, right),
        }))
    };

    match op {
        BinaryOp::Add => numeric(|x, y| x + y, |x, y| x + y),
        BinaryOp::Subtract => numeric(|x, y| x - y, |x, y| x - y),
        BinaryOp::Multiply => numeric(|x, y| x * y, |x, y| x * y),
        BinaryOp::Divide => numeric(|x, y| x / y, |x, y| x / y),
        BinaryOp::Power => numeric(f64::powf, f32::powf),
        BinaryOp::Equal => Ok(Value::Bool(left == right)),
        BinaryOp::NotEqual => Ok(Value::Bool(left != right)),
        BinaryOp::Less => compare(|x, y| x < y, |x, y| x < y),
        BinaryOp::LessEqual => compare(|x, y| x <= y, |x, y| x <= y),
        BinaryOp::Greater => compare(|x, y| x > y, |x, y| x > y),
        BinaryOp::GreaterEqual => compare(|x, y| x >= y, |x, y| x >= y),
        BinaryOp::And | BinaryOp::Or => unreachable!("short-circuited above"),
    }
}

pub(crate) fn evaluate_builtin_with_width(
    name: &Identifier,
    arguments: &[Value],
    float_width: FloatWidth,
) -> Result<Value, ExecutionError> {
    fn unary(arguments: &[Value], name: &str) -> Result<f64, ExecutionError> {
        if arguments.len() != 1 {
            return Err(ExecutionError::new(format!(
                "function `{name}` expects 1 argument, got {}",
                arguments.len(),
            )));
        }
        arguments[0].as_number()
    }

    fn binary(arguments: &[Value], name: &str) -> Result<(f64, f64), ExecutionError> {
        if arguments.len() != 2 {
            return Err(ExecutionError::new(format!(
                "function `{name}` expects 2 arguments, got {}",
                arguments.len(),
            )));
        }
        Ok((arguments[0].as_number()?, arguments[1].as_number()?))
    }

    let function = name.as_str();
    let result = match float_width {
        FloatWidth::F64 => match function {
            "abs" => unary(arguments, function)?.abs(),
            "sqrt" => unary(arguments, function)?.sqrt(),
            "cbrt" => unary(arguments, function)?.cbrt(),
            "exp" => unary(arguments, function)?.exp(),
            "exp2" => unary(arguments, function)?.exp2(),
            "ln" | "log" => unary(arguments, function)?.ln(),
            "log2" => unary(arguments, function)?.log2(),
            "log10" => unary(arguments, function)?.log10(),
            "sin" => unary(arguments, function)?.sin(),
            "cos" => unary(arguments, function)?.cos(),
            "tan" => unary(arguments, function)?.tan(),
            "asin" => unary(arguments, function)?.asin(),
            "acos" => unary(arguments, function)?.acos(),
            "atan" => unary(arguments, function)?.atan(),
            "floor" => unary(arguments, function)?.floor(),
            "ceil" => unary(arguments, function)?.ceil(),
            "round" => unary(arguments, function)?.round(),
            "trunc" => unary(arguments, function)?.trunc(),
            "fract" => unary(arguments, function)?.fract(),
            "sign" => unary(arguments, function)?.signum(),
            "atan2" => {
                let (y, x) = binary(arguments, function)?;
                y.atan2(x)
            }
            "pow" => {
                let (x, y) = binary(arguments, function)?;
                x.powf(y)
            }
            "hypot" => {
                let (x, y) = binary(arguments, function)?;
                x.hypot(y)
            }
            "min" => {
                let (x, y) = binary(arguments, function)?;
                x.min(y)
            }
            "max" => {
                let (x, y) = binary(arguments, function)?;
                x.max(y)
            }
            "clamp" => {
                if arguments.len() != 3 {
                    return Err(ExecutionError::new(format!(
                        "function `clamp` expects 3 arguments, got {}",
                        arguments.len(),
                    )));
                }
                arguments[0]
                    .as_number()?
                    .clamp(arguments[1].as_number()?, arguments[2].as_number()?)
            }
            _ => return Err(ExecutionError::new(format!("unknown function `{name}`"))),
        },
        FloatWidth::F32 => {
            let unary =
                |arguments: &[Value], name: &str| unary(arguments, name).map(|value| value as f32);
            let binary = |arguments: &[Value], name: &str| {
                binary(arguments, name).map(|(left, right)| (left as f32, right as f32))
            };
            f64::from(match function {
                "abs" => unary(arguments, function)?.abs(),
                "sqrt" => unary(arguments, function)?.sqrt(),
                "cbrt" => unary(arguments, function)?.cbrt(),
                "exp" => unary(arguments, function)?.exp(),
                "exp2" => unary(arguments, function)?.exp2(),
                "ln" | "log" => unary(arguments, function)?.ln(),
                "log2" => unary(arguments, function)?.log2(),
                "log10" => unary(arguments, function)?.log10(),
                "sin" => unary(arguments, function)?.sin(),
                "cos" => unary(arguments, function)?.cos(),
                "tan" => unary(arguments, function)?.tan(),
                "asin" => unary(arguments, function)?.asin(),
                "acos" => unary(arguments, function)?.acos(),
                "atan" => unary(arguments, function)?.atan(),
                "floor" => unary(arguments, function)?.floor(),
                "ceil" => unary(arguments, function)?.ceil(),
                "round" => unary(arguments, function)?.round(),
                "trunc" => unary(arguments, function)?.trunc(),
                "fract" => unary(arguments, function)?.fract(),
                "sign" => unary(arguments, function)?.signum(),
                "atan2" => {
                    let (y, x) = binary(arguments, function)?;
                    y.atan2(x)
                }
                "pow" => {
                    let (x, y) = binary(arguments, function)?;
                    x.powf(y)
                }
                "hypot" => {
                    let (x, y) = binary(arguments, function)?;
                    x.hypot(y)
                }
                "min" => {
                    let (x, y) = binary(arguments, function)?;
                    x.min(y)
                }
                "max" => {
                    let (x, y) = binary(arguments, function)?;
                    x.max(y)
                }
                "clamp" => {
                    if arguments.len() != 3 {
                        return Err(ExecutionError::new(format!(
                            "function `clamp` expects 3 arguments, got {}",
                            arguments.len(),
                        )));
                    }
                    (arguments[0].as_number()? as f32).clamp(
                        arguments[1].as_number()? as f32,
                        arguments[2].as_number()? as f32,
                    )
                }
                _ => return Err(ExecutionError::new(format!("unknown function `{name}`"))),
            })
        }
    };

    if result.is_nan() {
        return Err(ExecutionError::new(format!(
            "function `{name}` produced NaN",
        )));
    }

    Ok(Value::Number(result))
}

fn evaluate_ir_word(
    runtime: &CpuIrRuntime,
    word: &crate::ir::IrWord,
    production: Option<crate::ir::ProductionId>,
    bindings: &[Value],
) -> Result<Generation, ExecutionError> {
    struct Frame<'a> {
        word: &'a crate::ir::IrWord,
        next: usize,
        output: Vec<GenerationItem>,
    }

    fn frame(word: &crate::ir::IrWord) -> Result<Frame<'_>, ExecutionError> {
        let mut output = Vec::new();
        output
            .try_reserve(word.items.len())
            .map_err(|error| ExecutionError::resource("IR generation items", error))?;
        Ok(Frame {
            word,
            next: 0,
            output,
        })
    }

    let document = runtime.ir.view().document();
    let mut module_index = 0_u32;
    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|error| ExecutionError::resource("IR branch traversal stack", error))?;
    stack.push(frame(word)?);

    loop {
        let Some(current) = stack.last_mut() else {
            unreachable!("IR evaluation stack contains the root until return")
        };
        if current.next == current.word.items.len() {
            let completed = Generation(stack.pop().unwrap().output);
            if let Some(parent) = stack.last_mut() {
                parent.output.push(GenerationItem::Branch(completed));
                continue;
            }
            return Ok(completed);
        }

        let item = &current.word.items[current.next];
        current.next += 1;
        match item {
            crate::ir::IrWordItem::Module(module) => {
                let current_module = module_index;
                module_index = module_index.checked_add(1).ok_or_else(|| {
                    ExecutionError::resource("IR module index", "u32 index overflow")
                })?;
                let mut arguments = Vec::new();
                arguments
                    .try_reserve(module.arguments.len())
                    .map_err(|error| ExecutionError::resource("IR module arguments", error))?;
                for (argument_index, _) in module.arguments.iter().enumerate() {
                    let argument = u32::try_from(argument_index)
                        .map_err(|error| ExecutionError::resource("IR argument index", error))?;
                    let owner = production.map_or(
                        crate::ir::IrProgramOwner::AxiomArgument {
                            module: current_module,
                            argument,
                        },
                        |production| crate::ir::IrProgramOwner::SuccessorArgument {
                            production,
                            module: current_module,
                            argument,
                        },
                    );
                    let program = runtime.programs.get(&owner).ok_or_else(|| {
                        ExecutionError::new(format!("missing IR expression program for {owner:?}"))
                    })?;
                    arguments.push(runtime.ir.view().evaluate_program_with_width(
                        *program,
                        &runtime.globals,
                        bindings,
                        runtime.float_width,
                    )?);
                }
                let symbol = document
                    .symbols
                    .get(module.symbol.0 as usize)
                    .expect("validated IR symbol reference");
                current.output.push(GenerationItem::Module(Module {
                    name: Identifier::new(symbol.name.clone()),
                    arguments,
                }));
            }
            crate::ir::IrWordItem::Branch(branch) => {
                stack.try_reserve(1).map_err(|error| {
                    ExecutionError::resource("IR branch traversal stack", error)
                })?;
                stack.push(frame(branch)?);
            }
        }
    }
}

fn evaluate_word_with_width(
    word: &Word,
    globals: &Environment,
    locals: &Environment,
    float_width: FloatWidth,
) -> Result<Generation, ExecutionError> {
    struct Frame<'a> {
        word: &'a Word,
        next: usize,
        output: Vec<GenerationItem>,
    }

    fn frame(word: &Word) -> Result<Frame<'_>, ExecutionError> {
        let mut output = Vec::new();
        output
            .try_reserve(word.0.len())
            .map_err(|error| ExecutionError::resource("generation items", error))?;
        Ok(Frame {
            word,
            next: 0,
            output,
        })
    }

    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|error| ExecutionError::resource("branch traversal stack", error))?;
    stack.push(frame(word)?);

    loop {
        let Some(current) = stack.last_mut() else {
            unreachable!("evaluation stack always contains the root until return")
        };
        if current.next == current.word.0.len() {
            let completed = Generation(stack.pop().unwrap().output);
            if let Some(parent) = stack.last_mut() {
                parent.output.push(GenerationItem::Branch(completed));
                continue;
            }
            return Ok(completed);
        }

        let item = &current.word.0[current.next];
        current.next += 1;
        match item {
            WordItem::Module(module) => {
                let mut arguments = Vec::new();
                arguments
                    .try_reserve(module.arguments.len())
                    .map_err(|error| ExecutionError::resource("module arguments", error))?;
                for argument in &module.arguments {
                    arguments.push(evaluate_expr_with_width(
                        argument,
                        globals,
                        locals,
                        float_width,
                    )?);
                }
                current.output.push(GenerationItem::Module(Module {
                    name: module.name.clone(),
                    arguments,
                }));
            }
            WordItem::Branch(branch) => {
                stack
                    .try_reserve(1)
                    .map_err(|error| ExecutionError::resource("branch traversal stack", error))?;
                stack.push(frame(branch)?);
            }
        }
    }
}

fn check_generation_limits(
    generation: &Generation,
    limits: ExecutionLimits,
) -> Result<(), ExecutionError> {
    let mut cancelled = || false;
    let mut progress = |_| {};
    let mut observer = CpuStepObserver::new(&mut cancelled, &mut progress);
    generation_metrics_with_control(
        generation,
        limits,
        CpuStepPhase::ValidatingOutput,
        &mut observer,
    )?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct GenerationMetrics {
    modules: usize,
    items: usize,
    branches: usize,
}

fn generation_metrics_with_control(
    generation: &Generation,
    limits: ExecutionLimits,
    phase: CpuStepPhase,
    observer: &mut CpuStepObserver<'_>,
) -> Result<GenerationMetrics, ExecutionError> {
    let mut modules = 0usize;
    let mut items = 0usize;
    let mut branches = 0usize;
    let mut pending = Vec::new();
    pending
        .try_reserve(1)
        .map_err(|error| ExecutionError::resource("generation traversal stack", error))?;
    pending.push((generation, 0usize));
    observer.checkpoint(phase, 0, None, true)?;

    while let Some((word, depth)) = pending.pop() {
        for item in &word.0 {
            items = items.checked_add(1).ok_or_else(|| {
                ExecutionError::resource("generation item count", "usize overflow")
            })?;
            if items > limits.max_items {
                return Err(ExecutionError::limit(
                    "generation items",
                    items,
                    limits.max_items,
                ));
            }
            match item {
                GenerationItem::Module(_) => {
                    modules = modules.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("generation module count", "usize overflow")
                    })?;
                    if modules > limits.max_modules {
                        return Err(ExecutionError::limit(
                            "generation modules",
                            modules,
                            limits.max_modules,
                        ));
                    }
                }
                GenerationItem::Branch(branch) => {
                    branches = branches.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("generation branch count", "usize overflow")
                    })?;
                    let branch_depth = depth.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("generation branch depth", "usize overflow")
                    })?;
                    if branch_depth > limits.max_branch_depth {
                        return Err(ExecutionError::limit(
                            "generation branch depth",
                            branch_depth,
                            limits.max_branch_depth,
                        ));
                    }
                    pending.try_reserve(1).map_err(|error| {
                        ExecutionError::resource("generation traversal stack", error)
                    })?;
                    pending.push((branch, branch_depth));
                }
            }
            observer.checkpoint(phase, items, None, false)?;
        }
    }
    observer.checkpoint(phase, items, Some(items), true)?;
    Ok(GenerationMetrics {
        modules,
        items,
        branches,
    })
}

#[derive(Debug)]
struct GenerationIndex<'a> {
    nodes: Vec<NodeRef<'a>>,
    axes: Vec<Axis>,
    filter: &'a CompiledContextFilter,
}

#[derive(Debug)]
struct NodeRef<'a> {
    module: &'a Module,
    axis: usize,
    raw_index: usize,
    flat_token_position: u64,
}

#[derive(Debug, Clone)]
struct Axis {
    items: Vec<AxisItem>,
    parent_anchor: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum AxisItem {
    Module(usize),
    Branch(usize),
}

impl<'a> GenerationIndex<'a> {
    fn build(
        generation: &'a Generation,
        filter: &'a CompiledContextFilter,
        metrics: GenerationMetrics,
        observer: &mut CpuStepObserver<'_>,
    ) -> Result<Self, ExecutionError> {
        let mut index = Self {
            nodes: Vec::new(),
            axes: Vec::new(),
            filter,
        };
        index
            .nodes
            .try_reserve(metrics.modules)
            .map_err(|error| ExecutionError::resource("generation context index", error))?;
        index
            .axes
            .try_reserve(metrics.branches.saturating_add(1))
            .map_err(|error| ExecutionError::resource("generation branch index", error))?;
        fn axis(parent_anchor: Option<usize>, input_len: usize) -> Result<Axis, ExecutionError> {
            let mut items = Vec::new();
            items
                .try_reserve(input_len)
                .map_err(|error| ExecutionError::resource("generation axis index", error))?;
            Ok(Axis {
                items,
                parent_anchor,
            })
        }

        index.axes.push(axis(None, generation.0.len())?);

        struct Frame<'a> {
            word: &'a Generation,
            axis_id: usize,
            next: usize,
            current_anchor: Option<usize>,
            closes_branch: bool,
        }
        let mut stack = vec![Frame {
            word: generation,
            axis_id: 0,
            next: 0,
            current_anchor: None,
            closes_branch: false,
        }];
        let mut flat_token_position = 0_u64;
        let mut completed = 0usize;
        observer.checkpoint(CpuStepPhase::Indexing, 0, Some(metrics.items), true)?;

        while let Some(frame) = stack.last_mut() {
            if frame.next == frame.word.0.len() {
                let closes_branch = frame.closes_branch;
                stack.pop();
                if closes_branch {
                    flat_token_position = flat_token_position.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("generation token position", "u64 overflow")
                    })?;
                }
                continue;
            }
            let item = &frame.word.0[frame.next];
            frame.next += 1;
            let axis_id = frame.axis_id;
            let current_anchor = frame.current_anchor;
            match item {
                GenerationItem::Module(module) => {
                    let raw_index = index.axes[axis_id].items.len();
                    let node_id = index.nodes.len();
                    index.nodes.push(NodeRef {
                        module,
                        axis: axis_id,
                        raw_index,
                        flat_token_position,
                    });
                    flat_token_position = flat_token_position.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("generation token position", "u64 overflow")
                    })?;
                    index.axes[axis_id].items.push(AxisItem::Module(node_id));
                    frame.current_anchor = Some(node_id);
                }
                GenerationItem::Branch(branch) => {
                    flat_token_position = flat_token_position.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("generation token position", "u64 overflow")
                    })?;
                    let child_axis = index.axes.len();
                    index.axes.push(axis(current_anchor, branch.0.len())?);
                    index.axes[axis_id].items.push(AxisItem::Branch(child_axis));
                    stack.try_reserve(1).map_err(|error| {
                        ExecutionError::resource("generation indexing stack", error)
                    })?;
                    stack.push(Frame {
                        word: branch,
                        axis_id: child_axis,
                        next: 0,
                        current_anchor,
                        closes_branch: true,
                    });
                }
            }
            completed += 1;
            observer.checkpoint(
                CpuStepPhase::Indexing,
                completed,
                Some(metrics.items),
                false,
            )?;
        }
        observer.checkpoint(CpuStepPhase::Indexing, completed, Some(metrics.items), true)?;
        Ok(index)
    }

    fn previous_visible(&self, node_id: usize) -> Option<usize> {
        let mut current = node_id;
        loop {
            let node = &self.nodes[current];
            let axis = &self.axes[node.axis];
            for item in axis.items[..node.raw_index].iter().rev() {
                if let AxisItem::Module(candidate) = *item
                    && self.filter.is_visible(&self.nodes[candidate].module.name)
                {
                    return Some(candidate);
                }
            }

            let anchor = axis.parent_anchor?;
            if self.filter.is_visible(&self.nodes[anchor].module.name) {
                return Some(anchor);
            }
            current = anchor;
        }
    }

    fn next_visible(&self, node_id: usize) -> Option<usize> {
        let node = &self.nodes[node_id];
        let axis = &self.axes[node.axis];
        for item in axis.items[node.raw_index + 1..].iter() {
            if let AxisItem::Module(candidate) = *item
                && self.filter.is_visible(&self.nodes[candidate].module.name)
            {
                return Some(candidate);
            }
        }
        None
    }
}

fn rewrite_indexed_generation(
    index: &GenerationIndex<'_>,
    program: &CpuProgram,
    seed: u64,
    generation_index: u64,
    total_items: usize,
    capture_lineage: bool,
    observer: &mut CpuStepObserver<'_>,
) -> Result<(Generation, Option<RewriteLineage>), ExecutionError> {
    struct Frame {
        axis_id: usize,
        next: usize,
        output: Vec<GenerationItem>,
    }

    fn frame(axis_id: usize, input_len: usize) -> Result<Frame, ExecutionError> {
        let mut output = Vec::new();
        output
            .try_reserve(input_len)
            .map_err(|error| ExecutionError::resource("rewritten generation", error))?;
        Ok(Frame {
            axis_id,
            next: 0,
            output,
        })
    }

    let mut replacements = select_replacements(index, program, seed, generation_index, observer)?;
    let lineage = if capture_lineage {
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(replacements.len())
            .map_err(|error| ExecutionError::resource("rewrite lineage", error))?;
        for replacement in &replacements {
            observer.check_cancelled()?;
            let replacement = replacement
                .as_ref()
                .expect("selected replacements are present before assembly");
            counts.push(checked_generation_module_count(replacement, observer)?);
        }
        Some(RewriteLineage {
            successor_modules_per_input: counts,
        })
    } else {
        None
    };

    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|error| ExecutionError::resource("rewrite traversal stack", error))?;
    stack.push(frame(0, index.axes[0].items.len())?);
    let mut completed = 0usize;
    observer.checkpoint(CpuStepPhase::Rewriting, 0, Some(total_items), true)?;

    loop {
        let Some(current) = stack.last_mut() else {
            unreachable!("rewrite stack always contains the root until return")
        };
        let axis = &index.axes[current.axis_id];
        if current.next == axis.items.len() {
            let completed_axis = Generation(stack.pop().unwrap().output);
            if let Some(parent) = stack.last_mut() {
                parent.output.push(GenerationItem::Branch(completed_axis));
                continue;
            }
            observer.checkpoint(CpuStepPhase::Rewriting, completed, Some(total_items), true)?;
            return Ok((completed_axis, lineage));
        }

        let item = axis.items[current.next];
        current.next += 1;
        match item {
            AxisItem::Module(node_id) => {
                let replacement = replacements[node_id]
                    .take()
                    .expect("each indexed module is assembled exactly once");
                let mut replacement_items = replacement.into_items();
                current
                    .output
                    .try_reserve(replacement_items.len())
                    .map_err(|error| ExecutionError::resource("rewritten generation", error))?;
                current.output.append(&mut replacement_items);
            }
            AxisItem::Branch(child_axis) => {
                stack
                    .try_reserve(1)
                    .map_err(|error| ExecutionError::resource("rewrite traversal stack", error))?;
                stack.push(frame(child_axis, index.axes[child_axis].items.len())?);
            }
        }
        completed += 1;
        observer.checkpoint(CpuStepPhase::Rewriting, completed, Some(total_items), false)?;
    }
}

fn checked_generation_module_count(
    generation: &Generation,
    observer: &mut CpuStepObserver<'_>,
) -> Result<usize, ExecutionError> {
    let mut count = 0usize;
    let mut pending = Vec::new();
    pending
        .try_reserve(1)
        .map_err(|error| ExecutionError::resource("rewrite lineage traversal", error))?;
    pending.push(generation);
    while let Some(word) = pending.pop() {
        for item in word.items() {
            observer.check_cancelled()?;
            match item {
                GenerationItem::Module(_) => {
                    count = count.checked_add(1).ok_or_else(|| {
                        ExecutionError::resource("rewrite lineage module count", "usize overflow")
                    })?;
                }
                GenerationItem::Branch(branch) => {
                    pending.try_reserve(1).map_err(|error| {
                        ExecutionError::resource("rewrite lineage traversal", error)
                    })?;
                    pending.push(branch);
                }
            }
        }
    }
    Ok(count)
}

const CPU_PARALLEL_CHUNK_ITEMS: usize = 1_024;

fn select_replacements(
    index: &GenerationIndex<'_>,
    program: &CpuProgram,
    seed: u64,
    generation_index: u64,
    observer: &mut CpuStepObserver<'_>,
) -> Result<Vec<Option<Generation>>, ExecutionError> {
    let total = index.nodes.len();
    let mut replacements = Vec::new();
    replacements
        .try_reserve(total)
        .map_err(|error| ExecutionError::resource("module replacements", error))?;
    observer.checkpoint(CpuStepPhase::SelectingProductions, 0, Some(total), true)?;

    for start in (0..total).step_by(CPU_PARALLEL_CHUNK_ITEMS) {
        let end = start.saturating_add(CPU_PARALLEL_CHUNK_ITEMS).min(total);

        #[cfg(not(target_arch = "wasm32"))]
        let selected: Vec<Result<Generation, ExecutionError>> = program.worker_pool.install(|| {
            (start..end)
                .into_par_iter()
                .map(|node_id| {
                    let mut rng = position_rng(
                        seed,
                        generation_index,
                        index.nodes[node_id].flat_token_position,
                        program.float_width,
                    );
                    rewrite_module(node_id, index, program, &mut rng)
                })
                .collect()
        });

        #[cfg(target_arch = "wasm32")]
        let selected: Vec<Result<Generation, ExecutionError>> = (start..end)
            .map(|node_id| {
                let mut rng = position_rng(
                    seed,
                    generation_index,
                    index.nodes[node_id].flat_token_position,
                    program.float_width,
                );
                rewrite_module(node_id, index, program, &mut rng)
            })
            .collect();

        for replacement in selected {
            replacements.push(Some(replacement?));
        }
        observer.checkpoint(CpuStepPhase::SelectingProductions, end, Some(total), true)?;
    }

    Ok(replacements)
}

fn position_rng(
    seed: u64,
    generation_index: u64,
    flat_token_position: u64,
    float_width: FloatWidth,
) -> PositionRng {
    PositionRng::new(seed, generation_index, flat_token_position, float_width)
}

struct ApplicableRule {
    production_index: usize,
    bindings: Environment,
    weight: Option<f64>,
}

fn rewrite_module(
    node_id: usize,
    index: &GenerationIndex<'_>,
    program: &CpuProgram,
    rng: &mut PositionRng,
) -> Result<Generation, ExecutionError> {
    let node = &index.nodes[node_id];
    let mut candidate_indices = program
        .dispatch
        .get(&node.module.name)
        .cloned()
        .unwrap_or_default();
    candidate_indices.extend(program.wildcard_dispatch.iter().copied());
    candidate_indices.sort_unstable();

    let mut applicable = Vec::new();
    for production_index in candidate_indices {
        let production = &program.productions[production_index];
        if let Some((bindings, weight)) =
            production_matches(production_index, production, node_id, index, program)?
        {
            applicable.push(ApplicableRule {
                production_index,
                bindings,
                weight,
            });
        }
    }

    if applicable.is_empty() {
        return Ok(Generation(vec![GenerationItem::Module(
            node.module.clone(),
        )]));
    }

    let weighted = applicable
        .iter()
        .filter(|rule| rule.weight.is_some())
        .count();
    let selected = if weighted != 0 {
        if weighted != applicable.len() {
            return Err(ExecutionError::new(format!(
                "module `{}` has both weighted and unweighted applicable productions",
                node.module.name,
            )));
        }
        select_weighted(&applicable, rng, program.float_width)?
    } else {
        match applicable.len() {
            1 => 0,
            _ => match program.ambiguous_rules {
                AmbiguousRulePolicy::First => 0,
                AmbiguousRulePolicy::Uniform => rng.index(applicable.len()),
                AmbiguousRulePolicy::Error => {
                    return Err(ExecutionError::new(format!(
                        "module `{}` has {} applicable unweighted productions",
                        node.module.name,
                        applicable.len(),
                    )));
                }
            },
        }
    };

    let selected = &applicable[selected];
    let production = &program.productions[selected.production_index];
    if let Some(runtime) = &program.ir {
        let binding_values =
            ir_binding_values(runtime, selected.production_index, &selected.bindings)?;
        evaluate_ir_word(
            runtime,
            &runtime.ir.view().document().productions[selected.production_index].successor,
            Some(crate::ir::ProductionId(
                u32::try_from(selected.production_index)
                    .map_err(|error| ExecutionError::resource("IR production index", error))?,
            )),
            &binding_values,
        )
    } else {
        evaluate_word_with_width(
            &production.successor,
            &program.globals,
            &selected.bindings,
            program.float_width,
        )
    }
}

fn select_weighted(
    rules: &[ApplicableRule],
    rng: &mut PositionRng,
    float_width: FloatWidth,
) -> Result<usize, ExecutionError> {
    let total = match float_width {
        FloatWidth::F32 => f64::from(rules.iter().fold(0.0_f32, |total, rule| {
            total + rule.weight.unwrap_or(0.0) as f32
        })),
        FloatWidth::F64 => rules.iter().map(|rule| rule.weight.unwrap_or(0.0)).sum(),
    };
    if !total.is_finite() || total <= 0.0 {
        return Err(ExecutionError::new(
            "applicable stochastic production weights do not have a positive finite sum",
        ));
    }

    let mut sample = match float_width {
        FloatWidth::F32 => f64::from(rng.next_f32() * total as f32),
        FloatWidth::F64 => rng.next_f64() * total,
    };
    for (index, rule) in rules.iter().enumerate() {
        sample = match float_width {
            FloatWidth::F32 => f64::from(sample as f32 - rule.weight.unwrap_or(0.0) as f32),
            FloatWidth::F64 => sample - rule.weight.unwrap_or(0.0),
        };
        if sample < 0.0 {
            return Ok(index);
        }
    }
    Ok(rules.len() - 1)
}

fn ir_binding_values(
    runtime: &CpuIrRuntime,
    production_index: usize,
    bindings: &Environment,
) -> Result<Vec<Value>, ExecutionError> {
    let names = &runtime.ir.view().document().productions[production_index].bindings;
    let mut values = Vec::new();
    values
        .try_reserve_exact(names.len())
        .map_err(|error| ExecutionError::resource("IR binding values", error))?;
    for name in names {
        let value = bindings
            .iter()
            .find(|(identifier, _)| identifier.as_str() == name)
            .map(|(_, value)| *value)
            .ok_or_else(|| ExecutionError::new(format!("IR binding `{name}` was not captured")))?;
        values.push(value);
    }
    Ok(values)
}

fn production_matches(
    production_index: usize,
    production: &Production,
    node_id: usize,
    index: &GenerationIndex<'_>,
    program: &CpuProgram,
) -> Result<Option<(Environment, Option<f64>)>, ExecutionError> {
    let mut bindings = BTreeMap::new();
    if !match_module_pattern(
        &production.center,
        index.nodes[node_id].module,
        &mut bindings,
        program.float_width,
    )? {
        return Ok(None);
    }

    if let Some(left) = &production.left
        && !match_left_context(left, node_id, index, &mut bindings, program.float_width)?
    {
        return Ok(None);
    }

    if let Some(right) = &production.right
        && !match_right_context(right, node_id, index, &mut bindings, program.float_width)?
    {
        return Ok(None);
    }

    let production_id = crate::ir::ProductionId(
        u32::try_from(production_index)
            .map_err(|error| ExecutionError::resource("IR production index", error))?,
    );
    let ir_bindings = program
        .ir
        .as_ref()
        .map(|runtime| ir_binding_values(runtime, production_index, &bindings))
        .transpose()?;
    let condition = if let (Some(runtime), Some(binding_values)) = (&program.ir, &ir_bindings) {
        runtime
            .programs
            .get(&crate::ir::IrProgramOwner::Condition(production_id))
            .map(|program_id| {
                runtime.ir.view().evaluate_program_with_width(
                    *program_id,
                    &runtime.globals,
                    binding_values,
                    runtime.float_width,
                )
            })
            .transpose()?
            .map(Value::as_bool)
            .transpose()?
    } else {
        production
            .condition
            .as_ref()
            .map(|condition| {
                evaluate_expr_with_width(
                    condition,
                    &program.globals,
                    &bindings,
                    program.float_width,
                )
            })
            .transpose()?
            .map(Value::as_bool)
            .transpose()?
    };

    if condition == Some(false) {
        return Ok(None);
    }

    let weight = if let (Some(runtime), Some(binding_values)) = (&program.ir, &ir_bindings) {
        runtime
            .programs
            .get(&crate::ir::IrProgramOwner::Weight(production_id))
            .map(|program_id| {
                runtime.ir.view().evaluate_program_with_width(
                    *program_id,
                    &runtime.globals,
                    binding_values,
                    runtime.float_width,
                )
            })
            .transpose()?
            .map(Value::as_number)
            .transpose()?
    } else {
        production
            .weight
            .as_ref()
            .map(|weight| {
                evaluate_expr_with_width(weight, &program.globals, &bindings, program.float_width)
            })
            .transpose()?
            .map(Value::as_number)
            .transpose()?
    };

    if let Some(weight) = weight
        && (!weight.is_finite() || weight <= 0.0)
    {
        return Err(ExecutionError::new(format!(
            "production weight evaluated to invalid value {weight}",
        )));
    }

    Ok(Some((bindings, weight)))
}

fn match_module_pattern(
    pattern: &ModulePattern,
    module: &Module,
    bindings: &mut Environment,
    float_width: FloatWidth,
) -> Result<bool, ExecutionError> {
    if pattern.name.as_str() != "_" && pattern.name != module.name {
        return Ok(false);
    }

    if pattern.name.as_str() == "_" && pattern.arguments.is_empty() {
        return Ok(true);
    }

    if pattern.arguments.len() != module.arguments.len() {
        return Ok(false);
    }

    let checkpoint = bindings.clone();
    for (pattern_argument, actual) in pattern.arguments.iter().zip(&module.arguments) {
        match pattern_argument {
            PatternArgument::Wildcard => {}
            PatternArgument::Literal(expected) => {
                if *actual != Value::Number(round_number(*expected, float_width)) {
                    *bindings = checkpoint;
                    return Ok(false);
                }
            }
            PatternArgument::Bind(name) => {
                if let Some(existing) = bindings.get(name) {
                    if existing != actual {
                        *bindings = checkpoint;
                        return Ok(false);
                    }
                } else {
                    bindings.insert(name.clone(), *actual);
                }
            }
        }
    }
    Ok(true)
}

fn match_left_context(
    pattern: &PatternWord,
    center: usize,
    index: &GenerationIndex<'_>,
    bindings: &mut Environment,
    float_width: FloatWidth,
) -> Result<bool, ExecutionError> {
    if pattern
        .0
        .iter()
        .any(|item| matches!(item, PatternItem::Branch(_)))
    {
        let checkpoint = bindings.clone();
        let node = &index.nodes[center];
        let matched = match_pattern_ending_at_axis_position(
            pattern,
            node.axis,
            node.raw_index,
            index,
            bindings,
            float_width,
        )?;
        if !matched {
            *bindings = checkpoint;
        }
        return Ok(matched);
    }

    let checkpoint = bindings.clone();
    let mut current = center;
    for item in pattern.0.iter().rev() {
        let PatternItem::Module(module_pattern) = item else {
            unreachable!();
        };
        let Some(candidate) = index.previous_visible(current) else {
            *bindings = checkpoint;
            return Ok(false);
        };
        if !match_module_pattern(
            module_pattern,
            index.nodes[candidate].module,
            bindings,
            float_width,
        )? {
            *bindings = checkpoint;
            return Ok(false);
        }
        current = candidate;
    }
    Ok(true)
}

fn match_right_context(
    pattern: &PatternWord,
    center: usize,
    index: &GenerationIndex<'_>,
    bindings: &mut Environment,
    float_width: FloatWidth,
) -> Result<bool, ExecutionError> {
    if pattern
        .0
        .iter()
        .any(|item| matches!(item, PatternItem::Branch(_)))
    {
        let checkpoint = bindings.clone();
        let node = &index.nodes[center];
        let matched = match_pattern_from_axis_position(
            pattern,
            node.axis,
            node.raw_index + 1,
            index,
            bindings,
            float_width,
        )?;
        if !matched {
            *bindings = checkpoint;
        }
        return Ok(matched);
    }

    let checkpoint = bindings.clone();
    let mut current = center;
    for item in &pattern.0 {
        let PatternItem::Module(module_pattern) = item else {
            unreachable!();
        };
        let Some(candidate) = index.next_visible(current) else {
            *bindings = checkpoint;
            return Ok(false);
        };
        if !match_module_pattern(
            module_pattern,
            index.nodes[candidate].module,
            bindings,
            float_width,
        )? {
            *bindings = checkpoint;
            return Ok(false);
        }
        current = candidate;
    }
    Ok(true)
}

fn match_pattern_from_axis_position(
    pattern: &PatternWord,
    axis_id: usize,
    mut raw_position: usize,
    index: &GenerationIndex<'_>,
    bindings: &mut Environment,
    float_width: FloatWidth,
) -> Result<bool, ExecutionError> {
    let axis = &index.axes[axis_id];
    for pattern_item in &pattern.0 {
        match pattern_item {
            PatternItem::Module(module_pattern) => {
                let mut found = None;
                while raw_position < axis.items.len() {
                    match axis.items[raw_position] {
                        AxisItem::Module(candidate) => {
                            raw_position += 1;
                            if index.filter.is_visible(&index.nodes[candidate].module.name) {
                                found = Some(candidate);
                                break;
                            }
                        }
                        // A side branch is skipped when the pattern asks for the next axial module.
                        AxisItem::Branch(_) => raw_position += 1,
                    }
                }
                let Some(candidate) = found else {
                    return Ok(false);
                };
                if !match_module_pattern(
                    module_pattern,
                    index.nodes[candidate].module,
                    bindings,
                    float_width,
                )? {
                    return Ok(false);
                }
            }
            PatternItem::Branch(branch_pattern) => {
                // Permit ignored modules before an explicitly requested branch, but not a visible
                // axial module, because that would change the branch attachment point.
                let branch_axis = loop {
                    if raw_position >= axis.items.len() {
                        return Ok(false);
                    }
                    match axis.items[raw_position] {
                        AxisItem::Branch(branch_axis) => {
                            raw_position += 1;
                            break branch_axis;
                        }
                        AxisItem::Module(candidate)
                            if !index.filter.is_visible(&index.nodes[candidate].module.name) =>
                        {
                            raw_position += 1;
                        }
                        AxisItem::Module(_) => return Ok(false),
                    }
                };
                if !match_pattern_from_axis_position(
                    branch_pattern,
                    branch_axis,
                    0,
                    index,
                    bindings,
                    float_width,
                )? {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn match_pattern_ending_at_axis_position(
    pattern: &PatternWord,
    axis_id: usize,
    mut raw_position: usize,
    index: &GenerationIndex<'_>,
    bindings: &mut Environment,
    float_width: FloatWidth,
) -> Result<bool, ExecutionError> {
    let axis = &index.axes[axis_id];
    for pattern_item in pattern.0.iter().rev() {
        match pattern_item {
            PatternItem::Module(module_pattern) => {
                let mut found = None;
                while raw_position > 0 {
                    raw_position -= 1;
                    match axis.items[raw_position] {
                        AxisItem::Module(candidate) => {
                            if index.filter.is_visible(&index.nodes[candidate].module.name) {
                                found = Some(candidate);
                                break;
                            }
                        }
                        // Side branches do not interrupt an axial path.
                        AxisItem::Branch(_) => {}
                    }
                }
                let Some(candidate) = found else {
                    return Ok(false);
                };
                if !match_module_pattern(
                    module_pattern,
                    index.nodes[candidate].module,
                    bindings,
                    float_width,
                )? {
                    return Ok(false);
                }
            }
            PatternItem::Branch(branch_pattern) => {
                let branch_axis = loop {
                    if raw_position == 0 {
                        return Ok(false);
                    }
                    raw_position -= 1;
                    match axis.items[raw_position] {
                        AxisItem::Branch(branch_axis) => break branch_axis,
                        AxisItem::Module(candidate)
                            if !index.filter.is_visible(&index.nodes[candidate].module.name) => {}
                        AxisItem::Module(_) => return Ok(false),
                    }
                };
                if !match_pattern_from_axis_position(
                    branch_pattern,
                    branch_axis,
                    0,
                    index,
                    bindings,
                    float_width,
                )? {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[derive(Debug, Clone)]
enum PositionRng {
    F32 { value: f32 },
    F64 { state: u64 },
}

impl PositionRng {
    fn new(seed: u64, iteration: u64, position: u64, float_width: FloatWidth) -> Self {
        match float_width {
            FloatWidth::F32 => Self::F32 {
                value: wgpu_random_unit(seed, iteration, position),
            },
            FloatWidth::F64 => Self::F64 {
                state: seed
                    ^ iteration.wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let Self::F64 { state } = self else {
            unreachable!("u64 sampling is used only by the f64 semantic profile")
        };
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    fn next_f32(&mut self) -> f32 {
        match self {
            Self::F32 { value } => *value,
            Self::F64 { .. } => self.next_f64() as f32,
        }
    }

    fn index(&mut self, length: usize) -> usize {
        debug_assert!(length > 0);
        match self {
            Self::F32 { value } => ((*value * length as f32) as usize).min(length - 1),
            Self::F64 { .. } => ((self.next_f64() * length as f64) as usize).min(length - 1),
        }
    }
}

fn wgpu_random_unit(seed: u64, iteration: u64, position: u64) -> f32 {
    fn hash_word(mut value: u32) -> u32 {
        value = (value ^ (value >> 16)).wrapping_mul(0x7FEB_352D);
        value = (value ^ (value >> 15)).wrapping_mul(0x846C_A68B);
        value ^ (value >> 16)
    }

    let mut key = seed as u32 ^ hash_word(((seed >> 32) as u32).wrapping_add(0x9E37_79B9));
    key ^= hash_word((iteration as u32).wrapping_add(0x85EB_CA6B));
    key ^= hash_word(((iteration >> 32) as u32).wrapping_add(0xC2B2_AE35));
    key ^= hash_word((position as u32).wrapping_add(0x27D4_EB2F));
    key ^= hash_word((position >> 32) as u32);
    let bits = hash_word(key) >> 8;
    bits as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn f32_rounds_each_expression_operation_for_ast_and_ir_execution() {
        let source = "axiom A(16777216); match A(x) then A(x + 1);";
        let grammar = Grammar::parse(source).unwrap();
        let ir = crate::ir::DerivationIr::from_grammar_source(&grammar, Some(source));

        for compile_ir in [false, true] {
            let backend = CpuBackend::new().float_width(FloatWidth::F32);
            let program = if compile_ir {
                backend.compile_ir(&ir)
            } else {
                backend.compile(&grammar)
            }
            .unwrap();
            assert_eq!(program.run(1).unwrap().to_string(), "A(16777216)");

            let backend = CpuBackend::new().float_width(FloatWidth::F64);
            let program = if compile_ir {
                backend.compile_ir(&ir)
            } else {
                backend.compile(&grammar)
            }
            .unwrap();
            assert_eq!(program.run(1).unwrap().to_string(), "A(16777217)");
        }
    }

    #[test]
    fn f32_selection_uses_flat_token_positions_and_the_wgpu_random_stream() {
        let grammar =
            Grammar::parse("axiom A [ A ] A; match A weight 1 then X; match A weight 1 then Y;")
                .unwrap();
        let seed = 19;
        let generation = CpuBackend::new()
            .float_width(FloatWidth::F32)
            .compile(&grammar)
            .unwrap()
            .run_with_seed(1, seed)
            .unwrap();
        let selected = |position| {
            if wgpu_random_unit(seed, 0, position) < 0.5 {
                "X"
            } else {
                "Y"
            }
        };
        assert_eq!(
            generation.to_string(),
            format!("{} [ {} ] {}", selected(0), selected(2), selected(4))
        );
    }

    #[test]
    fn runs_fibonacci_word() {
        let grammar = Grammar::parse(
            r#"
                axiom b;
                match a then a b;
                match b then a;
            "#,
        )
        .unwrap();
        let program = grammar.compile_cpu().unwrap();
        assert_eq!(program.run(0).unwrap().to_string(), "b");
        assert_eq!(program.run(1).unwrap().to_string(), "a");
        assert_eq!(program.run(2).unwrap().to_string(), "a b");
        assert_eq!(program.run(5).unwrap().to_string(), "a b a a b a b a");
    }

    #[test]
    fn runs_parametric_rules_and_deletion() {
        let grammar = Grammar::parse(
            r#"
                axiom A(3) Delete;
                match A(x) when x > 0 then A(x - 1);
                match A(x) when x = 0 then Done;
                match Delete then nothing;
            "#,
        )
        .unwrap();
        let program = grammar.compile_cpu().unwrap();
        assert_eq!(program.run(4).unwrap().to_string(), "Done");
    }

    #[test]
    fn weighted_rules_are_reproducible() {
        let grammar = Grammar::parse(
            r#"
                axiom A A A A;
                match A weight 1 then B;
                match A weight 1 then C;
            "#,
        )
        .unwrap();
        let program = grammar.compile_cpu().unwrap();
        assert_eq!(
            program.run_with_seed(1, 123).unwrap(),
            program.run_with_seed(1, 123).unwrap(),
        );
    }

    #[test]
    fn ir_and_grammar_cpu_entry_points_are_semantically_equivalent() {
        let grammar = Grammar::parse(
            r#"
                let STEP = BASE + 1;
                let BASE = sqrt(4);
                only L A R;
                axiom L(1) Hidden A(2) [ Twig(3) ] R(4);
                match A(x) left L(l) right R(r)
                    when x > 0 and r > l
                    weight STEP + x
                    then A(x + STEP) [ Child(r ^ 2) ];
                match A(x) left L(l) right R(r)
                    when x > 0 and r > l
                    weight 1
                    then Alternate(l + r);
            "#,
        )
        .unwrap();
        let backend = CpuBackend::new().ambiguous_rules(AmbiguousRulePolicy::Uniform);
        let legacy = backend.compile(&grammar).unwrap();
        let ir = crate::ir::IrDocument::from_grammar(&grammar)
            .validate()
            .unwrap();
        let shared = backend.compile_ir(&ir).unwrap();

        assert_eq!(legacy.globals(), shared.globals());
        assert_eq!(legacy.axiom(), shared.axiom());
        for seed in 0..64 {
            assert_eq!(
                legacy.run_with_seed(3, seed).unwrap(),
                shared.run_with_seed(3, seed).unwrap(),
                "seed {seed}",
            );

            let legacy_trace = legacy
                .trace_rewrite_with_control(legacy.axiom(), 0, seed, || false, |_| {})
                .unwrap();
            let shared_trace = shared
                .trace_rewrite_with_control(shared.axiom(), 0, seed, || false, |_| {})
                .unwrap();
            assert_eq!(legacy_trace, shared_trace, "trace seed {seed}");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn stochastic_results_do_not_depend_on_worker_count() {
        let grammar = Grammar::parse(
            r#"
                axiom A A A A A A A A;
                match A weight 1 then B;
                match A weight 1 then C;
            "#,
        )
        .unwrap();
        let serial = grammar
            .compile_cpu_with(CpuBackend::new().worker_threads(1))
            .unwrap()
            .run_with_seed(1, 123)
            .unwrap();
        let parallel = grammar
            .compile_cpu_with(CpuBackend::new().worker_threads(4))
            .unwrap()
            .run_with_seed(1, 123)
            .unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn ambiguous_rules_are_uniform_by_default() {
        let grammar = Grammar::parse("axiom A; match A then B; match A then C;").unwrap();
        let program = grammar.compile_cpu().unwrap();
        let choices = (0..32)
            .map(|seed| program.run_with_seed(1, seed).unwrap().to_string())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(choices.len(), 2);
    }

    #[test]
    fn traced_step_records_identity_growth_branches_and_deletion() {
        let grammar = Grammar::parse(
            "axiom A Keep Delete; match A then A [ B C ] D; match Delete then nothing;",
        )
        .unwrap();
        let program = grammar.compile_cpu().unwrap();
        let mut state = program.start_with_seed(17);
        let (stats, lineage) = state
            .step_with_lineage_and_control(|| false, |_| {})
            .unwrap();

        assert_eq!(lineage.successor_modules_per_input(), &[4, 1, 0]);
        assert_eq!(lineage.input_modules(), stats.input_modules);
        assert_eq!(lineage.output_modules(), Some(stats.output_modules));
        assert_eq!(state.generation().to_string(), "A [ B C ] D Keep");
    }

    #[test]
    fn traced_stochastic_step_uses_the_selected_successor() {
        let grammar =
            Grammar::parse("axiom A; match A weight 1 then B; match A weight 1 then C D E;")
                .unwrap();
        let program = grammar.compile_cpu().unwrap();
        for seed in 0..32 {
            let mut state = program.start_with_seed(seed);
            let (_, lineage) = state
                .step_with_lineage_and_control(|| false, |_| {})
                .unwrap();
            assert_eq!(
                lineage.successor_modules_per_input(),
                &[state.generation().module_count()]
            );
        }
    }

    #[test]
    fn supports_linear_context_with_only_filter() {
        let grammar = Grammar::parse(
            r#"
                only F;
                axiom F(10) Move F(2) H F(4);
                match F(x) left F(l) right F(r) then F(l + x + r);
            "#,
        )
        .unwrap();
        let program = grammar.compile_cpu().unwrap();
        assert_eq!(
            program.run(1).unwrap().to_string(),
            "F(10) Move F(16) H F(4)",
        );
    }

    #[test]
    fn supports_acropetal_branch_context() {
        let grammar = Grammar::parse(
            r#"
                ignore TurnLeft TurnRight;
                axiom Fb [ TurnLeft Fa ] Fa [ TurnRight Fa ] Fa;
                match Fa left Fb then Fb;
            "#,
        )
        .unwrap();
        let program = grammar.compile_cpu().unwrap();
        assert_eq!(
            program.run(1).unwrap().to_string(),
            "Fb [ TurnLeft Fb ] Fb [ TurnRight Fa ] Fa",
        );
    }

    #[test]
    fn evaluates_anabaena_rules() {
        let grammar = Grammar::parse(
            r#"
                let CH = 900;
                let CT = 0.4;
                let ST = 3.9;
                only F;
                axiom F(0,0,CH) F(4,1,CH) F(0,0,CH);
                match F(s,t,c) when t = 1 and s >= 6
                    then F(s / 3 * 2,2,c) Move(1) F(s / 3,1,c);
                match F(s,t,c) when t = 2 and s >= 6
                    then F(s / 3,2,c) Move(1) F(s / 3 * 2,1,c);
                match F(s,t,c) left F(_,_,k) right F(_,_,r)
                    when s > ST or c > CT
                    then F(s + 0.1,t,c + 0.25 * (k + r - 3 * c));
                match F(s,t,c) left F(_,_,_) right F(_,_,_)
                    then F(0,0,CH) H(1);
                match H(s) when s < 3 then H(s * 1.1);
            "#,
        )
        .unwrap();
        let program = grammar
            .compile_cpu_with(CpuBackend::new().ambiguous_rules(AmbiguousRulePolicy::First))
            .unwrap();
        let output = program.run(1).unwrap();
        assert_eq!(output.module_count(), 3);
    }

    #[test]
    fn cancellation_inside_one_iteration_is_transactional() {
        let axiom = std::iter::repeat_n("A", 20_000)
            .collect::<Vec<_>>()
            .join(" ");
        let grammar = Grammar::parse(&format!("axiom {axiom}; match A then A A;")).unwrap();
        let program = grammar.compile_cpu().unwrap();
        let mut state = program.start_with_seed(7);
        let original = state.generation().clone();
        let mut checkpoints = 0usize;
        let error = state
            .step_with_control(
                || {
                    checkpoints += 1;
                    checkpoints > 256
                },
                |_| {},
            )
            .unwrap_err();

        assert!(error.is_cancelled());
        assert_eq!(state.generation_index(), 0);
        assert_eq!(state.generation(), &original);
    }

    #[test]
    fn cancellation_during_lineage_capture_is_transactional() {
        let axiom = std::iter::repeat_n("A", 20_000)
            .collect::<Vec<_>>()
            .join(" ");
        let grammar = Grammar::parse(&format!("axiom {axiom}; match A then A A;")).unwrap();
        let program = grammar.compile_cpu().unwrap();
        let mut state = program.start_with_seed(11);
        let original = state.generation().clone();
        let mut checkpoints = 0usize;
        let error = state
            .step_with_lineage_and_control(
                || {
                    checkpoints += 1;
                    checkpoints > 45_000
                },
                |_| {},
            )
            .unwrap_err();

        assert!(error.is_cancelled());
        assert_eq!(state.generation_index(), 0);
        assert_eq!(state.generation(), &original);
    }

    #[test]
    fn configured_limits_remain_typed() {
        let grammar = Grammar::parse("axiom A; match A then A A;").unwrap();
        let program = grammar
            .compile_cpu_with(CpuBackend::new().limits(ExecutionLimits {
                max_modules: 1,
                ..ExecutionLimits::default()
            }))
            .unwrap();
        let error = program.run(1).unwrap_err();
        assert!(matches!(
            error.kind(),
            ExecutionErrorKind::LimitExceeded {
                resource: "generation modules",
                actual: 2,
                limit: 1,
            }
        ));
    }

    #[test]
    fn deeply_nested_generation_drops_iteratively() {
        let mut generation = Generation::default();
        for _ in 0..20_000 {
            generation = Generation(vec![GenerationItem::Branch(generation)]);
        }
        assert_eq!(generation.max_branch_depth(), 20_000);
        drop(generation);
    }
}
