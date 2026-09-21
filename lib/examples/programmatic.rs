//! Construct and validate an L-system directly from Rust values.

use std::error::Error;

use braken::{Document, Item, ModuleExpr, ModulePattern, Production, Word, WordItem};

fn module(name: &str) -> WordItem {
    WordItem::Module(ModuleExpr {
        name: name.into(),
        arguments: Vec::new(),
    })
}

fn rule(center: &str, successor: Vec<WordItem>) -> Item {
    Item::Production(Production {
        center: ModulePattern {
            name: center.into(),
            arguments: Vec::new(),
        },
        left: None,
        right: None,
        condition: None,
        weight: None,
        successor: Word(successor),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let grammar = Document {
        items: vec![
            Item::Axiom(Word(vec![module("b")])),
            rule("a", vec![module("a"), module("b")]),
            rule("b", vec![module("a")]),
        ],
    }
    .validate()?;

    let generation = grammar.compile_cpu()?.run(8)?;
    println!("{generation}");
    Ok(())
}
