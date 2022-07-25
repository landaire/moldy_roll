use std::{cell::RefCell, sync::Arc, collections::HashMap};

use crate::{ast::AstNode, error::ParserErrorInternal};

#[derive(Debug, Eq, PartialEq)]
enum Value {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    String(String),
    Array(Vec<Value>),
}

struct Vm<'source> {
    definitions: Vec<AstNode<'source>>,
    output: Vec<Value>,
    bytecode: Vec<Opcode<'source>>,
    stack: Vec<Value>,
}

impl<'source> Vm<'source> {
    pub fn from_source(text: &'source str) -> Self {
        let parser = crate::ast::Parser::new(text);
        let mut ast = parser.parse().expect("error occurred while parsing");

        for node in ast {
            match node.typ {
                crate::ast::AstNodeType::VarDecl(decl) => {
                },
                crate::ast::AstNodeType::FuncCall => todo!(),
                crate::ast::AstNodeType::Expression(_) => todo!(),
                crate::ast::AstNodeType::Statement => todo!(),
                crate::ast::AstNodeType::StructDef(_) => todo!(),
                crate::ast::AstNodeType::EnumDef => todo!(),
                crate::ast::AstNodeType::FuncDef => todo!(),
                crate::ast::AstNodeType::Identifier(_) => todo!(),
                crate::ast::AstNodeType::HexLiteral(_) => todo!(),
                crate::ast::AstNodeType::OctalLiteral(_) => todo!(),
                crate::ast::AstNodeType::BinaryLiteral(_) => todo!(),
                crate::ast::AstNodeType::DecimalLiteral(_) => todo!(),
                crate::ast::AstNodeType::ControlFlow(_) => todo!(),
                crate::ast::AstNodeType::UnaryExpression(_) => todo!(),
            }
        }

        todo!();
    }

    pub fn eval(node: &AstNode<'source>, scope: Arc<RefCell<Scope>>) {

    }
}

struct Scope {
    parent: Arc<RefCell<Self>>,
    vars: HashMap<String, Value>,
}

#[derive(Debug, Eq, PartialEq)]
enum Opcode<'source> {
    DeclareLocal(&'source str, Value),
    Push(Value),
    Add(Value),
    Store(&'source str),
    Load(&'source str),
    LoadConst(Value),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_basic_expression_to_opcode() {
        let code = r#"int x = 2;
        int y = x + 2"#;

        let expected = vec![
            Opcode::Push(Value::U32(2)),
            Opcode::DeclareLocal("x", Value::U32(0)),
            Opcode::Load("x"),
            Opcode::LoadConst(Value::U32(2)),
            Opcode::DeclareLocal("y", Value::U32(0)),
            Opcode::Store("y"),
        ];

        let vm = Vm::from_source(code);

        assert!(vm.bytecode == expected);
    }
}