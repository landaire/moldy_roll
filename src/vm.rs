use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::{
    ast::{AstNode, Expression},
    error::ParserErrorInternal,
};

#[derive(Debug, Eq, PartialEq)]
enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn from_type_name(name: &str, value: u64) -> Value {
        match name {
            "int" | "int32" | "long" | "INT" | "INT32" | "LONG" => {
                Value::I32(value.try_into().expect("failed to convert value"))
            }
            _ => todo!(),
        }
    }
}

struct Vm<'source> {
    definitions: Vec<AstNode<'source>>,
    output: Vec<Value>,
    bytecode: Vec<Opcode<'source>>,
    stack: Vec<Value>,
}

impl<'source> Vm<'source> {
    pub fn from_source(text: &'source str) -> Self {
        let bytecode = SourceTranslator::source_to_bytecode(text);

        Vm {
            definitions: vec![],
            output: vec![],
            bytecode,
            stack: vec![],
        }
    }

    pub fn eval(node: &AstNode<'source>, scope: Arc<RefCell<Scope>>) {}
}

struct SourceTranslator;

impl SourceTranslator {
    pub fn source_to_bytecode<'source>(text: &'source str) -> Vec<Opcode<'source>> {
        let parser = crate::ast::Parser::new(text);
        let mut ast = parser.parse().expect("error occurred while parsing");

        let mut bytecode = vec![];

        for node in ast {
            match node.typ {
                crate::ast::AstNodeType::VarDecl(decl) => {
                    if let Some(assignment) = decl.assignment {
                        let expression = assignment
                            .as_expression()
                            .expect("assignment is not an expression?");

                        Self::handle_expression(expression, &mut bytecode);
                    } else {
                        // TODO: handle structs
                        bytecode.push(Opcode::PushConst(Value::from_type_name(decl.typ.0, 0)))
                    }

                    bytecode.push(Opcode::DeclareLocal(decl.ident.0, decl.is_local));
                }
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

        bytecode
    }

    fn handle_expression<'source>(
        mut expression: &Expression<'source>,
        bytecode: &mut Vec<Opcode<'source>>,
    ) {
        let mut last_operator = None;
        loop {
            match &expression.lhs.typ {
                crate::ast::AstNodeType::Identifier(ident) => {
                    bytecode.push(Opcode::Push(ident.0));
                }
                crate::ast::AstNodeType::HexLiteral(lit)
                | crate::ast::AstNodeType::OctalLiteral(lit)
                | crate::ast::AstNodeType::BinaryLiteral(lit)
                | crate::ast::AstNodeType::DecimalLiteral(lit) => {
                    bytecode.push(Opcode::PushConst(Value::from_type_name("int32", *lit)));
                }
                _ => todo!(),
            }

            if let Some(last_operator) = last_operator {
                match last_operator {
                    crate::tokens::Operator::Add => bytecode.push(Opcode::Add),
                    crate::tokens::Operator::AddAssign => todo!(),
                    crate::tokens::Operator::Subtract => todo!(),
                    crate::tokens::Operator::Multiply => todo!(),
                    crate::tokens::Operator::Divide => todo!(),
                    crate::tokens::Operator::Xor => todo!(),
                    crate::tokens::Operator::BitOr => todo!(),
                    crate::tokens::Operator::BitAnd => todo!(),
                    crate::tokens::Operator::BitNot => todo!(),
                    crate::tokens::Operator::LeftShift => todo!(),
                    crate::tokens::Operator::RightShift => todo!(),
                    crate::tokens::Operator::LogicalNot => todo!(),
                    crate::tokens::Operator::LogicalOr => {
                        // Logical or short circuits
                        bytecode.push(Opcode::JumpIfTrue(9))
                    }
                    crate::tokens::Operator::LogicalAnd => todo!(),
                    crate::tokens::Operator::LogicalEquals => todo!(),
                    crate::tokens::Operator::LogicalNotEquals => todo!(),
                    crate::tokens::Operator::LogicalLessThan => bytecode.push(Opcode::LessThan),
                    crate::tokens::Operator::LogicalGreaterThan => bytecode.push(Opcode::GreaterThan),
                    crate::tokens::Operator::Not => todo!(),
                    crate::tokens::Operator::Assignment => todo!(),
                }
            }

            if let Some(rhs) = expression.rhs.as_ref() {
                last_operator = expression.operator;
                expression = rhs.as_expression().expect("rhs is not an expression");
                eprintln!("{:#?}", expression);

            } else {
                break;
            }
        }
    }

    fn default_value_for_type(type_name: &str) -> Option<Value> {
        match type_name {
            "int" | "int32" | "long" | "INT" | "INT32" | "LONG" => Some(Value::I32(0)),
            _ => None,
        }
    }
}

struct Scope {
    parent: Arc<RefCell<Self>>,
    vars: HashMap<String, Value>,
}

#[derive(Debug, Eq, PartialEq)]
enum Opcode<'source> {
    DeclareLocal(&'source str, bool),
    Add,
    Store(&'source str),
    Push(&'source str),
    PushConst(Value),
    JumpIfTrue(usize),
    GreaterThan,
    LessThan,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_basic_expression_to_opcode() {
        let code = r#"int x = 2;
        int y = x + 2;"#;

        let expected = vec![
            Opcode::PushConst(Value::I32(2)),
            Opcode::DeclareLocal("x", false),
            Opcode::Push("x"),
            Opcode::PushConst(Value::I32(2)),
            Opcode::Add,
            Opcode::DeclareLocal("y", false),
        ];

        let vm = Vm::from_source(code);

        assert!(vm.bytecode == expected);
    }

    #[test]
    fn translate_complex_expression_to_opcode() {
        let code = r#"int x = 2;
        int test = x > 2 || x < 1;"#;

        let expected = vec![
            Opcode::PushConst(Value::I32(2)),
            Opcode::DeclareLocal("x", false),
            Opcode::Push("x"),
            Opcode::PushConst(Value::I32(2)),
            Opcode::GreaterThan,
            Opcode::JumpIfTrue(9), // index of DeclareLocal("y")
            Opcode::Push("x"),
            Opcode::PushConst(Value::I32(1)),
            Opcode::LessThan,
            Opcode::DeclareLocal("y", false),
        ];

        let vm = Vm::from_source(code);

        panic!("{:#?}\n{:#?}", vm.bytecode, expected);

        assert!(vm.bytecode == expected);
    }
}
