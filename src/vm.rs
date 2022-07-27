use std::{cell::RefCell, collections::HashMap, sync::Arc};
use wasm_encoder::{
    CodeSection, Encode, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

use crate::{
    ast::{AstNode, Expression},
    error::ParserErrorInternal,
};

fn type_name_to_value(name: &str) -> ValType {
    match name {
        "int" | "int32" | "long" | "INT" | "INT32" | "LONG" => ValType::I32,
        _ => todo!(),
    }
}

struct Vm<'source> {
    definitions: Vec<AstNode<'source>>,
    bytecode: Vec<u8>,
}

impl<'source> Vm<'source> {
    pub fn from_source(text: &'source str) -> Self {
        let bytecode = SourceTranslator::source_to_bytecode(text);

        Vm {
            definitions: vec![],
            bytecode,
        }
    }
}

struct SourceTranslator;

impl SourceTranslator {
    pub fn source_to_bytecode<'source>(text: &'source str) -> Vec<u8> {
        let parser = crate::ast::Parser::new(text);
        let mut ast = parser.parse().expect("error occurred while parsing");

        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.function(vec![], vec![ValType::I32]);
        module.section(&types);

        let mut functions = FunctionSection::new();
        let type_index = 0;
        functions.function(type_index);
        module.section(&functions);

        let mut exports = ExportSection::new();
        exports.export("run", ExportKind::Func, type_index);
        module.section(&exports);

        let mut codes = CodeSection::new();
        let locals = vec![];
        let mut f = Function::new(locals);

        for node in ast {
            match node.typ {
                crate::ast::AstNodeType::VarDecl(decl) => {
                    todo!();
                }
                crate::ast::AstNodeType::FuncCall => todo!(),
                crate::ast::AstNodeType::Expression(_) => todo!(),
                crate::ast::AstNodeType::Statement(_) => todo!(),
                crate::ast::AstNodeType::StructDef(_) => todo!(),
                crate::ast::AstNodeType::EnumDef => todo!(),
                crate::ast::AstNodeType::FuncDef => todo!(),
                crate::ast::AstNodeType::Identifier(_) => todo!(),
                crate::ast::AstNodeType::HexLiteral(_) => todo!(),
                crate::ast::AstNodeType::OctalLiteral(_) => todo!(),
                crate::ast::AstNodeType::BinaryLiteral(_) => todo!(),
                crate::ast::AstNodeType::DecimalLiteral(_) => todo!(),
                crate::ast::AstNodeType::ControlFlow(control) => {
                    todo!()
                }
                crate::ast::AstNodeType::UnaryExpression(_) => todo!(),
                crate::ast::AstNodeType::Return(ret) => {
                    Self::handle_expression(ret.as_expression().unwrap(), &mut f);
                    f.instruction(&Instruction::End);
                }
            }
        }

        codes.function(&f);
        module.section(&codes);

        module.finish()
    }

    fn handle_expression<'source>(expression: &Expression, function: &mut Function) {
        match &expression.lhs.typ {
            crate::ast::AstNodeType::VarDecl(_) => todo!(),
            crate::ast::AstNodeType::FuncCall => todo!(),
            crate::ast::AstNodeType::Expression(_) => todo!(),
            crate::ast::AstNodeType::Statement(_) => todo!(),
            crate::ast::AstNodeType::Return(_) => todo!(),
            crate::ast::AstNodeType::StructDef(_) => todo!(),
            crate::ast::AstNodeType::EnumDef => todo!(),
            crate::ast::AstNodeType::FuncDef => todo!(),
            crate::ast::AstNodeType::Identifier(_) => todo!(),
            crate::ast::AstNodeType::HexLiteral(lit)
            | crate::ast::AstNodeType::OctalLiteral(lit)
            | crate::ast::AstNodeType::BinaryLiteral(lit)
            | crate::ast::AstNodeType::DecimalLiteral(lit) => {
                function.instruction(&Instruction::I32Const(*lit as i32));
            }
            crate::ast::AstNodeType::ControlFlow(control) => {}
            crate::ast::AstNodeType::UnaryExpression(_) => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_basic_expression_to_opcode() {
        let code = r#"
        return 0x41;"#;

        let expected_code = r#"(module
  (type (;0;) (func (result i32)))
  (func (;0;) (type 0) (result i32)
    i32.const 65
  )
  (export "run" (func 0))
)"#;

        let vm = Vm::from_source(code);
        let result_code =
            wasmprinter::print_bytes(&vm.bytecode).expect("failed to convert bytecode to string");

        // Compile the expected code to bytecode, then back to avoid formatting issues
        assert!(result_code == expected_code);
    }

    #[test]
    fn translate_basic_expression_to_opcode() {
        let code = r#"
        int x = 0;
        x += 2;"#;

        let expected_code = r#"(module
  (type (;0;) (func (result i32)))
  (func (;0;) (type 0) (result i32)
    i32.const 65
  )
  (export "run" (func 0))
)"#;

        let vm = Vm::from_source(code);
        let result_code =
            wasmprinter::print_bytes(&vm.bytecode).expect("failed to convert bytecode to string");

        eprintln!("{}\n{}", result_code);

        // Compile the expected code to bytecode, then back to avoid formatting issues
        assert!(result_code == expected_code);
    }
}
}
