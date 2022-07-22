use crate::error::{ParserErrorInternal, ParserErrorWithSpan};
use crate::tokens::{Operator, Token};
use peekmore::{PeekMore, PeekMoreIterator};
use std::iter::Peekable;
use std::str::FromStr;

use std::{ops::Range, str::Chars};

use smallvec::{smallvec, SmallVec};

pub(crate) type Span = Range<usize>;

#[derive(Debug, PartialEq, Eq)]
struct VarDecl<'source> {
    typ: Identifier<'source>,
    ident: Identifier<'source>,
    array_size: Option<SmallVec<[u8; 2]>>,
    is_local: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Expression<'source> {
    lhs: Box<AstNode<'source>>,
    operator: Option<Operator>,
    rhs: Option<Box<Expression<'source>>>,
}

#[derive(Debug, PartialEq, Eq)]
struct StructDef<'source> {
    idents: SmallVec<[Identifier<'source>; 3]>,
    members: Vec<AstNode<'source>>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct AstNode<'source> {
    typ: AstNodeType<'source>,
    span: Span,
}

impl<'source> AstNode<'source> {
    fn as_ident(&self) -> Option<&Identifier<'source>> {
        if let AstNodeType::Identifier(ident) = &self.typ {
            Some(ident)
        } else {
            None
        }
    }

    fn as_struct(&self) -> Option<&StructDef<'source>> {
        if let AstNodeType::StructDef(s) = &self.typ {
            Some(s)
        } else {
            None
        }
    }

    fn as_struct_mut(&mut self) -> Option<&mut StructDef<'source>> {
        if let AstNodeType::StructDef(s) = &mut self.typ {
            Some(s)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Identifier<'source>(&'source str);

#[derive(Debug, Eq, PartialEq)]
enum AstNodeType<'source> {
    VarDecl(VarDecl<'source>),
    FuncCall,
    Expression(Expression<'source>),
    Statement,
    StructDef(StructDef<'source>),
    EnumDef,
    FuncDef,
    Identifier(Identifier<'source>),
    HexLiteral(u64),
    OctalLiteral(u64),
    BinaryLiteral(u64),
    DecimalLiteral(u64),
    ControlFlow(ControlFlow<'source>),
    UnaryExpression(UnaryExpression<'source>),
}

#[derive(Debug, Eq, PartialEq)]
struct UnaryExpression<'source> {
    op: Operator,
    expr: Box<AstNode<'source>>,
}

#[derive(Debug, Eq, PartialEq)]
struct IfCondition<'source> {
    condition: Expression<'source>,
    body: Vec<AstNode<'source>>,
}

#[derive(Debug, Eq, PartialEq)]
enum ControlFlow<'source> {
    If(IfCondition<'source>),
}

trait CharExt {
    fn is_ident_start(&self) -> bool;
}

impl CharExt for char {
    fn is_ident_start(&self) -> bool {
        self.is_ascii_alphabetic()
    }
}

struct Parser<'source> {
    input: &'source str,
    iter: Vec<char>,
    position: usize,
    line_num: usize,
}

impl<'source> Parser<'source> {
    pub fn new(input: &'source str) -> Self {
        let iter = input.chars().collect();
        Parser {
            input,
            iter,
            position: 0,
            line_num: 1,
        }
    }

    fn parse_number(&mut self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position;

        let number_start = self.next()?;
        let number = if number_start == '0' {
            // If the number starts with 0, we may be parsing a hex, binary, or octal literal
            match self.peek() {
                Ok('x') => {
                    self.next();

                    let start = self.position;

                    let end =
                        self.advance_while(|c| matches!(c, 'a'..='f' | 'A'..='F' | '0'..='9'));

                    AstNodeType::HexLiteral(
                        u64::from_str_radix(&self.input[start..end], 16)
                            .expect("expected valid hex literal"),
                    )
                }
                Ok('b') => {
                    self.next();

                    let start = self.position;

                    let end = self.advance_while(|c| matches!(c, '0'..='1'));
                    AstNodeType::BinaryLiteral(
                        u64::from_str_radix(&self.input[start..end], 2)
                            .expect("expected valid binary literal"),
                    )
                }
                Ok('1'..='7') => {
                    let start = self.position;
                    let end = self.advance_while(|c| matches!(c, '0'..='9'));
                    AstNodeType::OctalLiteral(
                        u64::from_str_radix(&self.input[start..end], 8)
                            .expect("expected valid octal literal"),
                    )
                }
                _other => {
                    self.next();
                    AstNodeType::DecimalLiteral(0)
                }
            }
        } else {
            // Parse a decimal number
            let end = self.advance_while(char::is_ascii_digit);
            AstNodeType::DecimalLiteral(
                u64::from_str_radix(&self.input[start..end], 10)
                    .expect("expected valid decimal literal"),
            )
        };

        Ok(AstNode {
            typ: number,
            span: start..self.position,
        })
    }

    fn chomp_whitespace(&mut self) -> bool {
        let start = self.position;
        self.advance_while(char::is_ascii_whitespace);

        start != self.position
    }

    fn chomp_multiline_comment(&mut self) {}

    fn chomp_ignored_tokens(&mut self) -> Result<bool, ParserErrorInternal<'source>> {
        let start = self.position;

        loop {
            self.chomp_whitespace();

            // Peek the next token to see if it's a comment.
            match self.peek() {
                Ok('/') => {
                    self.next()?;
                    match self.peek() {
                        Ok('/') => {
                            self.next()?;

                            loop {
                                if matches!(
                                    self.next(),
                                    Ok('\n') | Err(ParserErrorInternal::UnexpectedEndOfFile)
                                ) {
                                    // We reached either the end of file or the end
                                    // of this line -- we can stop chomping
                                    break;
                                }
                            }
                        }
                        Ok('*') => {
                            self.next()?;

                            loop {
                                if matches!(
                                    self.next(),
                                    Ok('*') | Err(ParserErrorInternal::UnexpectedEndOfFile)
                                ) {
                                    if matches!(
                                        self.next(),
                                        Ok('/') | Err(ParserErrorInternal::UnexpectedEndOfFile)
                                    ) {
                                        break;
                                    }
                                }
                            }
                        }
                        _ => {
                            // Rewind the last token -- it's not a comment
                            self.rewind();
                            break;
                        }
                    }
                }
                _ => break,
            }
        }

        Ok(start != self.position)
    }

    fn parse_ident(&mut self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position;

        // identifiers must start with a letter
        if !self.peek()?.is_ident_start() {
            return Err(ParserErrorInternal::UnexpectedWithDescription("identifier"));
        }

        let end = self.advance_while(|c| c.is_ascii_alphanumeric());
        let ntype = AstNodeType::Identifier(Identifier(&self.input[start..end]));

        Ok(AstNode {
            typ: ntype,
            span: start..end,
        })
    }

    fn parse_struct_members(
        &mut self,
    ) -> Result<Vec<AstNode<'source>>, ParserErrorInternal<'source>> {
        let mut members = Vec::with_capacity(8);
        loop {
            self.chomp_ignored_tokens()?;

            // Parse the first identifier (type) if it exists
            let next = self.peek()?;
            if next.is_ident_start() {
                let start = self.position;
                let typ = self.parse_ident()?;

                self.chomp_ignored_tokens()?;

                // Parse the second identifier (name) if it exists
                let next = self.peek()?;
                if next.is_ident_start() {
                    let name = self.parse_ident()?;

                    self.chomp_ignored_tokens()?;

                    let node_ty = VarDecl {
                        typ: typ.as_ident().expect("expected ident").clone(),
                        ident: name.as_ident().expect("expected ident").clone(),
                        array_size: None,
                        is_local: false,
                    };
                    let ast_node = AstNode {
                        typ: AstNodeType::VarDecl(node_ty),
                        span: (start..self.position),
                    };

                    members.push(ast_node);

                    let next = self.next()?;
                    let next_token = next
                        .try_into()
                        .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;
                    match next_token {
                        Token::Semicolon => {
                            continue;
                        }
                        Token::AngleBracketOpen => {
                            todo!();
                        }
                        _other => {
                            return Err(ParserErrorInternal::UnexpectedCharacter(next));
                        }
                    }
                } else {
                    return Err(ParserErrorInternal::UnexpectedCharacter(next));
                }
            }

            // If we didn't get an ident, we should be expecting a closing bracket
            self.expect_next_token(Token::CloseBrace)?;
        }

        Ok(members)
    }

    fn parse_struct(
        &mut self,
        must_have_name: bool,
    ) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position;

        self.chomp_ignored_tokens()?;

        // Next token *may* be an identifier
        let ident = if self.peek()?.is_ident_start() {
            Some(self.parse_ident()?)
        } else {
            None
        };

        if must_have_name {
            if ident.is_none() {
                return Err(ParserErrorInternal::UnexpectedWithDescription("ident"));
            }

            // If we must have a name, we could just have a "named" struct with
            // no body (ZST)
            if Token::try_from(self.peek()?)
                .map(|t| t == Token::Semicolon)
                .unwrap_or(false)
            {
                self.next()?;

                let typ = AstNodeType::StructDef(StructDef {
                    idents: smallvec![ident.unwrap().as_ident().expect("expected ident").clone()],
                    members: Vec::new(),
                });

                return Ok(AstNode {
                    typ,
                    span: start..self.position,
                });
            }
        }

        // Next token must be either an open paren (arguments) or bracket (body)
        let next = self.next()?;
        let next_token = next
            .try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;
        let mut struct_members = Vec::new();

        match next_token {
            Token::OpenBrace => {
                struct_members = self.parse_struct_members()?;
            }
            Token::OpenParen => {
                todo!();
            }
            _other => {
                return Err(ParserErrorInternal::UnexpectedCharacter(next));
            }
        };

        let mut idents = SmallVec::new();
        if ident.is_some() {
            idents.push(ident.unwrap().as_ident().expect("expected ident").clone());
        }

        let s = StructDef {
            idents,
            members: struct_members,
        };

        Ok(AstNode {
            typ: AstNodeType::StructDef(s),
            span: start..self.position,
        })
    }

    fn parse_typedef(&mut self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        self.chomp_ignored_tokens()?;

        let start = self.position;

        // Try parsing the "typedef" keyword
        let ident = self.parse_ident()?;

        // Next token must be whitespace
        if !self.chomp_ignored_tokens()? {
            let next = self.next()?;

            return Err(ParserErrorInternal::UnexpectedCharacter(next));
        }

        let type_ident = self.parse_ident()?;
        let typedef_type = type_ident
            .as_ident()
            .expect("ident result should be ident type");
        let mut s = match typedef_type.0 {
            "enum" => {
                todo!()
            }
            "struct" => self.parse_struct(false)?,
            other => {
                return Err(ParserErrorInternal::UnexpectedWithDescription(
                    "\"enum\" or \"struct\"",
                ));
            }
        };

        let s_def = s.as_struct_mut().expect("expected struct");
        loop {
            self.chomp_ignored_tokens()?;

            let next = self.peek()?;
            if next.is_ident_start() {
                let ident = self.parse_ident()?;
                s_def
                    .idents
                    .push(ident.as_ident().expect("expected ident").clone());
            } else {
                let next_token: Token = next
                    .try_into()
                    .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;
                match next_token {
                    Token::Semicolon => {
                        self.next();

                        s.span.start = start;
                        s.span.end = self.position;

                        return Ok(s);
                    }
                    Token::AngleBracketOpen => {
                        todo!();
                    }
                    _other => {
                        return Err(ParserErrorInternal::UnexpectedCharacter(next));
                    }
                }
            }
        }
    }

    fn parse_expression(&mut self) -> Result<AstNode<'source>, ParserErrorInternal> {
        // at this point we've already parsed the `if` token...
        self.chomp_ignored_tokens();

        let start = self.position;

        // Parse an identifier
        let next_char = self.peek()?;
        let next_token = self.peek_token();
        let lhs = if next_char.is_ident_start() {
            self.parse_ident()?
        } else if next_char.is_ascii_digit() {
            self.parse_number()?
        } else {
            // We may have an operator or symbol
            let next_token = next_token?;
            match next_token {
                Token::OpenParen => {
                    todo!();
                }
                other if other.as_unary_operator().is_some() => {
                    let inner_expr = AstNode {
                            typ: AstNodeType::UnaryExpression(UnaryExpression {
                                op: other
                                    .as_unary_operator()
                                    .expect("operator should be unary operator"),
                                expr: Box::new(self.parse_expression()?),
                            }),
                            span: start..self.position,
                        };

                    inner_expr
                }
                other => {
                    return Err(ParserErrorInternal::UnexpectedCharacter(next_char));
                }
            }
        };

        let expr_ty = Expression {
            lhs: Box::new(lhs),
            operator: None,
            rhs: None,
        };

        Ok(AstNode {
            typ: AstNodeType::Expression(expr_ty),
            span: start..self.position,
        })
    }

    fn parse_if_condition(&mut self) -> Result<AstNode<'source>, ParserErrorInternal> {
        // at this point we've already parsed the `if` token...
        self.chomp_ignored_tokens();

        self.peek_expect_token(Token::OpenParen)?;

        let condition = self.parse_expression();

        self.chomp_ignored_tokens();

        // Check for an opening brace
        if self.peek_expect_token(token)

        Ok(())
    }

    fn parse_root(&mut self) -> Result<AstNode<'source>, ParserErrorWithSpan> {
        todo!();
    }

    fn next(&mut self) -> Result<char, ParserErrorInternal<'source>> {
        let next = self
            .iter
            .get(self.position)
            .cloned()
            .ok_or(ParserErrorInternal::UnexpectedEndOfFile);
        if next.is_ok() {
            self.position += 1;
        }

        next
    }

    fn next_token(&mut self) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.next()?;

        next.try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))
    }

    fn peek_token(&mut self) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.peek()?;

        next.try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))
    }

    fn peek_expect_token(&mut self, token: Token) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.peek()?;

        let next_token = next
            .try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;

        if next_token != token {
            return Err(ParserErrorInternal::UnexpectedCharacter(next));
        }

        Ok(next_token)
    }

    fn expect_next_token(&mut self, token: Token) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.next()?;

        let next_token = next
            .try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;

        if next_token != token {
            return Err(ParserErrorInternal::UnexpectedCharacter(next));
        }

        Ok(next_token)
    }

    fn advance_while<F: Fn(&char) -> bool>(&mut self, func: F) -> usize {
        while self.position < self.iter.len() && func(&self.iter[self.position]) {
            self.position += 1;
        }

        self.position
    }

    fn peek(&mut self) -> Result<char, ParserErrorInternal<'source>> {
        self.iter
            .get(self.position)
            .cloned()
            .ok_or(ParserErrorInternal::UnexpectedEndOfFile)
    }

    fn rewind(&mut self) {
        self.position = self.position.saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_number() {
        let s = "100";
        let mut parser = Parser::new(s);

        let result = parser.parse_number().unwrap();

        assert!(matches!(result.typ, AstNodeType::DecimalLiteral(100)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_hex_string() {
        let s = "0x100";
        let mut parser = Parser::new(s);

        let result = parser.parse_number().unwrap();

        assert!(matches!(result.typ, AstNodeType::HexLiteral(0x100)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_hex_string_with_alpha() {
        let s = "0xabc3";
        let mut parser = Parser::new(s);

        let result = parser.parse_number().unwrap();

        assert!(matches!(result.typ, AstNodeType::HexLiteral(0xabc3)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_binary_string() {
        let s = "0b101";
        let mut parser = Parser::new(s);

        let result = parser.parse_number().unwrap();

        assert!(matches!(result.typ, AstNodeType::BinaryLiteral(0b101)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_octal_string() {
        let s = "01777";
        let mut parser = Parser::new(s);

        let result = parser.parse_number().unwrap();

        assert!(matches!(result.typ, AstNodeType::OctalLiteral(0o1777)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_basic_struct_def() {
        let s = r#"typedef struct {
            char field;
        } Foo;"#;

        let mut parser = Parser::new(s);

        let result = parser.parse_typedef().unwrap();

        let idents = smallvec![Identifier("Foo")];
        let members = vec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 0..1,
        }];

        let expected = AstNodeType::StructDef(StructDef { idents, members });

        assert!(result.typ == expected);
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_basic_struct_def_with_comments() {
        let s = r#"
        // start comment
        typedef struct {
            // comment in the middle of the def
            char /* comment between idents */ field;// and at the end of the decl
            /*
                multiline comment
             */
        } Foo;"#;

        let mut parser = Parser::new(s);

        let result = parser.parse_typedef().unwrap();

        let idents = smallvec![Identifier("Foo")];
        let members = vec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 0..1,
        }];

        let expected = AstNodeType::StructDef(StructDef { idents, members });

        assert!(result.typ == expected);
        assert!(result.span == (s.find("typedef").unwrap()..s.len()));
    }

    fn parse_if_condition() {
        let s = r#"
            if (foo)
                char field;
        "#;

        let mut parser = Parser::new(s);

        let result = parser.parse_if_condition().unwrap();

        let idents = smallvec![Identifier("Foo")];
        let body = smallvec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 0..1,
        }];

        let typ = AstNodeType::Identifier("foo");

        let expected = AstNodeType::ControlFlow(ControlFlow::If(IfCondition {
            condition: Expression {
                lhs: AstNode {
                    typ: Box::new(typ),
                    span: (0..5),
                },
                operator: None,
                rhs: None,
            },
            body,
        }));

        assert!(result.typ == expected);
        assert!(result.span == (s.find("if").unwrap()..s.len()));
    }

    // #[test]
    fn struct_with_conditional_field() {
        let s = r#"
        // start comment
        typedef struct {
            if (foo)
                char field;
        } Foo;"#;

        let mut parser = Parser::new(s);

        let result = parser.parse_typedef().unwrap();

        let idents = smallvec![Identifier("Foo")];
        let members = vec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 0..1,
        }];

        let expected = AstNodeType::StructDef(StructDef { idents, members });

        assert!(result.typ == expected);
        assert!(result.span == (s.find("typedef").unwrap()..s.len()));
    }
}
