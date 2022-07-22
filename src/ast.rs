use crate::error::{ParserErrorInternal, ParserErrorWithSpan};
use crate::tokens::{Operator, Token};
use peekmore::{PeekMore, PeekMoreIterator};
use std::cell::RefCell;
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
    rhs: Option<Box<AstNode<'source>>>,
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

    fn as_expression(&self) -> Option<&Expression<'source>> {
        if let AstNodeType::Expression(expression) = &self.typ {
            Some(expression)
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

impl<'source> Identifier<'source> {
    fn is_keyword_if(&self) -> bool {
        self.0 == "if"
    }
}

macro_rules! ident {
    ($ident:expr) => {
        Identifier($ident)
    };
}

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
    condition: Box<AstNode<'source>>,
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
    position: RefCell<usize>,
    line_num: usize,
}

impl<'source> Parser<'source> {
    pub fn new(input: &'source str) -> Self {
        let iter = input.chars().collect();
        Parser {
            input,
            iter,
            position: RefCell::new(0),
            line_num: 1,
        }
    }

    fn position(&self) -> usize {
        *self.position.borrow()
    }

    fn parse_number(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position();

        let number_start = self.next()?;
        let number = if number_start == '0' {
            // If the number starts with 0, we may be parsing a hex, binary, or octal literal
            match self.peek() {
                Ok('x') => {
                    self.next();

                    let start = self.position();

                    let end =
                        self.advance_while(|c| matches!(c, 'a'..='f' | 'A'..='F' | '0'..='9'));

                    AstNodeType::HexLiteral(
                        u64::from_str_radix(&self.input[start..end], 16)
                            .expect("expected valid hex literal"),
                    )
                }
                Ok('b') => {
                    self.next();

                    let start = self.position();

                    let end = self.advance_while(|c| matches!(c, '0'..='1'));
                    AstNodeType::BinaryLiteral(
                        u64::from_str_radix(&self.input[start..end], 2)
                            .expect("expected valid binary literal"),
                    )
                }
                Ok('1'..='7') => {
                    let start = self.position();
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
            span: start..self.position(),
        })
    }

    fn chomp_whitespace(&self) -> bool {
        let start = self.position();
        self.advance_while(char::is_ascii_whitespace);

        start != self.position()
    }

    fn chomp_multiline_comment(&self) {}

    fn chomp_ignored_tokens(&self) -> Result<bool, ParserErrorInternal<'source>> {
        let start = self.position();

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
                                    println!("chomping line comment");
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
                                ) && matches!(
                                    self.next(),
                                    Ok('/') | Err(ParserErrorInternal::UnexpectedEndOfFile)
                                ) {
                                    break;
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

        Ok(start != self.position())
    }

    fn parse_ident(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position();

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

    fn parse_var_decl(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position();
        let typ = self.parse_ident()?;
        if typ.as_ident().unwrap().is_keyword_if() {
            self.rewind_to(start);
            return self.parse_if_condition();
        }

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
                span: (start..self.position()),
            };

            self.chomp_ignored_tokens()?;

            let next = self.peek()?;
            let next_token = next
                .try_into()
                .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;
            match next_token {
                Token::Semicolon => {
                    let _ = self.next()?;
                    return Ok(ast_node);
                }
                Token::AngleBracketOpen => {
                    todo!();
                }
                _other => {
                    println!("bad next token");
                    return Err(ParserErrorInternal::UnexpectedCharacter(next));
                }
            }
        } else {
            return Err(ParserErrorInternal::UnexpectedCharacter(next));
        }
    }

    fn parse_struct_members(&self) -> Result<Vec<AstNode<'source>>, ParserErrorInternal<'source>> {
        let mut members = Vec::with_capacity(8);
        loop {
            self.chomp_ignored_tokens()?;

            // Parse the first identifier (type) if it exists
            let next = self.peek()?;
            if next.is_ident_start() {
                let var_decl = self.parse_var_decl()?;
                members.push(var_decl);

                let next = self.peek()?;
                let next_token: Token = next
                    .try_into()
                    .map_err(|_e| ParserErrorInternal::UnexpectedCharacter('a'))?;
                if next_token == Token::Semicolon {
                    todo!();
                }

                continue;
            }

            // If we didn't get an ident, we should be expecting a closing bracket
            self.expect_next_token(Token::CloseBrace)?;
            break;
        }

        Ok(members)
    }

    fn parse_struct(
        &self,
        must_have_name: bool,
    ) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position();

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
                    span: start..self.position(),
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
        if let Some(ident) = ident {
            idents.push(ident.as_ident().expect("expected ident").clone());
        }

        let s = StructDef {
            idents,
            members: struct_members,
        };

        Ok(AstNode {
            typ: AstNodeType::StructDef(s),
            span: start..self.position(),
        })
    }

    fn parse_typedef(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        self.chomp_ignored_tokens()?;

        let start = self.position();

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
                        let _ = self.next();

                        s.span.start = start;
                        s.span.end = self.position();

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

    fn parse_operator(&self) -> Result<Operator, ParserErrorInternal<'source>> {
        let first_token = self.next_token()?;
        let next_char = self.peek()?;
        match first_token {
            Token::AngleBracketOpen => {
                if next_char.is_ident_start() || next_char.is_ascii_whitespace() {
                    return Ok(Operator::LogicalLessThan);
                }

                match self.peek_token()? {
                    // Left shift
                    Token::AngleBracketOpen => {
                        // Consume the token
                        let _ = self.next_token()?;
                        return Ok(Operator::LeftShift);
                    }
                    Token::OpenParen | Token::Whitespace => {
                        return Ok(Operator::LogicalLessThan);
                    }
                    other => {
                        // anything else is an error
                        return Err(ParserErrorInternal::UnexpectedCharacter(self.peek()?));
                    }
                }
            }
            Token::AngleBracketClose => {
                if next_char.is_ident_start() || next_char.is_ascii_whitespace() {
                    return Ok(Operator::LogicalGreaterThan);
                }

                match self.peek_token()? {
                    // Right shift
                    Token::AngleBracketClose => {
                        // Consume the token
                        let _ = self.next_token()?;
                        return Ok(Operator::RightShift);
                    }
                    Token::OpenParen | Token::Whitespace => {
                        return Ok(Operator::LogicalGreaterThan);
                    }
                    other => {
                        // anything else is an error
                        return Err(ParserErrorInternal::UnexpectedCharacter(self.peek()?));
                    }
                }
            }
            Token::Operator(Operator::BitAnd) => {
                if next_char.is_ident_start() || next_char.is_ascii_whitespace() {
                    return Ok(Operator::BitAnd);
                }

                match self.peek_token()? {
                    // Left shift
                    Token::Operator(Operator::BitAnd) => {
                        // Consume the token
                        let _ = self.next_token()?;
                        return Ok(Operator::LogicalAnd);
                    }
                    Token::OpenParen | Token::Whitespace => {
                        return Ok(Operator::BitAnd);
                    }
                    other => {
                        // anything else is an error
                        return Err(ParserErrorInternal::UnexpectedCharacter(self.peek()?));
                    }
                }
            }
            Token::Operator(Operator::Assignment) => {
                if next_char.is_ident_start() || next_char.is_ascii_whitespace() {
                    return Ok(Operator::Assignment);
                }

                match self.peek_token()? {
                    Token::Operator(Operator::Assignment) => {
                        // Consume the token
                        let _ = self.next_token()?;
                        return Ok(Operator::LogicalEquals);
                    }
                    Token::OpenParen | Token::Whitespace | Token::Quote => {
                        return Ok(Operator::Assignment);
                    }
                    other => {
                        // anything else is an error
                        return Err(ParserErrorInternal::UnexpectedCharacter(self.peek()?));
                    }
                }
            }
            _ => todo!(),
        }
    }

    fn parse_expression(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        self.chomp_ignored_tokens()?;
        let start = self.position();

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
                        span: start..self.position(),
                    };

                    inner_expr
                }
                other => {
                    return Err(ParserErrorInternal::UnexpectedCharacter(next_char));
                }
            }
        };

        self.chomp_ignored_tokens()?;

        // We might have an operator here
        let (operator, rhs) = match self.peek_token() {
            Ok(Token::AngleBracketClose | Token::AngleBracketOpen | Token::Operator(_)) => {
                let op = Some(self.parse_operator()?);
                let rhs = Some(Box::new(self.parse_expression()?));

                (op, rhs)
            }
            Err(e) => return Err(e),
            other => (None, None),
        };

        let expr_ty = Expression {
            lhs: Box::new(lhs),
            operator,
            rhs,
        };

        Ok(AstNode {
            typ: AstNodeType::Expression(expr_ty),
            span: start..self.position(),
        })
    }

    fn parse_if_condition(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        self.chomp_ignored_tokens()?;

        let start = self.position();
        // Parse the `if` keyword
        let ident = self.parse_ident()?;
        let ident = ident.as_ident().expect("identifier expected");
        if ident.0 != "if" {
            return Err(
                crate::error::ParserErrorInternal::UnexpectedWithDescription("expected `if`"),
            );
        }

        self.chomp_ignored_tokens()?;

        self.expect_next_token(Token::OpenParen)?;

        let condition = self.parse_expression()?;

        self.expect_next_token(Token::CloseParen)?;

        self.chomp_ignored_tokens()?;

        // Check for an opening brace
        let body = if self.peek_expect_token(Token::OpenBrace).is_ok() {
            // we have multiple statements in the if condition's body
            let mut statements = vec![];
            loop {
                match self.peek_expect_token(Token::CloseBrace) {
                    Err(ParserErrorInternal::UnexpectedEndOfFile) => {
                        // We've reached the end of the file but were expecting a closing brace
                        return Err(ParserErrorInternal::UnexpectedEndOfFile);
                    }
                    Ok(_) => {
                        break;
                    }
                    _other => {
                        // We don't care if we reached an unexpected token -- so long as it's not an EOF
                    }
                }

                statements.push(self.parse_any()?);
            }

            statements
        } else {
            vec![self.parse_any()?]
        };

        Ok(AstNode {
            typ: AstNodeType::ControlFlow(ControlFlow::If(IfCondition {
                condition: Box::new(condition),
                body,
            })),
            span: start..self.position(),
        })
    }

    fn parse_any(&self) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        self.chomp_ignored_tokens()?;

        let next_char = self.peek()?;
        if next_char.is_ident_start() {
            // Record the position so we can rewind once we figure out what we're
            // dealing with

            let start = self.position();

            // We have an identifier that may be a function call or var declaration
            let ident = self.parse_ident()?;
            let ident = ident.as_ident().expect("expected identifier");
            self.chomp_whitespace();

            if ident.is_keyword_if() {
                self.rewind_to(start);

                return self.parse_if_condition();
            }

            self.chomp_ignored_tokens()?;

            // We may have another identifier -- that indiciates a variable declaration
            let next_char = self.peek()?;
            if next_char.is_ident_start() {
                self.rewind_to(start);
                return self.parse_var_decl();
            }
        }

        todo!()
    }

    fn next(&self) -> Result<char, ParserErrorInternal<'source>> {
        let next = self
            .iter
            .get(self.position())
            .cloned()
            .ok_or(ParserErrorInternal::UnexpectedEndOfFile);
        if next.is_ok() {
            *self.position.borrow_mut() += 1;
        }

        next
    }

    fn next_token(&self) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.next()?;

        next.try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))
    }

    fn peek_token(&self) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.peek()?;

        next.try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))
    }

    fn peek_expect_token(&self, token: Token) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.peek()?;

        let next_token = next
            .try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;

        if next_token != token {
            return Err(ParserErrorInternal::UnexpectedCharacter(next));
        }

        Ok(next_token)
    }

    fn expect_next_token(&self, token: Token) -> Result<Token, ParserErrorInternal<'source>> {
        let next = self.next()?;

        let next_token = next
            .try_into()
            .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;

        if next_token != token {
            return Err(ParserErrorInternal::UnexpectedCharacter(next));
        }

        Ok(next_token)
    }

    fn advance_while<F: Fn(&char) -> bool>(&self, func: F) -> usize {
        while self.position() < self.iter.len() && func(&self.iter[self.position()]) {
            let mut pos = self.position.borrow_mut();
            *pos += 1;
        }

        self.position()
    }

    fn peek(&self) -> Result<char, ParserErrorInternal<'source>> {
        self.iter
            .get(self.position())
            .cloned()
            .ok_or(ParserErrorInternal::UnexpectedEndOfFile)
    }

    fn select_range(&self, range: Range<usize>) -> &[char] {
        &self.iter[range]
    }

    fn rewind(&self) {
        *self.position.borrow_mut() = self.position().saturating_sub(1);
    }

    fn rewind_to(&self, pos: usize) {
        *self.position.borrow_mut() = pos;
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

        let parser = Parser::new(s);

        let result = parser.parse_typedef().unwrap();

        let idents = smallvec![Identifier("Foo")];
        let members = vec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 29..39,
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
            span: 111..150,
        }];

        let expected = AstNodeType::StructDef(StructDef { idents, members });

        assert!(result.typ == expected);
        assert!(result.span == (s.find("typedef").unwrap()..s.len()));
    }

    #[test]
    fn parse_expression() {
        let s = r#"foo && bar;"#;

        let parser = Parser::new(s);

        let result = parser.parse_expression().unwrap();

        let expected = AstNode {
            typ: AstNodeType::Expression(Expression {
                lhs: Box::new(AstNode {
                    typ: AstNodeType::Identifier(Identifier("foo")),
                    span: (0..3),
                }),
                operator: Some(Operator::LogicalAnd),
                rhs: Some(Box::new(AstNode {
                    typ: AstNodeType::Expression(Expression {
                        lhs: Box::new(AstNode {
                            typ: AstNodeType::Identifier(Identifier("bar")),
                            span: (7..10),
                        }),
                        operator: None,
                        rhs: None,
                    }),
                    span: 7..10,
                })),
            }),
            span: 0..10,
        };

        assert!(result.typ == expected.typ);
        assert!(result.span == (0..s.len() - 1));
    }

    #[test]
    fn struct_with_conditional_field() {
        let s = r#"
        // start comment
        typedef struct {
            if (foo)
                char field;
        } Foo;"#;

        let parser = Parser::new(s);

        let result = parser.parse_typedef().unwrap();

        let idents = smallvec![Identifier("Foo")];

        let if_body = vec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 88..98,
        }];

        let if_expression_typ = AstNodeType::Identifier(Identifier("foo"));

        let expected_if = AstNodeType::ControlFlow(ControlFlow::If(IfCondition {
            condition: Box::new(AstNode {
                typ: AstNodeType::Expression(Expression {
                    lhs: Box::new(AstNode {
                        typ: if_expression_typ,
                        span: (67..70),
                    }),
                    operator: None,
                    rhs: None,
                }),
                span: 67..70,
            }),
            body: if_body,
        }));

        let members = vec![AstNode {
            typ: expected_if,
            span: 63..99,
        }];

        let expected = AstNodeType::StructDef(StructDef { idents, members });

        assert!(result.typ == expected);
        assert!(result.span == (s.find("typedef").unwrap()..s.len()));
    }

    #[test]
    fn parse_if_condition() {
        let s = r#"if (foo)
                char field;"#;

        let parser = Parser::new(s);

        let result = parser.parse_if_condition().unwrap();

        let body = vec![AstNode {
            typ: AstNodeType::VarDecl(VarDecl {
                typ: Identifier("char"),
                ident: Identifier("field"),
                array_size: None,
                is_local: false,
            }),
            span: 25..35,
        }];

        let typ = AstNodeType::Identifier(Identifier("foo"));

        let expected = AstNodeType::ControlFlow(ControlFlow::If(IfCondition {
            condition: Box::new(AstNode {
                typ: AstNodeType::Expression(Expression {
                    lhs: Box::new(AstNode { typ, span: (4..7) }),
                    operator: None,
                    rhs: None,
                }),
                span: 4..7,
            }),
            body,
        }));

        assert!(result.typ == expected);
        assert!(result.span == (s.find("if").unwrap()..s.len()));
    }
}
