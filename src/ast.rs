use crate::error::{ParserErrorInternal, ParserErrorWithSpan};
use crate::tokens::Token;
use peekmore::{PeekMore, PeekMoreIterator};
use std::iter::Peekable;
use std::str::FromStr;

use std::{ops::Range, str::Chars};

use smallvec::{smallvec, SmallVec};

pub(crate) type Span = Range<usize>;

#[derive(Debug)]
struct VarDecl<'source> {
    typ: Identifier<'source>,
    ident: Identifier<'source>,
    array_size: Option<SmallVec<[u8; 2]>>,
}

#[derive(Debug)]
struct StructMember<'source> {
    decl: VarDecl<'source>,
    condition: Option<u8>,
}

#[derive(Debug)]
struct StructDef<'source> {
    idents: SmallVec<[Identifier<'source>; 3]>,
    members: SmallVec<[StructMember<'source>; 8]>,
}

#[derive(Debug)]
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

#[derive(Debug, Clone)]
struct Identifier<'source>(&'source str);

#[derive(Debug)]
enum AstNodeType<'source> {
    VarDecl,
    FuncCall,
    Expression,
    Statement,
    StructDef(StructDef<'source>),
    EnumDef,
    FuncDef,
    Identifier(Identifier<'source>),
    HexLiteral(u64),
    OctalLiteral(u64),
    BinaryLiteral(u64),
    DecimalLiteral(u64),
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
    iter: Peekable<Chars<'source>>,
    position: usize,
    line_num: usize,
}

impl<'source> Parser<'source> {
    pub fn new(input: &'source str) -> Self {
        let iter = input.chars().peekable();
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
    ) -> Result<SmallVec<[StructMember<'source>; 8]>, ParserErrorInternal<'source>> {
        let mut members = SmallVec::new();
        loop {
            self.chomp_whitespace();

            // Parse the first identifier (type) if it exists
            let next = self.peek()?;
            if next.is_ident_start() {
                let typ = self.parse_ident()?;

                self.chomp_whitespace();

                // Parse the second identifier (name) if it exists
                let next = self.peek()?;
                if next.is_ident_start() {
                    let name = self.parse_ident()?;

                    self.chomp_whitespace();

                    members.push(StructMember {
                        decl: VarDecl {
                            typ: typ.as_ident().expect("expected ident").clone(),
                            ident: name.as_ident().expect("expected ident").clone(),
                            array_size: None,
                        },
                        condition: None,
                    });

                    let next = self.next()?;
                    let next_token = next
                        .try_into()
                        .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;
                    match next_token {
                        Token::Semicolon => {
                            continue;
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
            let next_token = next
                .try_into()
                .map_err(|_e| ParserErrorInternal::UnexpectedCharacter(next))?;
            match next_token {
                Token::CloseBrace => {
                    self.next()?;
                    break;
                }
                _other => {
                    return Err(ParserErrorInternal::UnexpectedCharacter(next));
                }
            }
        }

        Ok(members)
    }

    fn parse_struct(
        &mut self,
        must_have_name: bool,
    ) -> Result<AstNode<'source>, ParserErrorInternal<'source>> {
        let start = self.position;

        self.chomp_whitespace();

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
                    members: SmallVec::new(),
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
        let mut struct_members = SmallVec::new();

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
        self.chomp_whitespace();

        let start = self.position;

        // Try parsing the "typedef" keyword
        let ident = self.parse_ident()?;

        // Next token must be whitespace
        if !self.chomp_whitespace() {
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
            self.chomp_whitespace();

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

    fn parse_root(&mut self) -> Result<AstNode<'source>, ParserErrorWithSpan> {
        todo!();
    }

    fn next(&mut self) -> Result<char, ParserErrorInternal<'source>> {
        let next = self
            .iter
            .next()
            .ok_or(ParserErrorInternal::UnexpectedEndOfFile);
        if next.is_ok() {
            self.position += 1;
        }

        next
    }

    fn advance_while<F: Fn(&char) -> bool>(&mut self, func: F) -> usize {
        while let Some(_) = self.iter.next_if(&func) {
            self.position += 1;
        }

        self.position
    }

    fn peek(&mut self) -> Result<char, ParserErrorInternal<'source>> {
        self.iter
            .peek()
            .cloned()
            .ok_or(ParserErrorInternal::UnexpectedEndOfFile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_number() {
        let s = "100";
        let mut parser = Parser::new(s);

        let result = parser.parse_number();
        assert!(result.is_ok());

        let result = result.unwrap();

        assert!(matches!(result.typ, AstNodeType::DecimalLiteral(100)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_hex_string() {
        let s = "0x100";
        let mut parser = Parser::new(s);

        let result = parser.parse_number();
        assert!(result.is_ok());

        let result = result.unwrap();

        assert!(matches!(result.typ, AstNodeType::HexLiteral(0x100)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_hex_string_with_alpha() {
        let s = "0xabc3";
        let mut parser = Parser::new(s);

        let result = parser.parse_number();
        assert!(result.is_ok());

        let result = result.unwrap();

        assert!(matches!(result.typ, AstNodeType::HexLiteral(0xabc3)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_binary_string() {
        let s = "0b101";
        let mut parser = Parser::new(s);

        let result = parser.parse_number();
        assert!(result.is_ok());

        let result = result.unwrap();

        assert!(matches!(result.typ, AstNodeType::BinaryLiteral(0b101)));
        assert!(result.span == (0..s.len()));
    }

    #[test]
    fn parse_octal_string() {
        let s = "01777";
        let mut parser = Parser::new(s);

        let result = parser.parse_number();
        assert!(result.is_ok());

        let result = result.unwrap();

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
    }
}
