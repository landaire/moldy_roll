use crate::error::ParserErrorInternal;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Token {
    Semicolon,
    OpenBrace,
    CloseBrace,
    OpenParen,
    CloseParen,
    AngleBracketOpen,
    AngleBracketClose,
    Whitespace,
    Quote,
    Operator(Operator),
}

impl TryFrom<char> for Token {
    type Error = ();

    fn try_from(value: char) -> Result<Self, Self::Error> {
        let token = match value {
            ';' => Token::Semicolon,
            '{' => Token::OpenBrace,
            '}' => Token::CloseBrace,
            '(' => Token::OpenParen,
            ')' => Token::CloseParen,
            '<' => Token::AngleBracketOpen,
            '>' => Token::AngleBracketClose,
            '\"' => Token::Quote,
            '*' => Token::Operator(Operator::Multiply),
            '/' => Token::Operator(Operator::Divide),
            '-' => Token::Operator(Operator::Subtract),
            '+' => Token::Operator(Operator::Add),
            '^' => Token::Operator(Operator::Xor),
            '|' => Token::Operator(Operator::BitOr),
            '!' => Token::Operator(Operator::Not),
            '=' => Token::Operator(Operator::Assignment),
            other if other.is_ascii_whitespace() => Token::Whitespace,
            other => {
                return Err(());
            }
        };

        Ok(token)
    }
}

impl Token {
    pub fn as_unary_operator(&self) -> Option<Operator> {
        match self {
            Token::Operator(op) => match op {
                Operator::Add | Operator::Subtract | Operator::LogicalNot => Some(op.clone()),
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Xor,
    BitOr,
    BitAnd,
    BitNot,
    LeftShift,
    RightShift,
    LogicalNot,
    LogicalOr,
    LogicalAnd,
    LogicalEquals,
    LogicalNotEquals,
    Not,
    Assignment,
}
