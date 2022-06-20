use crate::error::ParserErrorInternal;

#[derive(Copy, Clone, Eq, PartialEq)]
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
            other if other.is_ascii_whitespace() => {
                Token::Whitespace
            }
            other => {
                return Err(());
            }
        };

        Ok(token)
    }
}