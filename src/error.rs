use thiserror::Error;

use crate::{tokens, ast::Span};

#[derive(Error, Debug)]
pub enum ParserErrorInternal<'source> {
    #[error("unexpected token {0:?}, expected {1}")]
    UnexpectedToken(char, &'static str),

    #[error("unexpected character {0:?}")]
    UnexpectedCharacter(char),

    #[error("expected {0:?}")]
    UnexpectedWithDescription(&'static str),

    #[error("unexpected end of file")]
    UnexpectedEndOfFile,

    #[error("placeholder")]
    Placeholder(&'source str)
}

#[derive(Error, Debug)]
pub enum ParserErrorWithSpan<'source> {
    #[error("An error occurred at {0:?}: {1:}")]
    InternalError(Span, ParserErrorInternal<'source>),
}