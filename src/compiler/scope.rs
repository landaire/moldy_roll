type TypeIndex = usize;

pub enum Scope<'source> {
    Function {
        symbols: HashMap<&'source str, TypeIndex>,
    },
}

pub(crate) struct ScopeContainer<'source> {}
