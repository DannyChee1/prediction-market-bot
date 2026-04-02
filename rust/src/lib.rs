use pyo3::prelude::*;

mod book;
mod client;
mod feed;
mod types;

pub(crate) use client::RUNTIME;

#[pymodule]
fn polybot_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<client::OrderClient>()?;
    m.add_class::<feed::BookFeed>()?;
    m.add_class::<feed::PriceFeed>()?;
    m.add_class::<feed::BinanceFeed>()?;
    m.add_class::<feed::UserFeed>()?;
    m.add_class::<book::BookSnapshot>()?;
    Ok(())
}
