use pyo3::prelude::*;

/// Immutable snapshot of an order book side, returned by BookFeed.snapshot().
///
/// Python code accesses fields directly:
///   snap.best_bid, snap.best_ask, snap.bids[0], snap.asks[0]
///
/// Compatible with the existing snapshot_from_live() construction in feeds.py.
#[pyclass]
#[derive(Clone)]
pub struct BookSnapshot {
    #[pyo3(get)]
    pub best_bid: Option<f64>,
    #[pyo3(get)]
    pub best_ask: Option<f64>,
    /// Top N bids as (price, size) tuples, highest price first.
    #[pyo3(get)]
    pub bids: Vec<(f64, f64)>,
    /// Top N asks as (price, size) tuples, lowest price first.
    #[pyo3(get)]
    pub asks: Vec<(f64, f64)>,
    /// Timestamp of the last book update (seconds since epoch).
    #[pyo3(get)]
    pub timestamp: f64,
}

#[pymethods]
impl BookSnapshot {
    #[new]
    #[pyo3(signature = (best_bid=None, best_ask=None, bids=Vec::new(), asks=Vec::new(), timestamp=0.0))]
    fn new(
        best_bid: Option<f64>,
        best_ask: Option<f64>,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        timestamp: f64,
    ) -> Self {
        Self {
            best_bid,
            best_ask,
            bids,
            asks,
            timestamp,
        }
    }

    /// Size at best bid, or None if no bids.
    #[getter]
    fn size_at_best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(_, size)| *size)
    }

    /// Size at best ask, or None if no asks.
    #[getter]
    fn size_at_best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(_, size)| *size)
    }

    /// Bid-ask spread, or None if either side is empty.
    #[getter]
    fn spread(&self) -> Option<f64> {
        match (self.best_ask, self.best_bid) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Mid price, or None if either side is empty.
    #[getter]
    fn mid(&self) -> Option<f64> {
        match (self.best_ask, self.best_bid) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BookSnapshot(bid={:?}, ask={:?}, bids={}, asks={})",
            self.best_bid,
            self.best_ask,
            self.bids.len(),
            self.asks.len(),
        )
    }
}
