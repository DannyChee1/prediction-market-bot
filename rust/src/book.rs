/// `bids` / `asks` hold the top 10 levels — kept for the depth-check gate
/// (P1 in tasks/todo.md) even though the current arb only reads best_ask.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct BookSnapshot {
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub timestamp: f64,
}

impl BookSnapshot {
    pub fn age_ms(&self, now_s: f64) -> Option<f64> {
        if self.timestamp <= 0.0 {
            None
        } else {
            Some((now_s - self.timestamp).max(0.0) * 1000.0)
        }
    }
}
