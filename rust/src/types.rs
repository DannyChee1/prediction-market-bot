use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rust_decimal::prelude::*;

/// Format f64 as a price string with 4 decimal places (tick precision).
pub fn f64_to_price_str(val: f64) -> String {
    format!("{:.4}", val)
}

/// Format f64 as a size string with 1 decimal place.
pub fn f64_to_size_str(val: f64) -> String {
    format!("{:.1}", val)
}

/// Format f64 as a Decimal-compatible string (up to 6 decimal places).
pub fn f64_to_amount_str(val: f64) -> String {
    format!("{:.6}", val)
}

/// Convert f64 to rust_decimal::Decimal.
pub fn f64_to_decimal(val: f64) -> PyResult<Decimal> {
    Decimal::from_f64(val).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Cannot convert {val} to Decimal"))
    })
}

/// Recursively convert a serde_json::Value to a Python object.
pub fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py).unwrap().to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().into_any().unbind())
            } else {
                Ok(n.to_string().into_pyobject(py).unwrap().into_any().unbind())
            }
        }
        serde_json::Value::String(s) => {
            Ok(s.into_pyobject(py).unwrap().into_any().unbind())
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr
                .iter()
                .map(|v| json_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            let list = PyList::new(py, items)?;
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Convert a polyfill-rs PostOrderResponse to a Python dict matching
/// the py-clob-client response format expected by orders.py.
///
/// Adds both polyfill-rs field names and py-clob-client field names
/// for maximum compatibility.
pub fn post_response_to_py(
    py: Python<'_>,
    resp: &polyfill_rs::PostOrderResponse,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    dict.set_item("success", resp.success)?;
    dict.set_item("orderID", &resp.order_id)?;
    dict.set_item("order_id", &resp.order_id)?;
    dict.set_item("id", &resp.order_id)?;

    if let Some(ref msg) = resp.error_msg {
        dict.set_item("errorMsg", msg)?;
    }

    if let Some(ref making) = resp.making_amount {
        dict.set_item("makingAmount", making.to_string())?;
    }

    if let Some(ref taking) = resp.taking_amount {
        dict.set_item("takingAmount", taking.to_string())?;
    }

    // Determine status from response fields (matches CLOB API behavior).
    // taking_amount can be Some(0) for GTC orders with no immediate match —
    // only treat as MATCHED if taking_amount is present AND > 0.
    let status = if !resp.success {
        "REJECTED"
    } else if resp.taking_amount.map_or(false, |v| !v.is_zero()) {
        "MATCHED"
    } else {
        "LIVE"
    };
    dict.set_item("status", status)?;

    if !resp.transaction_hashes.is_empty() {
        let hashes = PyList::new(
            py,
            resp.transaction_hashes
                .iter()
                .map(|s| s.into_pyobject(py).unwrap().into_any().unbind())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("transactionHashes", hashes)?;
    }

    Ok(dict.into_any().unbind())
}

/// Serialize any serde::Serialize type to a Python dict via JSON intermediary.
pub fn serialize_to_py<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<PyObject> {
    let json_val = serde_json::to_value(value)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &json_val)
}
