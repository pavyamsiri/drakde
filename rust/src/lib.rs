#[pyo3::pymodule]
mod _drakde {
    use pyo3::prelude::*;

    #[pyfunction]
    fn hello() -> PyResult<String> {
        Ok("Hello".to_owned())
    }
}
