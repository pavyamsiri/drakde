mod bivariate;

#[pyo3::pymodule]
mod _drakde {
    use numpy::PyReadonlyArray1;
    use pyo3::prelude::*;

    use crate::bivariate;

    #[pyclass(skip_from_py_object)]
    #[derive(Clone)]
    pub struct BivariateKDE(bivariate::BivariateKDE);

    #[pymethods]
    impl BivariateKDE {
        #[new]
        #[pyo3(signature = (x, y, weights=None))]
        pub fn new(
            x: PyReadonlyArray1<'_, f64>,
            y: PyReadonlyArray1<'_, f64>,
            weights: Option<PyReadonlyArray1<'_, f64>>,
        ) -> PyResult<Self> {
            let x_vec = x.as_array().to_vec();
            let y_vec = y.as_array().to_vec();

            if x_vec.len() != y_vec.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "x and y must have the same length",
                ));
            }

            let weights_vec = match weights {
                Some(w) => {
                    let w_vec = w.as_array().to_vec();
                    if w_vec.len() != x_vec.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "weights must have the same length as x and y",
                        ));
                    }
                    w_vec
                }
                None => vec![1.0; x_vec.len()],
            };

            Ok(BivariateKDE(bivariate::BivariateKDE {
                x: x_vec,
                y: y_vec,
                weights: weights_vec,
            }))
        }

        pub fn __repr__(&self) -> String {
            format!(
                "BivariateKDE(n_points={}, x_range=[{:.2}, {:.2}], y_range=[{:.2}, {:.2}])",
                self.0.x.len(),
                self.0.x.iter().copied().fold(f64::INFINITY, f64::min),
                self.0.x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                self.0.y.iter().copied().fold(f64::INFINITY, f64::min),
                self.0.y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        }

        pub fn estimate_scalar(&self, x: f64, y: f64, scale_length: f64) -> f64 {
            self.0.estimate_scalar(x, y, scale_length)
        }
    }

    #[pyfunction]
    fn hello() -> PyResult<String> {
        Ok("Hello".to_owned())
    }
}
