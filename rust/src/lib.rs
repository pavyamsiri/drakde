pub mod bivariate;

#[pyo3::pymodule]
mod _drakde {
    use numpy::{AllowTypeChange, PyArrayLike1};
    use pyo3::prelude::*;

    use crate::bivariate::{self, BivariateGaussian};

    #[pyclass(skip_from_py_object)]
    pub struct BivariateKDE(bivariate::BivariateKDE);

    impl Clone for BivariateKDE {
        fn clone(&self) -> Self {
            let inner = &self.0;
            BivariateKDE(bivariate::BivariateKDE::new(
                inner.x.clone(),
                inner.y.clone(),
                inner.weights.clone(),
            ))
        }
    }

    use rayon::prelude::*;

    #[pymethods]
    impl BivariateKDE {
        #[new]
        #[pyo3(signature = (x, y, weights=None))]
        pub fn new(
            x: PyArrayLike1<'_, f32, AllowTypeChange>,
            y: PyArrayLike1<'_, f32, AllowTypeChange>,
            weights: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
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

            Ok(BivariateKDE(bivariate::BivariateKDE::new(
                x_vec,
                y_vec,
                weights_vec,
            )))
        }

        pub fn __repr__(&self) -> String {
            format!(
                "BivariateKDE(num_points={}, x_range=[{:.2}, {:.2}], y_range=[{:.2}, {:.2}])",
                self.0.x.len(),
                self.0.x.iter().copied().fold(f32::INFINITY, f32::min),
                self.0.x.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                self.0.y.iter().copied().fold(f32::INFINITY, f32::min),
                self.0.y.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        }

        #[pyo3(signature = (x, y, scale_length, num_sigma=4.0))]
        pub fn estimate_scalar(&self, x: f32, y: f32, scale_length: f32, num_sigma: f32) -> f32 {
            self.0
                .estimate_scalar::<BivariateGaussian>(x, y, scale_length, num_sigma)
        }

        /// Estimate for many (x,y) pairs in a single call. This avoids Python-level loops.
        /// xs and ys must be 1-D arrays of the same length. Returns a NumPy array of results.
        #[pyo3(signature = (xs, ys, scale_length, num_sigma=4.0))]
        pub fn estimate_vector<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            xs: PyArrayLike1<'py, f32, AllowTypeChange>,
            ys: PyArrayLike1<'py, f32, AllowTypeChange>,
            scale_length: f32,
            num_sigma: f32,
        ) -> pyo3::PyResult<pyo3::Py<numpy::PyArray1<f32>>> {
            let xs_slice = xs.as_slice()?;
            let ys_slice = ys.as_slice()?;
            if xs_slice.len() != ys_slice.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "xs and ys must have the same length",
                ));
            }
            let n = xs_slice.len();

            // Parallel evaluation using rayon
            let results: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| {
                    self.0.estimate_scalar::<BivariateGaussian>(
                        xs_slice[i],
                        ys_slice[i],
                        scale_length,
                        num_sigma,
                    )
                })
                .collect();

            let arr = numpy::PyArray1::from_vec(py, results);
            // convert borrowed reference into owned Py<PyArray1<f64>>
            Ok(arr.to_owned().into())
        }
    }
}
