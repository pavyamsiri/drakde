pub mod bivariate;

#[pyo3::pymodule]
mod _drakde {
    use numpy::{AllowTypeChange, PyArrayLike1};
    use pyo3::prelude::*;

    use crate::bivariate::{self, KernelKind};

    #[pyclass(from_py_object, eq, eq_int)]
    #[derive(Clone, Copy, PartialEq)]
    pub enum PyKernelKind {
        Gaussian = 0,
        Epanechnikov = 1,
        Quartic = 2,
    }

    impl From<PyKernelKind> for KernelKind {
        fn from(k: PyKernelKind) -> Self {
            match k {
                PyKernelKind::Gaussian => KernelKind::Gaussian,
                PyKernelKind::Epanechnikov => KernelKind::Epanechnikov,
                PyKernelKind::Quartic => KernelKind::Quartic,
            }
        }
    }

    #[pyclass(skip_from_py_object)]
    pub struct BivariateKDE {
        inner: bivariate::BivariateKDE,
        kernel: KernelKind,
    }

    impl Clone for BivariateKDE {
        fn clone(&self) -> Self {
            BivariateKDE {
                inner: bivariate::BivariateKDE::new(
                    self.inner.x.clone(),
                    self.inner.y.clone(),
                    self.inner.weights.clone(),
                ),
                kernel: self.kernel,
            }
        }
    }

    use rayon::prelude::*;

    #[pymethods]
    impl BivariateKDE {
        #[new]
        #[pyo3(signature = (x, y, weights=None, kernel=PyKernelKind::Gaussian))]
        pub fn new(
            x: PyArrayLike1<'_, f32, AllowTypeChange>,
            y: PyArrayLike1<'_, f32, AllowTypeChange>,
            weights: Option<PyArrayLike1<'_, f32, AllowTypeChange>>,
            kernel: PyKernelKind,
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

            Ok(BivariateKDE {
                inner: bivariate::BivariateKDE::new(x_vec, y_vec, weights_vec),
                kernel: kernel.into(),
            })
        }

        pub fn __repr__(&self) -> String {
            format!(
                "BivariateKDE(num_points={}, x_range=[{:.2}, {:.2}], y_range=[{:.2}, {:.2}])",
                self.inner.x.len(),
                self.inner.x.iter().copied().fold(f32::INFINITY, f32::min),
                self.inner
                    .x
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max),
                self.inner.y.iter().copied().fold(f32::INFINITY, f32::min),
                self.inner
                    .y
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max),
            )
        }

        #[pyo3(signature = (x, y, scale_length))]
        pub fn estimate_scalar(&self, x: f32, y: f32, scale_length: f32) -> f32 {
            self.inner.estimate(x, y, scale_length, self.kernel)
        }

        /// Estimate for many (x,y) pairs in a single call. This avoids Python-level loops.
        /// xs and ys must be 1-D arrays of the same length. Returns a NumPy array of results.
        #[pyo3(signature = (xs, ys, scale_length))]
        pub fn estimate_vector<'py>(
            &self,
            py: pyo3::prelude::Python<'py>,
            xs: PyArrayLike1<'py, f32, AllowTypeChange>,
            ys: PyArrayLike1<'py, f32, AllowTypeChange>,
            scale_length: f32,
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
                    self.inner
                        .estimate(xs_slice[i], ys_slice[i], scale_length, self.kernel)
                })
                .collect();

            let arr = numpy::PyArray1::from_vec(py, results);
            // convert borrowed reference into owned Py<PyArray1<f64>>
            Ok(arr.to_owned().into())
        }
    }
}
