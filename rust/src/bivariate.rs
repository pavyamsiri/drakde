#[derive(Debug, Clone)]
pub struct BivariateKDE {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

impl BivariateKDE {
    pub fn new(x: Vec<f64>, y: Vec<f64>, weights: Vec<f64>) -> Self {
        Self { x, y, weights }
    }

    fn kernel(&self, delta_x: f64, scale_length: f64) -> f64 {
        let coeff = ((2.0 * core::f64::consts::PI).sqrt() * scale_length).recip();
        let exp_arg = -0.5 * (delta_x / scale_length).powi(2);
        coeff * exp_arg.exp()
    }

    pub fn estimate_scalar(&self, x: f64, y: f64, scale_length: f64) -> f64 {
        let denom: f64 = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(xi, yi)| self.kernel(x - xi, scale_length) * self.kernel(y - yi, scale_length))
            .sum();

        let numer: f64 = self
            .x
            .iter()
            .zip(self.y.iter())
            .zip(self.weights.iter())
            .map(|((xi, yi), wi)| {
                wi * self.kernel(x - xi, scale_length) * self.kernel(y - yi, scale_length)
            })
            .sum();

        numer / denom
    }
}
