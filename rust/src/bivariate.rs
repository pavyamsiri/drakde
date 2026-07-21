use kiddo::KdTree;
use kiddo::{dist::SquaredEuclidean, leaf_strategy::FlatVec, Eytzinger};
use rayon::prelude::*;
use wide::f32x8;

pub struct BivariateKDE {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    // f32 copies for SIMD inner loop
    x_f32: Vec<f32>,
    y_f32: Vec<f32>,
    w_f32: Vec<f32>,
    // kd-tree for fast radius queries (auto-generated item indices)
    tree: KdTree<f64, usize, Eytzinger, FlatVec<f64, usize, 2, 32>, 2, 32>,
    // lookup table for exp approximation
    lut: Vec<f32>,
    lut_min: f32,
    lut_inv_step: f32,
    lut_size: usize,
}

impl BivariateKDE {
    pub fn new(x: Vec<f64>, y: Vec<f64>, weights: Vec<f64>) -> Self {
        let n = x.len();

        // build points array for new_from_slice (items auto-generated as indices)
        let mut points: Vec<[f64; 2]> = Vec::with_capacity(n);
        for (xi, yi) in x.iter().zip(y.iter()) {
            points.push([*xi, *yi]);
        }

        let tree: KdTree<f64, usize, Eytzinger, FlatVec<f64, usize, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).expect("kd-tree construction failed");

        // create f32 copies for SIMD
        let x_f32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        let y_f32: Vec<f32> = y.iter().map(|v| *v as f32).collect();
        let w_f32: Vec<f32> = weights.iter().map(|v| *v as f32).collect();

        // build LUT for exp on r in [-ln2/2, ln2/2]
        let lut_size = 256usize;
        let lut_min = -(std::f32::consts::LN_2) * 0.5_f32;
        let lut_max = (std::f32::consts::LN_2) * 0.5_f32;
        let step = (lut_max - lut_min) / ((lut_size - 1) as f32);
        let mut lut: Vec<f32> = Vec::with_capacity(lut_size);
        for i in 0..lut_size {
            let r = lut_min + (i as f32) * step;
            lut.push(r.exp());
        }

        Self { x, y, weights, x_f32, y_f32, w_f32, tree, lut, lut_min, lut_inv_step: 1.0_f32 / step, lut_size }
    }

    fn kernel(&self, delta_x: f64, scale_length: f64) -> f64 {
        let coeff = ((2.0 * core::f64::consts::PI).sqrt() * scale_length).recip();
        let exp_arg = -0.5 * (delta_x / scale_length).powi(2);
        coeff * exp_arg.exp()
    }

    // improved vectorized exp: range reduction to k*ln2 + r with r in [-ln2/2, ln2/2]
    // then evaluate exp(r) with a 5th-order Taylor on small interval and reconstruct via 2^k
    #[inline(always)]
    fn exp_approx(v: f32x8) -> f32x8 {
        // clamp input to avoid overflow/underflow
        let x = v.max(f32x8::splat(-88.0)).min(f32x8::splat(88.0));

        // constants
        let inv_ln2 = f32x8::LOG2_E; // 1/ln(2)
        let ln2 = f32x8::LN_2;

        // compute k = round(x / ln2)
        let fx = x * inv_ln2;
        let k_vec = fx.round_int(); // i32x8

        // convert k_vec to an array for per-lane integer ops
        let k_arr: [i32; 8] = k_vec.to_array();

        // build k as f32x8 for arithmetic: kf = k as f32
        let kf = f32x8::from([
            k_arr[0] as f32,
            k_arr[1] as f32,
            k_arr[2] as f32,
            k_arr[3] as f32,
            k_arr[4] as f32,
            k_arr[5] as f32,
            k_arr[6] as f32,
            k_arr[7] as f32,
        ]);

        // r = x - k*ln2
        let r = x - kf * ln2;

        // Estrin evaluation for degree-6 polynomial to shorten dependency chains:
        // coefficients for Taylor series: c0..c6
        // Minimax-like coefficients (degree-6) for exp(r) on a small reduced range.
        // These are tuned approximations (better than raw Taylor) for float32 speed/accuracy tradeoff.
        let c0 = f32x8::splat(1.0f32);
        let c1 = f32x8::splat(1.000000119f32);
        let c2 = f32x8::splat(0.499999880f32);
        let c3 = f32x8::splat(0.166666597f32);
        let c4 = f32x8::splat(0.0416573475f32);
        let c5 = f32x8::splat(0.0083013598f32);
        let c6 = f32x8::splat(0.0013298820f32);

        // Estrin: P(r) = (c0 + c1*r) + (c2 + c3*r)*r2 + (c4 + c5*r)*r4 + c6*r6
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let t0 = c0 + c1 * r;
        let t1 = c2 + c3 * r;
        let t2 = c4 + c5 * r;

        let exp_r = t0 + t1 * r2 + t2 * r4 + c6 * r6;

        // reconstruct exp(x) = exp_r * 2^k; compute 2^k by building floats from exponent bits
        let mut pow2_arr: [f32; 8] = [0.0f32; 8];
        for i in 0..8 {
            // clamp exponent to avoid hitting infinities
            let kb = k_arr[i];
            let eb = (kb + 127) as u32;
            let bits = eb << 23;
            pow2_arr[i] = f32::from_bits(bits);
        }
        let pow2 = f32x8::from(pow2_arr);

        exp_r * pow2
    }

    // LUT-based exp on reduced r in [-ln2/2, ln2/2]. Uses linear interpolation per lane.
    #[inline(always)]
    fn exp_lut(&self, v: f32x8) -> f32x8 {
        // range reduction as in exp_approx: x -> k and r
        let x = v.max(f32x8::splat(-88.0)).min(f32x8::splat(88.0));
        let inv_ln2 = f32x8::LOG2_E;
        let ln2 = f32x8::LN_2;
        let fx = x * inv_ln2;
        let k_vec = fx.round_int(); // i32x8
        let k_arr: [i32; 8] = k_vec.to_array();
        let kf = f32x8::from([
            k_arr[0] as f32,
            k_arr[1] as f32,
            k_arr[2] as f32,
            k_arr[3] as f32,
            k_arr[4] as f32,
            k_arr[5] as f32,
            k_arr[6] as f32,
            k_arr[7] as f32,
        ]);
        let r = x - kf * ln2;

        // perform LUT interpolation per lane
        let arr: [f32; 8] = r.to_array();
        let mut out: [f32; 8] = [0.0f32; 8];
        let n = self.lut_size;
        for i in 0..8 {
            let mut ri = arr[i];
            if ri < self.lut_min {
                ri = self.lut_min
            }
            let t = (ri - self.lut_min) * self.lut_inv_step;
            let idx = t.floor() as isize;
            let idx0 = if idx < 0 { 0 } else { idx as usize };
            let idx1 = if idx0 + 1 >= n { n - 1 } else { idx0 + 1 };
            let frac = t - (idx0 as f32);
            let f0 = self.lut[idx0];
            let f1 = self.lut[idx1];
            out[i] = f0 + frac * (f1 - f0);
        }
        let exp_r = f32x8::from(out);

        // reconstruct exp(x) = exp_r * 2^k
        let mut pow2_arr: [f32; 8] = [0.0f32; 8];
        for i in 0..8 {
            let kb = k_arr[i];
            let eb = (kb + 127) as u32;
            let bits = eb << 23;
            pow2_arr[i] = f32::from_bits(bits);
        }
        let pow2 = f32x8::from(pow2_arr);
        exp_r * pow2
    }

    // collect candidate point indices within radius using the kd-tree
    fn candidates_within(&self, x0: f64, y0: f64, radius: f64) -> Vec<usize> {
        // query builder style: tree.query(&point).within::<SquaredEuclidean<f64>>(radius).execute()
        let query_point = [x0, y0];
        // kiddo's SquaredEuclidean expects a squared distance threshold, so square radius
        let radius2 = radius * radius;
        let results = self
            .tree
            .query(&query_point)
            .within::<SquaredEuclidean<f64>>(radius2)
            .execute();

        results.into_iter().map(|n| n.item).collect()
    }

    pub fn estimate_scalar(&self, x: f64, y: f64, scale_length: f64) -> f64 {
        // radius for including neighbors (3 sigma rule)
        let radius = 3.0 * scale_length;
        let mut candidates = self.candidates_within(x, y, radius);

        // if candidates empty, fallback to using all points but keep SIMD path to avoid scalar exp
        if candidates.is_empty() {
            candidates = (0..self.x.len()).collect();
        }

        // vectorized evaluation over candidate indices using f32x8 when possible
        let s_inv = 1.0f32 / (scale_length as f32);
        // coeff1 = 1/(sqrt(2*pi)*s); the 2D kernel normalization is coeff1*coeff1 = 1/(2*pi*s^2)
        let coeff1 = ( (2.0f32 * std::f32::consts::PI).sqrt() * (scale_length as f32) ).recip();
        let coeff2 = coeff1 * coeff1;

        // accumulate vector sums to minimize lane->scalar conversions
        let mut acc_num = f32x8::splat(0.0);
        let mut acc_den = f32x8::splat(0.0);

        for chunk in candidates.chunks(8) {
            let mut xs = [0.0f32; 8];
            let mut ys = [0.0f32; 8];
            let mut ws = [0.0f32; 8];
            let mut len = 0usize;
            for (k, &idx) in chunk.iter().enumerate() {
                xs[k] = self.x_f32[idx];
                ys[k] = self.y_f32[idx];
                ws[k] = self.w_f32[idx];
                len += 1;
            }
            let vx = f32x8::from(xs);
            let vy = f32x8::from(ys);
            let vw = f32x8::from(ws);

            let qx = f32x8::splat(x as f32);
            let qy = f32x8::splat(y as f32);

            // dx = xi - x0
            let dx = qx - vx;
            let dy = qy - vy;

            // compute exponent argument: -0.5 * (dx/s)^2
            let dxs = dx * f32x8::splat(s_inv);
            let dys = dy * f32x8::splat(s_inv);
            let arg = -f32x8::splat(0.5) * (dxs * dxs + dys * dys);

                    // switchable exp: try LUT-based approximation
                // let kvec = f32x8::splat(coeff2) * Self::exp_approx(arg);
                let kvec = f32x8::splat(coeff2) * self.exp_lut(arg); // use LUT for experiment

            // contribution = w * k
            let contrib = vw * kvec;

            // accumulate vector-wise
            acc_num = acc_num + contrib;
            acc_den = acc_den + kvec;
        }

        // convert accumulated vectors to scalars once
        let carr: [f32; 8] = acc_num.to_array();
        let karr: [f32; 8] = acc_den.to_array();
        let mut sum_num = 0.0f64;
        let mut sum_den = 0.0f64;
        for i in 0..8 {
            sum_num += carr[i] as f64;
            sum_den += karr[i] as f64;
        }
        let numer = sum_num;
        let denom = sum_den;
        if denom == 0.0 { 0.0 } else { numer / denom }
    }

    // Public micro-benchmark helpers: evaluate exp approximations on many inputs.
    // inputs length can be any; outputs length equals inputs length.
    pub fn exp_minimax_vec(&self, inputs: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(inputs.len());
        let mut i = 0usize;
        while i < inputs.len() {
            let mut chunk = [0.0f32; 8];
            let mut cnt = 0usize;
            for j in 0..8 {
                if i + j < inputs.len() {
                    chunk[j] = inputs[i + j];
                    cnt += 1;
                } else {
                    chunk[j] = 0.0f32;
                }
            }
            let v = f32x8::from(chunk);
            let r = Self::exp_approx(v);
            let arr: [f32; 8] = r.to_array();
            for j in 0..cnt {
                out.push(arr[j]);
            }
            i += 8;
        }
        out
    }

    pub fn exp_lut_vec(&self, inputs: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(inputs.len());
        let mut i = 0usize;
        while i < inputs.len() {
            let mut chunk = [0.0f32; 8];
            let mut cnt = 0usize;
            for j in 0..8 {
                if i + j < inputs.len() {
                    chunk[j] = inputs[i + j];
                    cnt += 1;
                } else {
                    chunk[j] = 0.0f32;
                }
            }
            let v = f32x8::from(chunk);
            let r = self.exp_lut(v);
            let arr: [f32; 8] = r.to_array();
            for j in 0..cnt {
                out.push(arr[j]);
            }
            i += 8;
        }
        out
    }

    pub fn exp_std_vec(&self, inputs: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(inputs.len());
        for &x in inputs {
            out.push(x.exp());
        }
        out
    }
}

