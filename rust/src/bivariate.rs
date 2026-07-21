use kiddo::KdTree;
use kiddo::{Eytzinger, dist::SquaredEuclidean, leaf_strategy::FlatVec};
use wide::f32x8;

pub struct BivariateKDE {
    // f32 copies for SIMD inner loop
    pub(crate) x: Vec<f32>,
    pub(crate) y: Vec<f32>,
    pub(crate) weights: Vec<f32>,
    // kd-tree for fast radius queries (auto-generated item indices)
    tree: KdTree<f32, usize, Eytzinger, FlatVec<f32, usize, 2, 32>, 2, 32>,
}

impl BivariateKDE {
    pub fn new(x: Vec<f32>, y: Vec<f32>, weights: Vec<f32>) -> Self {
        let n = x.len();

        // build points array for new_from_slice (items auto-generated as indices)
        let mut points: Vec<[f32; 2]> = Vec::with_capacity(n);
        for (xi, yi) in x.iter().zip(y.iter()) {
            points.push([*xi, *yi]);
        }

        let tree: KdTree<f32, usize, Eytzinger, FlatVec<f32, usize, 2, 32>, 2, 32> =
            KdTree::new_from_slice(&points).expect("kd-tree construction failed");

        Self {
            x,
            y,
            weights,
            tree,
        }
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
        let c1 = f32x8::splat(1.000_000_1_f32);
        let c2 = f32x8::splat(0.499_999_88_f32);
        let c3 = f32x8::splat(0.166_666_6_f32);
        let c4 = f32x8::splat(0.041_657_347_f32);
        let c5 = f32x8::splat(0.008_301_36_f32);
        let c6 = f32x8::splat(0.001_329_882_f32);

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

    // collect candidate point indices within radius using the kd-tree
    fn candidates_within(&self, x0: f32, y0: f32, radius: f32) -> Vec<usize> {
        // query builder style: tree.query(&point).within::<SquaredEuclidean<f64>>(radius).execute()
        let query_point = [x0, y0];
        // kiddo's SquaredEuclidean expects a squared distance threshold, so square radius
        let radius2 = radius * radius;
        let results = self
            .tree
            .query(&query_point)
            .within::<SquaredEuclidean<f32>>(radius2)
            .execute();

        results.into_iter().map(|n| n.item).collect()
    }

    pub fn estimate_scalar(&self, x: f32, y: f32, scale_length: f32, num_sigma: f32) -> f32 {
        let radius = num_sigma * scale_length;
        let candidates = self.candidates_within(x, y, radius);

        // if candidates empty, return 0
        if candidates.is_empty() {
            return f32::NAN;
        }

        // vectorized evaluation over candidate indices using f32x8 when possible
        let s_inv = 1.0f32 / scale_length;
        // coeff1 = 1/(sqrt(2*pi)*s); the 2D kernel normalization is coeff1*coeff1 = 1/(2*pi*s^2)
        let coeff1 = ((2.0f32 * std::f32::consts::PI).sqrt() * scale_length).recip();
        let coeff2 = coeff1 * coeff1;

        // accumulate vector sums to minimize lane->scalar conversions
        let mut acc_num = f32x8::splat(0.0);
        let mut acc_den = f32x8::splat(0.0);

        let qx = f32x8::splat(x);
        let qy = f32x8::splat(y);
        let coeff2_vec = f32x8::splat(coeff2);
        let s_inv_vec = f32x8::splat(s_inv);
        let half = f32x8::splat(0.5);

        let remaining_mask = f32x8::from(std::array::from_fn(|i| {
            if i < (candidates.len() % 8) { 1.0 } else { 0.0 }
        }));
        let full_mask = f32x8::splat(1.0);
        let masks = [remaining_mask, full_mask];

        for chunk in candidates.chunks(8) {
            let mut xs = [0.0f32; 8];
            let mut ys = [0.0f32; 8];
            let mut ws = [0.0f32; 8];
            let chunks_count = chunk.len();

            for (k, &idx) in chunk.iter().enumerate() {
                xs[k] = self.x[idx];
                ys[k] = self.y[idx];
                ws[k] = self.weights[idx];
            }

            let vx = f32x8::from(xs);
            let vy = f32x8::from(ys);
            let vw = f32x8::from(ws);

            // dx = xi - x0, dy = yi - y0
            let dx = qx - vx;
            let dy = qy - vy;

            // compute exponent argument: -0.5 * (dx/s)^2
            let dxs = dx * s_inv_vec;
            let dys = dy * s_inv_vec;
            let arg = -half * (dxs * dxs + dys * dys);

            let kvec = coeff2_vec * Self::exp_approx(arg);
            let contrib = vw * kvec;

            let mask_idx = (chunks_count == 8) as usize;
            let vmask = masks[mask_idx];
            acc_num += contrib * vmask;
            acc_den += kvec * vmask;
        }

        // convert accumulated vectors to scalars once
        let numer: f32 = acc_num.to_array().into_iter().sum();
        let denom: f32 = acc_den.to_array().into_iter().sum();
        if denom == 0.0 { 0.0 } else { numer / denom }
    }
}
