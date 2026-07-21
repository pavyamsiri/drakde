use drakde_rs::bivariate::BivariateKDE;
use std::time::Instant;

fn lcg(mut s: u64) -> u64 {
    // simple LCG
    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    s
}

fn main() {
    // build a tiny KDE object to access LUT and methods
    // create 32 dummy points
    let n = 32usize;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    for i in 0..n {
        xs.push((i as f64) * 0.1);
        ys.push(((i * 7) as f64) * 0.07);
        ws.push(1.0f64);
    }
    let kde = BivariateKDE::new(xs, ys, ws);

    // generate many random inputs in range [-5, 0]
    let m = 1_000_000usize; // one million inputs to get measurable times
    let mut inputs: Vec<f32> = Vec::with_capacity(m);
    let mut state: u64 = 123456789;
    for _ in 0..m {
        state = lcg(state);
        let v = ((state as i64 % 10000) as f32) / 10000.0f32 * -5.0f32; // [-5,0)
        inputs.push(v);
    }

    // warmup
    let _ = kde.exp_minimax_vec(&inputs[0..1024]);
    let _ = kde.exp_lut_vec(&inputs[0..1024]);
    let _ = kde.exp_std_vec(&inputs[0..1024]);

    // measure each method
    let runs = 3;
    let mut t_minimax = 0u128;
    let mut t_lut = 0u128;
    let mut t_std = 0u128;

    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = kde.exp_minimax_vec(&inputs);
        t_minimax += t0.elapsed().as_nanos();

        let t1 = Instant::now();
        let _ = kde.exp_lut_vec(&inputs);
        t_lut += t1.elapsed().as_nanos();

        let t2 = Instant::now();
        let _ = kde.exp_std_vec(&inputs);
        t_std += t2.elapsed().as_nanos();
    }

    println!("Inputs: {} elements", m);
    println!("Minimax avg time ms: {:.3}", (t_minimax as f64 / runs as f64) / 1e6);
    println!("LUT avg time ms:     {:.3}", (t_lut as f64 / runs as f64) / 1e6);
    println!("Std exp avg ms:      {:.3}", (t_std as f64 / runs as f64) / 1e6);
}
