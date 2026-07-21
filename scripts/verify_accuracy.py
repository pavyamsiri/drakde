#!/usr/bin/env python3
import math
import random

from drakde._drakde import BivariateKDE

import numpy as np
random.seed(1)
N = 400
xs = np.array([random.uniform(-10.0, 10.0) for _ in range(N)])
ys = np.array([random.uniform(-10.0, 10.0) for _ in range(N)])
ws = np.array([random.uniform(0.1, 2.0) for _ in range(N)])

scale = 10.0  # use large scale so KD-tree radius includes all points for fair comparison

smoother = BivariateKDE(xs, ys, ws)

# pick query points
M = 80
qxs = np.array([random.uniform(-10.0, 10.0) for _ in range(M)])
qys = np.array([random.uniform(-10.0, 10.0) for _ in range(M)])

# Rust results (batch)
rust_out = list(smoother.estimate_vector(qxs, qys, scale))

# reference scalar implementation
def reference_kde(xq, yq, xs, ys, ws, s):
    numer = 0.0
    denom = 0.0
    coeff = 1.0 / (math.sqrt(2.0 * math.pi) * s)
    for xi, yi, wi in zip(xs, ys, ws):
        kx = coeff * math.exp(-0.5 * ((xi - xq) / s) ** 2)
        ky = coeff * math.exp(-0.5 * ((yi - yq) / s) ** 2)
        k = kx * ky
        numer += wi * k
        denom += k
    return 0.0 if denom == 0.0 else numer / denom

# compute a float32-based reference to mimic SIMD path more closely
xs_f32 = xs.astype(np.float32)
ys_f32 = ys.astype(np.float32)
ws_f32 = ws.astype(np.float32)
coeff1 = 1.0 / (math.sqrt(2.0 * math.pi) * float(scale))

ref_out = []
for xq, yq in zip(qxs, qys):
    dx = (xs_f32 - np.float32(xq)) / np.float32(scale)
    dy = (ys_f32 - np.float32(yq)) / np.float32(scale)
    arg = -0.5 * (dx * dx + dy * dy)
    k = np.float32(coeff1 * 1.0) * np.exp(arg.astype(np.float32)).astype(np.float32)
    # 2D kernel should be coeff1^2 * exp(arg)
    k = k * np.float32(coeff1)
    numer = np.sum(ws_f32 * k, dtype=np.float32)
    denom = np.sum(k, dtype=np.float32)
    ref_out.append(float(0.0 if denom == 0.0 else numer / denom))

# compare
max_rel = 0.0
for i, (r, refv) in enumerate(zip(rust_out, ref_out)):
    denom = refv if refv != 0.0 else 1.0
    rel = abs((r - refv) / denom)
    if rel > max_rel:
        max_rel = rel
    if i < 8:
        print(f"i={i}: rust={r:.6e}, ref={refv:.6e}, rel={rel:.3e}")

print(f"Checked {len(ref_out)} points; max relative error = {max_rel:.3e}")
if max_rel > 1e-4:
    print("ERROR: relative error exceeded threshold")
    raise SystemExit(2)
else:
    print("PASS: accuracy within threshold")
