#!/usr/bin/env python3
import time
import numpy as np
from scipy.stats import gaussian_kde

# Load the same data used by __main__.py
data = np.load('data/data.npz')
x_mesh = data['x_mesh']
y_mesh = data['y_mesh']
mean_nu = data['mean_nu']
mask = np.isfinite(mean_nu)
xs = x_mesh[mask].flatten()
ys = y_mesh[mask].flatten()
ws = mean_nu[mask].flatten()

pts = np.vstack([xs, ys])  # shape (2, n)

# SciPy gaussian_kde expects weights parameter (supported in recent SciPy), but default bandwidth differs.
start = time.perf_counter()
# use gaussian_kde with weights if available
try:
    kde = gaussian_kde(pts, weights=ws)
except TypeError:
    # older SciPy: normalize weights into repeated points? fallback: ignore weights
    kde = gaussian_kde(pts)

flat_x = x_mesh.flatten()
flat_y = y_mesh.flatten()
coords = np.vstack([flat_x, flat_y])

start_eval = time.perf_counter()
dens = kde(coords)
elapsed_eval = time.perf_counter() - start_eval
elapsed_total = time.perf_counter() - start
print(f"SciPy gaussian_kde: total elapsed {elapsed_total:.3f}s, evaluation elapsed {elapsed_eval:.3f}s")
print(f"Output len: {len(dens)}")
