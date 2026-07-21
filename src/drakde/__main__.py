import numpy as np

from drakde._drakde import BivariateKDE
import time

show: bool = False

if __name__ == "__main__":
    data = np.load("./data/data.npz")
    x_mesh = data["x_mesh"]
    y_mesh = data["y_mesh"]
    mean_nu = data["mean_nu"]

    mask = np.isfinite(mean_nu)
    smoother = BivariateKDE(
        x_mesh[mask].flatten(), y_mesh[mask].flatten(), mean_nu[mask].flatten()
    )
    print(smoother)

    local_scale: float = 0.1
    smooth_scale: float = 1.0
    local_map = np.zeros_like(x_mesh)
    smooth_map = np.zeros_like(x_mesh)

    start = time.perf_counter()

    # Flatten meshes and call Rust batch estimator to avoid Python-level loops
    flat_x = x_mesh.flatten()
    flat_y = y_mesh.flatten()

    flat_local = smoother.estimate_vector(flat_x, flat_y, local_scale)
    flat_smooth = smoother.estimate_vector(flat_x, flat_y, smooth_scale)

    local_map = np.asarray(flat_local).reshape(x_mesh.shape)
    smooth_map = np.asarray(flat_smooth).reshape(x_mesh.shape)

    elapsed = time.perf_counter() - start
    print(
        f"Took {elapsed:2f} seconds to fill out a grid {x_mesh.shape[0]}x{x_mesh.shape[1]} or {x_mesh.shape[0] * x_mesh.shape[1]} points"
    )

    if show:
        from matplotlib import pyplot as plt
        from matplotlib import colors as mplcolors

        fig = plt.figure()
        axes = fig.add_subplot(111)

        axes.pcolormesh(
            x_mesh,
            y_mesh,
            local_map / smooth_map - 1.0,
            cmap="seismic",
            norm=mplcolors.Normalize(vmin=-0.05, vmax=0.05),
        )
        plt.show()
        plt.close()
