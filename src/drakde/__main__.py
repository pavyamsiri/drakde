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
    for i in range(x_mesh.shape[0]):
        for j in range(x_mesh.shape[1]):
            current_x = x_mesh[i, j]
            current_y = y_mesh[i, j]
            local_map[i, j] = smoother.estimate_scalar(
                current_x, current_y, local_scale
            )
            smooth_map[i, j] = smoother.estimate_scalar(
                current_x, current_y, smooth_scale
            )
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
