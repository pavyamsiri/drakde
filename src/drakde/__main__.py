from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from drakde._drakde import BivariateKDE, PyKernelKind
import time
from numpy.lib.npyio import NpzFile

if TYPE_CHECKING:
    from optype import numpy as onp

show: bool = True

if __name__ == "__main__":
    data = np.load("./data/data.npz")  # pyright: ignore[reportAny]
    assert isinstance(data, NpzFile)
    x_mesh: onp.Array2D[np.float32] = data["x_mesh"].astype(np.float32)
    y_mesh: onp.Array2D[np.float32] = data["y_mesh"].astype(np.float32)
    mean_nu: onp.Array2D[np.float32] = data["mean_nu"].astype(np.float32)

    mask = np.isfinite(mean_nu)
    smoother = BivariateKDE(
        x_mesh[mask].flatten(),
        y_mesh[mask].flatten(),
        mean_nu[mask].flatten(),
        kernel=PyKernelKind.Quartic,
    )

    bin_size_xy: float = 0.125
    local_scale: float = 0.125
    smooth_scale: float = 8 * local_scale

    display_xmin: float = -14.0
    display_xmax: float = -2.0
    display_ymin: float = -6.0
    display_ymax: float = 6.0

    display_bin_size: float = 0.5 * bin_size_xy
    max_contrast: float = 0.05

    display_x_bins = np.arange(
        display_xmin, display_xmax + display_bin_size, display_bin_size
    )
    display_y_bins = np.arange(
        display_ymin, display_ymax + display_bin_size, display_bin_size
    )
    display_x_centres = 0.5 * (display_x_bins[:-1] + display_x_bins[1:])
    display_y_centres = 0.5 * (display_y_bins[:-1] + display_y_bins[1:])
    display_x_mesh, display_y_mesh = np.meshgrid(display_x_centres, display_y_centres)

    display_shape = display_x_mesh.shape
    flat_x = display_x_mesh.flatten()
    flat_y = display_y_mesh.flatten()

    start = time.perf_counter()
    flat_local = smoother.estimate_vector(flat_x, flat_y, local_scale)
    flat_smooth = smoother.estimate_vector(flat_x, flat_y, smooth_scale)
    elapsed = time.perf_counter() - start
    print(
        f"Took {elapsed:2f} seconds to fill out a grid {display_x_mesh.shape[0]}x{display_x_mesh.shape[1]} or {display_x_mesh.shape[0] * display_x_mesh.shape[1]} points"
    )

    local_map = np.asarray(flat_local).reshape(display_shape)
    smooth_map = np.asarray(flat_smooth).reshape(display_shape)

    if show:
        from matplotlib import pyplot as plt
        from matplotlib import colors as mplcolors

        fig = plt.figure()
        axes = fig.add_subplot(111)

        _ = axes.pcolormesh(
            display_x_mesh,
            display_y_mesh,
            local_map / smooth_map - 1.0,
            cmap="seismic",
            norm=mplcolors.Normalize(vmin=-0.05, vmax=0.05),
        )
        axes.set_xlim(-14.0, -2.0)
        axes.set_ylim(-6.0, 6.0)
        plt.show()
        plt.close()
