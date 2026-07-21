from typing import override
import numpy as np
import optype.numpy as onp

class BivariateKDE:
    """Typed stub for the compiled BivariateKDE extension.

    Uses optype.numpy (aliased as `onp`) to express array dimensionality and dtype.
    """

    def __init__(
        self,
        x: onp.Array1D[np.float32 | np.float64],
        y: onp.Array1D[np.float32 | np.float64],
        weights: onp.Array1D[np.float32 | np.float64] | None = None,
    ) -> None: ...
    @override
    def __repr__(self) -> str: ...

    # number of points
    def len(self) -> int: ...

    # accessors (may return Python lists in the runtime implementation; typed here as 1-D arrays)
    def estimate_scalar(
        self,
        x: float,
        y: float,
        scale_length: float,
        num_sigma: float = 4.0,
    ) -> float: ...
    def estimate_vector(
        self,
        xs: onp.Array1D[np.float32 | np.float64],
        ys: onp.Array1D[np.float32 | np.float64],
        scale_length: float,
        num_sigma: float = 4.0,
    ) -> onp.Array1D[np.float32]: ...
