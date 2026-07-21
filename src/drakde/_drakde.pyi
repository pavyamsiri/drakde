from typing import Optional, TypeVar
import numpy as np
import optype.numpy as onp

# type vars for symbolic shape names
_N = TypeVar("_N", bound=int)
_M = TypeVar("_M", bound=int)

class BivariateKDE:
    """Typed stub for the compiled BivariateKDE extension.

    Uses optype.numpy (aliased as `onp`) to express array dimensionality and dtype.
    """

    def __new__(
        cls,
        x: onp.Array1D[np.float64],
        y: onp.Array1D[np.float64],
        weights: Optional[onp.Array1D[np.float64]] = None,
    ) -> "BivariateKDE": ...

    def __repr__(self) -> str: ...

    # number of points
    def len(self) -> int: ...

    # accessors (may return Python lists in the runtime implementation; typed here as 1-D arrays)
    def get_x(self) -> onp.Array1D[np.float64]: ...

    def get_y(self) -> onp.Array1D[np.float64]: ...

    def get_weights(self) -> onp.Array1D[np.float64]: ...

    def estimate_scalar(self, x: float, y: float, scale_length: float) -> float: ...

    def estimate_vector(
        self,
        xs: onp.Array1D[np.float64],
        ys: onp.Array1D[np.float64],
        scale_length: float,
    ) -> onp.Array1D[np.float64]: ...


def hello() -> str: ...

