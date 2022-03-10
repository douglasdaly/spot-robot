import math
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np


def bezier_function(
    points: Union[Iterable[float], Iterable[Iterable[float]]],
    weights: Optional[Iterable[float]] = None,
) -> Callable[[float], Tuple[float, ...]]:
    """Creates a new rational Bezier function from the given points.

    Parameters
    ----------
    points : Iterable[float] or Iterable[Tuple[float, ...]]
        The control points to use to construct the new Bezier function.
    weights : Iterable[float], optional
        The corresponding weights to use for the given `points` (if
        any).  If not provided the points will be equally-weighted.

    Returns
    -------
    Callable[[float], Tuple[float, ...]]
        The rational Bezier function constructed from the given control
        `points` and (optional) weights.

    """
    control_points: List[Tuple[float, ...]] = []
    for p in points:
        if isinstance(p, float):
            control_points.append((p,))
        else:
            control_points.append(tuple(p))
    cp_arr = np.array(control_points, dtype=float)

    if weights is None:
        wt_arr = np.ones(cp_arr.shape[0], dtype=float)
    else:
        wt_arr = np.repeat(
            np.array(weights, dtype=float),
            cp_arr.shape[1],
            axis=1,
        )

    lookup_tbl = np.array(
        [math.comb(cp_arr.shape[0], i) for i in range(cp_arr.shape[0])],
        dtype=float,
    )

    def _bezier_fn(t: float) -> Tuple[float, ...]:
        """Rational Bezier curve function.

        Parameters
        ----------
        t : float
            The point on the curve to compute the coordinates for (s.t.
            0 <= t <= 1).

        Returns
        -------
        Tuple[float]
            The coordinates of the point on the curve specified by `t`.

        """
        n = cp_arr.shape[0]
        num = np.zeros(cp_arr.shape[1], dtype=float)
        den = np.zeros_like(num)
        for i in range(n):
            t_bn = lookup_tbl[i] * ((1.0 - t) ** (n - i)) * (t ** i)
            num += t_bn * (cp_arr[i] * wt_arr[i])
            den += t_bn * wt_arr[i]
        return tuple(num / den)

    return _bezier_fn
