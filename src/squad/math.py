import math
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from .constants import LIMIT_LT_1


def _bezier_domain(t: float) -> float:
    """Constrains the value to the valid Bezier domain."""
    return max(min(t, LIMIT_LT_1), 0.0)


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
        Tuple[float, ...]
            The coordinates of the point on the curve specified by `t`.

        """
        t = _bezier_domain(t)
        n = cp_arr.shape[0]
        num = np.zeros(cp_arr.shape[1], dtype=float)
        den = np.zeros_like(num)
        for i in range(n):
            t_bn = lookup_tbl[i] * ((1.0 - t) ** (n - i)) * (t ** i)
            num += t_bn * (cp_arr[i] * wt_arr[i])
            den += t_bn * wt_arr[i]
        return tuple(num / den)

    return _bezier_fn


def linear_interpolation_function(
    initial_point: Union[float, Tuple[float, ...]],
    final_point: Union[float, Tuple[float, ...]],
) -> Callable[[float], Tuple[float, ...]]:
    """Creates a linear interpolation function between two points.

    Parameters
    ----------
    initial_point : float or Tuple[float, ...]
        The initial point to start the interpolation from.
    final_point : float or Tuple[float, ...]
        The final/target point to end the interpolation at.

    Returns
    -------
    Callable[[float], Tuple[float, ...]]
        The interpolation function between the two points.

    """
    if isinstance(initial_point, float):
        pt_s = (initial_point,)
    else:
        pt_s = initial_point

    if isinstance(final_point, float):
        pt_f = (final_point,)
    else:
        pt_f = final_point

    d_pts = tuple(f - s for s, f in zip(pt_s, pt_f))
    pt_iter = tuple(zip(pt_s, d_pts))

    def _linear_interpolation_fn(t: float) -> Tuple[float, ...]:
        """Linear interpolation function from initial to final points.

        Parameters
        ----------
        t : float
            The point on the curve to compute the coordinates for (s.t.
            0.0 <= t <= 1.0).

        Returns
        -------
        Tuple[float, ...]
            The coordinate(s) of the point specified by `t`.

        """
        return tuple(s + (t * d) for s, d in pt_iter)

    return _linear_interpolation_fn
