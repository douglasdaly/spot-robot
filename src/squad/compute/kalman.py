import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np


class KalmanFilter:
    """
    Kalman filter for an observed value.
    """

    if TYPE_CHECKING:
        _x: np.ndarray
        _p: np.ndarray
        _f: np.ndarray
        _b: np.ndarray
        _h: np.ndarray
        _r: np.ndarray
        _q: np.ndarray
        _ts: float

    __slots__ = (
        "_x",
        "_p",
        "_f",
        "_b",
        "_h",
        "_r",
        "_q",
        "_ts",
    )

    def __init__(
        self,
        initial_state: np.ndarray,
        state_transition: Optional[np.ndarray] = None,
        initial_covar: Optional[np.ndarray] = None,
        control_matrix: Optional[np.ndarray] = None,
        measurement_matrix: Optional[np.ndarray] = None,
        measurement_covar: Optional[np.ndarray] = None,
        noise_covar: Optional[np.ndarray] = None,
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        self._x = initial_state.copy()
        if len(self._x.shape) == 1:
            self._x = self._x.reshape(-1, 1)

        if initial_covar is None:
            self._p = np.identity(self._x.shape[1], dtype=float)
        else:
            self._p = initial_covar.copy()

        if state_transition is None:
            self._f = np.identity(self._x.shape[1], dtype=float)
        elif isinstance(state_transition, np.ndarray):
            self._f = state_transition.copy()

        if control_matrix is None:
            self._b = np.zeros_like(self._f)
        else:
            self._b = control_matrix.copy()

        if measurement_matrix is None:
            self._h = np.identity(self._x.shape[1], dtype=float)
        else:
            self._h = measurement_matrix.copy()

        if measurement_covar is None:
            self._r = np.identity(self._h.shape[0], dtype=float)
        else:
            self._r = measurement_covar.copy()

        if noise_covar is None:
            self._q = np.identity(self._f.shape[0], dtype=float)
        else:
            self._q = noise_covar.copy()

        self._ts = timestamp or time.time()

    def _kf_predict(
        self,
        x_0: np.ndarray,
        f_0: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman Filter prediction step from given state."""
        x_1 = f_0.dot(x_0)
        if u is not None:
            x_1 += self._b.dot(u)
        p_1 = np.dot(f_0, np.dot(self._p, f_0.T)) + (
            self._q * np.random.normal(0.0, 1.0)
        )
        return x_1, p_1

    def _kf_update(
        self,
        x_obs: np.ndarray,
        x_pred: np.ndarray,
        p_1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman Filter update step from given observation."""
        z_obs = self._h.dot(x_obs) + (self._r * np.random.normal(0.0, 1.0))
        k_1 = p_1.dot(self._h.T).dot(
            np.linalg.inv(
                self._h.dot(p_1.dot(self._h.T))
                + (self._r * np.random.normal(0.0, 1.0))
            ),
        )
        x_t = x_pred + k_1.dot(z_obs - self._h.dot(x_pred))
        p_t = p_1 - k_1.dot(self._h.dot(p_1))
        return x_t, p_t
