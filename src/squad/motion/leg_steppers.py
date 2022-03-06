from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import numpy as np

from squad.constants import Direction

from .base import StateStepper
from .states import LegServoState


class LegDiscreteStepper(StateStepper[LegServoState]):
    """
    Leg servo discrete phased angle state stepper.
    """

    if TYPE_CHECKING:
        _cycle_len: float
        _p_hip: np.ndarray
        _p_femur: np.ndarray
        _p_leg: np.ndarray
        _l_p_hip: int
        _l_p_femur: int
        _l_p_leg: int
        _d_hip: int
        _d_femur: int
        _d_leg: int

    def __init__(
        self,
        cycle_length: float,
        hip_points: Sequence[float],
        femur_points: Sequence[float],
        leg_points: Sequence[float],
        *args: Any,
        hip_direction: int = 1,
        femur_direction: int = 1,
        leg_direction: int = 1,
        **kwargs: Any,
    ) -> None:
        self._cycle_len = cycle_length
        self._p_hip = np.array(hip_points)
        self._p_femur = np.array(femur_points)
        self._p_leg = np.array(leg_points)

        self._l_p_hip = len(self._p_hip)
        self._l_p_femur = len(self._p_femur)
        self._l_p_leg = len(self._p_leg)

        self._d_hip = hip_direction
        self._d_femur = femur_direction
        self._d_leg = leg_direction

        self._debug = {}

        return super().__init__(*args, **kwargs)

    def _get_initial_state(self) -> LegServoState:
        return LegServoState(
            self._current_state.leg,
            self._p_hip[0],
            self._p_femur[0],
            self._p_leg[0],
            timestamp=datetime.now(),
        )

    def _get_target_state(self) -> LegServoState:
        tgt_ts = self._initial_state.timestamp + timedelta(
            seconds=(self._cycle_len / 1000.0)
        )
        return LegServoState(
            self._current_state.leg,
            self._p_hip[-1],
            self._p_femur[-1],
            self._p_leg[-1],
            timestamp=tgt_ts,
        )

    def _update_target(self) -> None:
        if self._direction == Direction.FORWARD:
            self._initial_state = self._get_target_state()
            self._target_state = self._get_initial_state()
            self._direction = Direction.REVERSE
        else:
            self._initial_state = self._get_initial_state()
            self._target_state = self._get_target_state()
            self._direction = Direction.FORWARD
        self._current_state.update()

    def _nearest_indices(self) -> Tuple[int, int, int]:
        i_hip = np.argmin(abs(self._p_hip - self._current_state.hip_theta))
        i_fem = np.argmin(abs(self._p_femur - self._current_state.femur_theta))
        i_leg = np.argmin(abs(self._p_leg - self._current_state.leg_theta))
        return i_hip, i_fem, i_leg  # type: ignore

    def _step(self, delta_t: float) -> Optional[LegServoState]:
        dt_pct = delta_t / self._cycle_len

        d_i_hip = int(round(dt_pct * self._l_p_hip)) * self._d_hip
        d_i_fem = int(round(dt_pct * self._l_p_femur)) * self._d_femur
        d_i_leg = int(round(dt_pct * self._l_p_leg)) * self._d_leg

        if d_i_hip == d_i_fem == d_i_leg == 0:
            return None

        c_i_hip, c_i_fem, c_i_leg = self._nearest_indices()
        n_i_hip = c_i_hip + d_i_hip
        n_i_fem = c_i_fem + d_i_fem
        n_i_leg = c_i_leg + d_i_leg

        if n_i_hip < 0 or n_i_hip >= self._l_p_hip:
            self._d_hip *= -1
        if n_i_fem < 0 or n_i_fem >= self._l_p_femur:
            self._d_femur *= -1
        if n_i_leg < 0 or n_i_leg >= self._l_p_leg:
            self._d_leg *= -1

        n_i_hip = min(max(n_i_hip, 0), self._l_p_hip - 1)
        n_i_fem = min(max(n_i_fem, 0), self._l_p_femur - 1)
        n_i_leg = min(max(n_i_leg, 0), self._l_p_leg - 1)

        t_hip = self._p_hip[n_i_hip]
        t_fem = self._p_femur[n_i_fem]
        t_leg = self._p_leg[n_i_leg]

        self._debug[datetime.now()] = {
            "delta_t": delta_t,
            "dt_pct": dt_pct,
            "d_i_hip": d_i_hip,
            "d_i_fem": d_i_fem,
            "d_i_leg": d_i_leg,
            "c_i_hip": c_i_hip,
            "c_i_fem": c_i_fem,
            "c_i_leg": c_i_leg,
            "n_i_hip": n_i_hip,
            "n_i_fem": n_i_fem,
            "n_i_leg": n_i_leg,
            "t_hip": t_hip,
            "t_fem": t_fem,
            "t_leg": t_leg,
        }

        return LegServoState(
            self._current_state.leg,
            t_hip,
            t_fem,
            t_leg,
        )
