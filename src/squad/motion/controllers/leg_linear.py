from typing import Any, Callable, List, Optional, Tuple

from squad.constants import Direction
from squad.math import linear_interpolation_function

from ..states import LegState
from .base import StateController


class LegLinear(StateController[LegState]):
    """
    LegState controller using linear interpolation between positions to
    reach the target.

    Parameters
    ----------
    state : LegState
        The initial/current state of the leg this controller is
        responsible for.
    x : float
        The desired/target X-coordinate of the foot on the leg to reach.
    y : float
        The desired/target Y-coordinate of the foot on the leg to reach.
    z : float
        The desired/target Z-coordinate of the foot on the leg to reach.

    """

    def __init__(
        self,
        state: LegState,
        x: float,
        y: float,
        z: float,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[LegState], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._tgt_x = x
        self._tgt_y = y
        self._tgt_z = z
        self._interp_fn = linear_interpolation_function(
            (state.x, state.y, state.z),
            (x, y, z),
        )
        return super().__init__(
            state,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def reset(
        self,
        state: LegState,
        *,
        direction: Direction = ...,
        progress: Optional[float] = ...,
    ) -> None:
        self._interp_fn = linear_interpolation_function(
            (state.x, state.y, state.z),
            (self._tgt_x, self._tgt_y, self._tgt_z),
        )
        super().reset(state, direction=direction, progress=progress)

    def _update(self, **kwargs: Any) -> None:
        self._state.update_position(
            *self._interp_fn(self._progress),
            **kwargs,
        )


class LegLinearDeltas(StateController[LegState]):
    """
    LegState controller using linear interpolation between positions,
    specified by deltas, to reach the new target.

    Parameters
    ----------
    state : LegState
        The initial/current state of the leg this controller is
        responsible for.
    delta_x : float
        The desired/target X-coordinate delta of the foot on the leg to
        reach.
    delta_y : float
        The desired/target Y-coordinate delta of the foot on the leg to
        reach.
    delta_z : float
        The desired/target Z-coordinate delta of the foot on the leg to
        reach.

    """

    def __init__(
        self,
        state: LegState,
        delta_x: float,
        delta_y: float,
        delta_z: float,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[LegState], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._d_x = delta_x
        self._d_y = delta_y
        self._d_z = delta_z

        x, y, z = self._compute_targets(state)
        self._interp_fn = linear_interpolation_function(
            (state.x, state.y, state.z),
            (x, y, z),
        )

        return super().__init__(
            state,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def _compute_targets(self, state: LegState) -> Tuple[float, float, float]:
        """Computes the new target positions from the given state."""
        return (state.x + self._d_x, state.y + self._d_y, state.z + self._d_z)

    def reset(
        self,
        state: LegState,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
    ) -> None:
        x, y, z = self._compute_targets(state)
        self._interp_fn = linear_interpolation_function(
            (state.x, state.y, state.z),
            (x, y, z),
        )
        return super().reset(state, direction=direction, progress=progress)

    def _update(self, **kwargs: Any) -> None:
        self._state.update_position(
            *self._interp_fn(self._progress),
            **kwargs,
        )


class LegLinearThetas(StateController[LegState]):
    """
    LegState controller using linear interpolation between angles to
    reach the target.

    Parameters
    ----------
    state : LegState
        The initial/current state of the leg this controller is
        responsible for.
    hip_theta : float
        The desired/target Hip-angle of the leg to reach.
    femur_theta : float
        The desired/target Femur-angle of the leg to reach.
    leg_theta : float
        The desired/target Leg-angle of the leg to reach.

    """

    def __init__(
        self,
        state: LegState,
        hip_theta: float,
        femur_theta: float,
        leg_theta: float,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[LegState], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._tgt_hip = hip_theta
        self._tgt_femur = femur_theta
        self._tgt_leg = leg_theta
        self._interp_fn = linear_interpolation_function(
            (state.hip_theta, state.femur_theta, state.leg_theta),
            (hip_theta, femur_theta, leg_theta),
        )
        return super().__init__(
            state,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def reset(
        self,
        state: LegState,
        *,
        direction: Direction = ...,
        progress: Optional[float] = ...,
    ) -> None:
        self._interp_fn = linear_interpolation_function(
            (state.hip_theta, state.femur_theta, state.leg_theta),
            (self._tgt_hip, self._tgt_femur, self._tgt_leg),
        )
        super().reset(state, direction=direction, progress=progress)

    def _update(self, **kwargs: Any) -> None:
        self._state.update_orientation(
            *self._interp_fn(self._progress),
            **kwargs,
        )
