from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from squad.compute.curves import bezier_function
from squad.constants import Direction

from ..states import LegState
from .base import StateController


class LegBezier(StateController[LegState]):
    """
    LegState controller using Bezier curves of foot positions.

    Parameters
    ----------
    state : LegState
        The initial/current state of the leg this controller is
        responsible for.
    control_points : List[Tuple[float, float, float]]
        The control position points (3-D) to construct the Bezier
        motion curve with.
    control_weights : List[float], optional
        The weights/ratios associated with the given `control_points` to
        use in the construction of the (rational) Bezier curve (if any).

    """

    def __init__(
        self,
        state: LegState,
        control_points: List[Tuple[float, float, float]],
        control_weights: Optional[List[float]] = None,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[LegState], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._control_points = np.array(control_points)
        self._bezier_fn = bezier_function(control_points, control_weights)
        return super().__init__(
            state,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def _update(self, **kwargs: Any) -> None:
        self._state.update_position(
            *self._bezier_fn(self._progress),
            **kwargs,
        )


class LegBezierDeltas(LegBezier):
    """
    LegState controller using Bezier curves of foot position deltas.

    Parameters
    ----------
    state : LegState
        The initial/current state of the leg this controller is
        responsible for.
    control_points : List[Tuple[float, float, float]]
        The control position deltas from the initial point (3-D) to
        construct the Bezier motion curve with.
    control_weights : List[float], optional
        The weights/ratios associated with the given `control_points` to
        use in the construction of the (rational) Bezier curve (if any).

    """

    def __init__(
        self,
        state: LegState,
        control_deltas: List[Tuple[float, float, float]],
        control_weights: Optional[List[float]] = None,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[LegState], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._control_deltas = control_deltas
        self._control_weights = control_weights
        control_points = self._compute_control_points(state)
        return super().__init__(
            state,
            control_points,
            control_weights=control_weights,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def _compute_control_points(
        self,
        state: LegState,
    ) -> List[Tuple[float, float, float]]:
        """Computes a new set of control points for the given state."""
        control_points = [(state.x, state.y, state.z)]
        for d in self._control_deltas:
            nxt_pt = tuple(control_points[-1][j] + d[j] for j in range(3))
            control_points.append(nxt_pt)
        return control_points

    def reset(
        self,
        state: LegState,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
    ) -> None:
        control_points = self._compute_control_points(state)
        self._bezier_fn = bezier_function(
            control_points,
            weights=self._control_weights,
        )
        return super().reset(state, direction=direction, progress=progress)


class LegBezierThetas(StateController[LegState]):
    """
    LegState controller using Bezier curves of servo angles.

    Parameters
    ----------
    state : LegState
        The initial/current state of the leg this controller is
        responsible for.
    control_points : List[Tuple[float, float, float]]
        The control angle points (3-D) of the (Hip, Femur, Leg) to
        construct the Bezier motion curve with.
    control_weights : List[float], optional
        The weights/ratios associated with the given `control_points` to
        use in the construction of the (rational) Bezier curve (if any).

    """

    def __init__(
        self,
        state: LegState,
        control_points: List[Tuple[float, float, float]],
        control_weights: Optional[List[float]] = None,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[LegState], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._control_points = np.array(control_points)
        self._bezier_fn = bezier_function(control_points, control_weights)
        return super().__init__(
            state,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def _update(self, **kwargs: Any) -> None:
        self._state.update_orientation(
            *self._bezier_fn(self._progress),
            **kwargs,
        )
