from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

from squad.constants import Direction


if TYPE_CHECKING:
    from ..base import BaseState


T = TypeVar("T", bound="BaseState")


class StateController(Generic[T]):
    """
    Abstract base class for state control and movement.

    Parameters
    ----------
    state : BaseState
        The state object the controller is responsible for.
    direction : Direction, default=Direction.FORWARD
        The initial direction of motion for the controller.
    progress : float, optional
        The initial progress to use for the controller (if not given it
        will be set using this object's ``_init_progress`` method,
        which, unless overloaded, defaults to ``0.0``).
    callbacks : List[Callable[[T], None]], default=[]
        The callback functions to run after completing any
        updates/changes to this controller's progress (if any).

    """

    if TYPE_CHECKING:
        _name: str
        _state: T
        _direction: Direction
        _progress: float
        _callbacks: List[Callable[[T], None]]

    def __init__(
        self,
        state: T,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[T], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        self._name = name or self.__class__.__name__
        self._state = state
        self._direction = direction
        self._callbacks = callbacks
        self._progress = self._init_progress(progress)

    @property
    def name(self) -> str:
        """str: The name of this particular controller."""
        return self._name

    @property
    def state(self) -> T:
        """BaseState: The current state of this controller."""
        return self._state

    @property
    def direction(self) -> Direction:
        """Direction: The current direction of this controller."""
        return self._direction

    @property
    def progress(self) -> float:
        """float: The current progress of this controller."""
        return self._progress

    def _init_progress(self, progress: Optional[float] = None) -> float:
        """Determines the current progress based on the state."""
        if progress is None:
            return 0.0
        return progress

    def reset(
        self,
        state: T,
        *,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
    ) -> None:
        """Resets this controller with the given state.

        Parameters
        ----------
        state : T
            The state to set this controller to.
        direction : Direction, default=Direction.FORWARD
            The initial direction to set for this controller.
        progress : float, optional
            The initial progress to set for this controller.

        """
        self._state = state
        self._direction = direction
        self._progress = self._init_progress(progress)

    def increment(
        self,
        delta: float,
        direction: Optional[Direction] = None,
        **kwargs: Any,
    ) -> T:
        """Computes the next state for this controller.

        Parameters
        ----------
        delta : float
            The progress increment to compute the interim state for.
        direction : Direction, optional
            The direction for the time-step interval to compute (if not
            given the current direction is used).

        """
        if direction is not None:
            self._direction = direction
        return self.set_progress(
            self._progress + (self._direction * delta),
            **kwargs,
        )

    def set_progress(self, progress: float, **kwargs: Any) -> T:
        """Sets the new progress for this state controller.

        Parameters
        ----------
        progress : float
            The progress value to set (between 0.0 and 1.0).

        """
        self._progress = min(max(progress, 0.0), 1.0)
        self._update(**kwargs)
        for cb in self._callbacks:
            cb(self._state)
        return self._state

    @abstractmethod
    def _update(self, **kwargs: Any) -> None:
        """Updates this controller-managed state."""
        raise NotImplementedError


class CompositeController(StateController[T]):
    """
    State controller consisting of a composite of multiple controllers.
    """

    if TYPE_CHECKING:
        _controllers: List[StateController[T]]
        _transition_points: np.ndarray

    def __init__(
        self,
        state: T,
        *controllers: StateController[T],
        transition_points: Optional[Sequence[float]] = None,
        direction: Direction = Direction.FORWARD,
        progress: Optional[float] = None,
        callbacks: List[Callable[[T], None]] = [],
        name: Optional[str] = None,
    ) -> None:
        if len(controllers) == 0:
            raise ValueError(
                "You must provide at least one controller for this composite"
            )

        self._controllers = list(controllers)
        for i, c in enumerate(self._controllers):
            c.reset(
                state,
                direction=direction,
                progress=progress if i == 0 else 0.0,
            )

        if transition_points is None:
            self._transition_points = np.array(
                [0.0]
                + [
                    float(i) / len(controllers)
                    for i in range(1, len(controllers))
                ]
                + [1.0],
                dtype=float,
            )
        elif len(transition_points) != len(controllers) - 1:
            raise ValueError(
                f"Transition point length doesn't match number of controllers,"
                f" given {len(transition_points)}, required:"
                f" {len(controllers) - 1}"
            )
        else:
            self._transition_points = np.array(
                [0.0, *transition_points, 1.0],
                dtype=float,
            )

        self._curr_controller_idx = 0

        return super().__init__(
            state,
            direction=direction,
            progress=progress,
            callbacks=callbacks,
            name=name,
        )

    def _update(self, **kwargs: Any) -> None:
        c_idx = self._get_index()
        if c_idx != self._curr_controller_idx:
            self._curr_controller_idx = c_idx
            c_controller = self._controllers[c_idx]
            c_controller.reset(self._state, direction=self._direction)
        else:
            c_controller = self._controllers[self._curr_controller_idx]

        c_progress = (self._progress - self._transition_points[c_idx]) / (
            self._transition_points[c_idx + 1] - self._transition_points[c_idx]
        )

        self._state = c_controller.set_progress(c_progress, **kwargs)

    def _get_index(self) -> Union[int, np.intp]:
        """Gets the current controller index to use based on progress."""
        p_diffs = self._progress - self._transition_points[:-1]
        p_diffs[p_diffs < 0.0] = np.nan
        return np.nanargmin(p_diffs)


class CompositeLoopController(CompositeController):
    """
    State controller consisting of a composite of multiple controllers
    which loops back to the first when progress >= 1.0.
    """

    def _get_index(self) -> Union[int, np.intp]:
        return super()._get_index() % (len(self._transition_points) - 1)
