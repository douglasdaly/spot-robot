from datetime import datetime
from typing import TYPE_CHECKING, Generic, Optional

from squad.constants import Direction, TimeType


if TYPE_CHECKING:
    from .controllers.base import StateController, T


class MotionManager(Generic[T]):
    """
    Manages a motion controller to update progress based on timestamps.
    """

    def __init__(
        self,
        name: str,
        controller: StateController,
        frequency: float,
        *,
        time_scale: TimeType = TimeType.SECOND,
    ) -> None:
        self._name = name
        self._controller = controller

        if time_scale == TimeType.MICROSECOND:
            self._freq = frequency * 10e6
        elif time_scale == TimeType.MILLISECOND:
            self._freq = frequency * 10e3
        elif time_scale == TimeType.SECOND:
            self._freq = frequency
        elif time_scale == TimeType.MINUTE:
            self._freq = frequency / 60.0
        elif time_scale == TimeType.HOUR:
            self._freq = frequency / (60.0 * 60.0)
        else:
            raise ValueError(f"Invalid time scale given: {time_scale}")

        self._last_update = controller.state.timestamp

    @property
    def name(self) -> str:
        """str: The name of this motion manager."""
        return self._name

    @property
    def frequency(self) -> float:
        """float: The frequency (per second) of this motion managed."""
        return self._freq

    @property
    def last_update(self) -> datetime:
        """datetime: The timestamp this manager was last updated."""
        return self._last_update

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._name))

    def update(
        self,
        ts: Optional[datetime] = None,
        direction: Optional[Direction] = None,
    ) -> T:
        """Updates the motion controller based on the given timestamp.

        Parameters
        ----------
        ts : datetime, optional
            The timestamp to update the managed controller to (if not
            provided the current datetime is used).
        direction : Direction, optional
            The direction to update the managed controller in (if not
            provided the direction determined by the controller is
            used.)

        Returns
        -------
        BaseState
            The updated/next state as determined by the managed
            controller based upon the given `ts` and `direction` (if
            any).

        """
        if ts is None:
            c_ts = datetime.now()
        else:
            c_ts = ts
        d_p = (c_ts - self._last_update).total_seconds() * self._freq
        next_state = self._controller.increment(d_p, direction)
        self._last_update = c_ts
        return next_state
