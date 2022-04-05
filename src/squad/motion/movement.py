from datetime import datetime
import time
from typing import TYPE_CHECKING, Optional

from squad.constants import Direction, TimeType
from squad.utils import convert_time


if TYPE_CHECKING:
    from .base import BaseState
    from .controllers.base import StateController


class Movement:
    """
    Manages a controller for a single motion.
    """

    __slots__ = (
        "_name",
        "_controller",
        "_duration",
        "_loop",
        "_repeat",
        "_last_update",
    )

    def __init__(
        self,
        name: str,
        controller: "StateController",
        duration: float,
        time_scale: TimeType = TimeType.SECOND,
        *,
        loop: bool = False,
        repeat: bool = False,
    ) -> None:
        if duration <= 0.0:
            raise ValueError("Duration must be positive")
        if loop and repeat:
            raise ValueError("Cannot specify both loop and repeat, only one")

        self._name = name
        self._controller = controller
        self._duration = convert_time(duration, time_scale, TimeType.SECOND)
        self._loop = loop
        self._repeat = repeat
        self._last_update: float = controller.state.timestamp.timestamp()

    @property
    def name(self) -> str:
        """str: The name of this motion manager."""
        return self._name

    @property
    def duration(self) -> float:
        """float: The duration (in seconds) of the motion."""
        return self._duration

    @property
    def progress(self) -> float:
        """float: The progress of the controller (in percent)."""
        return self._controller.progress

    @property
    def last_update(self) -> datetime:
        """datetime: The timestamp this manager was last updated."""
        return datetime.fromtimestamp(self._last_update)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._name))

    def start(self, ts: Optional[float] = None) -> None:
        """Starts this movement by setting the initial timestamp.

        Parameters
        ----------
        ts : float, optional
            The timestamp to set as the current last update time used as
            the basis for subsequent update calls (if not provided the
            current timestamp is used).

        """
        if ts is None:
            self._last_update = time.time()
        else:
            self._last_update = ts
        return

    def update(
        self,
        ts: Optional[float] = None,
        direction: Optional[Direction] = None,
    ) -> "BaseState":
        """Updates the motion controller based on the given timestamp.

        Parameters
        ----------
        ts : float, optional
            The timestamp to update the managed controller to (if not
            provided the current timestamp is used).
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
            c_ts = time.time()
        else:
            c_ts = ts

        d_p = (c_ts - self._last_update) / self._duration
        next_state = self._controller.increment(d_p, direction)

        if self._loop and not (0.0 < self._controller.progress < 1.0):
            if self._controller.direction == Direction.FORWARD:
                new_dir = Direction.REVERSE
                new_prog = 1.0
            else:
                new_dir = Direction.FORWARD
                new_prog = 0.0
            self._controller.reset(
                next_state,
                direction=new_dir,
                progress=new_prog,
            )
        elif self._repeat and not self._controller.progress < 1.0:
            self._controller.reset(
                next_state,
                progress=0.0,
            )

        self._last_update = c_ts
        return next_state
