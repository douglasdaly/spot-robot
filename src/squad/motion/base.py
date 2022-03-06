from abc import ABCMeta, abstractmethod
from datetime import datetime
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from squad.constants import Direction


T = TypeVar("T", bound="BaseState")


class BaseState(metaclass=ABCMeta):
    """
    Base class for state storage objects.
    """

    __slots__ = ("_timestamp",)

    def __init__(self, *, timestamp: Optional[datetime] = None) -> None:
        self._timestamp = timestamp or datetime.now()

    @property
    def timestamp(self) -> datetime:
        """datetime: The timestamp associated with this state object."""
        return self._timestamp

    def __str__(self) -> str:
        s_args, s_kws = self.__str_args__()
        if s_args or s_kws:
            arg_str = (
                "["
                + ", ".join(
                    tuple(f"{x}" for x in s_args)
                    + tuple(f"{k}={v}" for k, v in s_kws.items())
                )
                + "]"
            )
        else:
            arg_str = ""
        return f"<{self.__class__.__name__}{arg_str} @ {self._timestamp}>"

    def __str_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        return ([], {})

    def __repr__(self) -> str:
        r_args, r_kws = self.__repr_args__()
        r_kws.setdefault("timestamp", self._timestamp)
        arg_str = ", ".join(
            tuple(f"{x!r}" for x in r_args)
            + tuple(f"{k}={v!r}" for k, v in r_kws.items())
        )
        return f"{self.__class__.__name__}({arg_str})"

    def __repr_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        return ([], {})

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BaseState):
            raise ValueError(
                f"Cannot compare {self.__class__.__name__} with:"
                f" {other.__class__.__name__}"
            )
        return self._timestamp < other.timestamp

    def __hash_params__(self) -> Tuple[Any, ...]:
        return (self.__class__.__name__, self._timestamp)

    def __hash__(self) -> int:
        hash_params = self.__hash_params__()
        return hash(hash_params)

    def __getstate__(self) -> Dict[str, Any]:
        state = {}
        slot_iter = chain.from_iterable(
            getattr(x, "__slots__", []) for x in self.__class__.__mro__
        )
        for name in (x for x in slot_iter if hasattr(self, x)):
            k = name.removeprefix("_")
            v = getattr(self, name)
            if isinstance(v, BaseState):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, f"_{k}", v)
        return

    def to_dict(self) -> Dict[str, Any]:
        """Converts this state object to a data dictionary.

        Returns
        -------
        dict
            The dictionary representation of this state object's data.

        """
        return self.__getstate__()

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Creates a new state object from the given data.

        Parameters
        ----------
        data : dict
            The data dictionary to use to create the new state object.

        Returns
        -------
        T
            The new state object instance from the given `data`.

        """
        obj = object.__new__(cls)
        obj.__setstate__(data)
        return obj


class StateStepper(Generic[T]):
    """
    Abstract base class for state stepping and movement.
    """

    if TYPE_CHECKING:
        _initial_state: T
        _current_state: T
        _target_state: T
        _direction: Direction
        _last_update: datetime
        _min_delta_t: float
        _cbs_changed: List[Callable[[T], None]]

    def __init__(
        self,
        initial_state: T,
        initial_direction: Direction = Direction.FORWARD,
        *,
        min_time_delta: float = 0.0,
        changed_callbacks: List[Callable[[T], None]] = [],
    ) -> None:
        self._initial_state = initial_state
        self._current_state = initial_state
        self._target_state = initial_state
        self._direction = initial_direction
        self._last_update = datetime.now()
        self._min_delta_t = min_time_delta
        self._cbs_changed = changed_callbacks

    @property
    def initial_state(self) -> T:
        """BaseState: The starting state of this stepper."""
        return self._initial_state

    @property
    def current_state(self) -> T:
        """BaseState: The current state of this stepper."""
        return self._current_state

    @property
    def target_state(self) -> T:
        """BaseState: The target state of this stepper."""
        return self._target_state

    @property
    def direction(self) -> Direction:
        """Direction: The current direction of this stepper."""
        return self._direction

    @property
    def last_updated(self) -> datetime:
        """datetime: The timestamp of the last update to this stepper."""
        return self._last_update

    @property
    def min_time_delta(self) -> float:
        """float: The minimum time delta for updates (in ms)."""
        return self._min_delta_t

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[T]:
        return self.next(*args, **kwargs)

    def next(
        self,
        delta_t: Optional[float] = None,
        direction: Optional[Direction] = None,
        *,
        target: Optional[T] = None,
    ) -> Optional[T]:
        """Computes the next interim state of this stepper.

        Parameters
        ----------
        delta_t : float, optional
            The time-step interval to compute the interim state for (in
            milliseconds).  If not provided it will use the time since
            the last update.
        direction : Direction, optional
            The direction for the time-step interval to compute (if not
            given the current direction is used).
        target : BaseState, optional
            The new target state to set, if not provided (default) the
            last target state is used.

        Returns
        -------
        BaseState or None
            The next interim step between the current state and the
            target state for the given time-step interval (if any).

        """
        if target or direction is not None:
            self._initial_state = self._current_state
            if target is not None:
                self._target_state = target
            if direction is not None:
                self._direction = direction

        if delta_t is None:
            t_dt = datetime.now() - self._last_update
            d_t = (t_dt.seconds * 1000.0) + (t_dt.microseconds / 1000.0)
        else:
            d_t = delta_t

        if d_t >= self._min_delta_t:
            ret = self._step(d_t)
            if ret is not None:
                self._current_state = ret
        else:
            ret = None

        self._update(ret is not None)

        return ret

    @abstractmethod
    def _step(self, delta_t: float) -> Optional[T]:
        """Computes the next state between the current and target."""
        raise NotImplementedError

    def _update(self, state_changed: bool) -> None:
        """Updates this planner based on states and calls any callbacks
        needed.

        Parameters
        ----------
        state_changed : bool
            Whether or not the state has changed since the last update.

        """
        if state_changed:
            self._last_update = datetime.now()
            for cb in self._cbs_changed:
                cb(self._current_state)
        return
