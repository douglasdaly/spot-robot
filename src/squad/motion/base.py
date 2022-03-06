from abc import ABCMeta, abstractmethod
from datetime import datetime
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from squad.constants import Direction


T = TypeVar("T", bound="BaseState")


DEFAULT_DISTANCE_THRESHOLD = 1.0


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

    def update(self, **values: Any) -> None:
        """Updates values of this state object in-place.

        Parameters
        ----------
        **values : Any
            The values to update on this state object.

        """
        self.__setstate__(values)
        self._timestamp = datetime.now()

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

    @abstractmethod
    def distance(self: T, other: T) -> float:
        """Computes a distance metric between this state and another.

        Parameters
        ----------
        other : T
            The other state (of the same type) to assess the distance
            metric against.

        Returns
        -------
        float
            A numeric (and class-dependant) measure of the "distance"
            between this state and the `other` state given.

        Raises
        ------
        ValueError
            If the given `other` is not of the same type as this state
            object and cannot be compared.

        """
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Cannot measure distance between {self.__class__.__name__}"
                f" and given: {other.__class__.__name__}"
            )


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
        _dist_threshold: float
        _min_delta_t: float
        _cbs_changed: List[Callable[[T], None]]

    def __init__(
        self,
        state: T,
        initial_state: Optional[T] = None,
        target_state: Optional[T] = None,
        direction: Direction = Direction.FORWARD,
        *,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        min_time_delta: float = 0.0,
        changed_callbacks: List[Callable[[T], None]] = [],
    ) -> None:
        self._current_state = state
        self._direction = direction
        self._last_update = datetime.now()
        self._min_delta_t = min_time_delta
        self._dist_threshold = distance_threshold
        self._cbs_changed = changed_callbacks
        self._initial_state = initial_state or self._get_initial_state()
        self._target_state = target_state or self._get_target_state()

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
    def distance_threshold(self) -> float:
        """float: The minimum distance metric considered on target."""
        return self._dist_threshold

    @property
    def min_time_delta(self) -> float:
        """float: The minimum time delta for updates (in ms)."""
        return self._min_delta_t

    def __call__(self, delta_t: Optional[float] = None, **kwargs: Any) -> None:
        return self.next(delta_t, **kwargs)

    def next(
        self,
        delta_t: Optional[float] = None,
        direction: Optional[Direction] = None,
        *,
        initial: Optional[T] = None,
        target: Optional[T] = None,
    ) -> None:
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
        initial : BaseState, optional
            The initial state to set, if not provided (default) the last
            initial state is used.
        target : BaseState, optional
            The new target state to set, if not provided (default) the
            last target state is used.

        """
        if initial or target or direction is not None:
            if initial is not None:
                self._initial_state = initial
            if target is not None:
                self._target_state = target
            if direction is not None:
                self._direction = direction

        next_state = None
        if self._direction != Direction.NONE:
            if self._off_target():
                if delta_t is None:
                    t_dt = datetime.now() - self._last_update
                    d_t = (t_dt.seconds * 1000.0) + (
                        t_dt.microseconds / 1000.0
                    )
                    t_mdt = self._min_delta_t
                else:
                    d_t = delta_t
                    t_mdt = 0.0

                if d_t >= t_mdt:
                    next_state = self._step(d_t)
                    if next_state is not None:
                        self._current_state = next_state
            else:
                self._update_target()
        self._update(next_state)

    @abstractmethod
    def _step(self, delta_t: float) -> Optional[T]:
        """Computes the next state between the current and target."""
        raise NotImplementedError

    def _off_target(self) -> bool:
        """Determines if the state is not at the target."""
        curr_dist = self._current_state.distance(self._target_state)
        return curr_dist > self._dist_threshold

    def _update(self, next_state: Optional[T]) -> None:
        """Updates this planner based on states and calls any callbacks
        needed.

        Parameters
        ----------
        state_changed : bool
            Whether or not the state has changed since the last update.

        """
        if next_state is not None:
            self._last_update = next_state.timestamp
            self._current_state = next_state
            for cb in self._cbs_changed:
                cb(self._current_state)
        return

    def _update_target(self) -> None:
        """Updates the target state, if needed."""
        self._direction = Direction.NONE

    def _get_initial_state(self) -> T:
        """Gets the first initial state to use."""
        raise NotImplementedError

    def _get_target_state(self) -> T:
        """Gets the first target state to use."""
        raise NotImplementedError


class MotionController(metaclass=ABCMeta):
    """
    Controller base class for coordinated state movement.
    """

    if TYPE_CHECKING:
        _steppers: Dict[Hashable, StateStepper]
        _stepper_params: Dict[Hashable, Dict[str, Any]]

    def __init__(self, states: Dict[Hashable, BaseState]) -> None:
        self._stepper_params = self.init_stepper_params(states)
        self._steppers = self.create_steppers(states, self._stepper_params)

    def __call__(self, delta_t: Optional[float] = None, **kwargs: Any) -> None:
        return self.next(delta_t, **kwargs)

    def next(self, delta_t: Optional[float] = None, **kwargs: Any) -> None:
        """Updates this motion controller for the next time-step."""
        self._update_stepper_params_pre(**kwargs)
        for k, v in self._steppers.items():
            v_kws = self._stepper_params.get(k, {})
            v.next(delta_t, **v_kws)
        self._update_stepper_params_post(**kwargs)

    @abstractmethod
    def create_steppers(
        self,
        states: Dict[Hashable, T],
        init_params: Dict[Hashable, Dict[str, Any]],
    ) -> Dict[Hashable, StateStepper[T]]:
        """Creates this objects state steppers to use.

        Parameters
        ----------
        states : Dict[Hashable, BaseState]
            The state object(s) and identifiers to create steppers for.
        init_params : Dict[Hashable, Dict[str, Any]]
            The initial parameters to use when creating the stepper
            objects (if any).

        Returns
        -------
        Dict[Hashable, StateStepper]
            The state stepper objects and identifiers to use.

        """
        return {}

    @abstractmethod
    def init_stepper_params(
        self,
        states: Dict[Hashable, T],
    ) -> Dict[Hashable, Dict[str, Any]]:
        """Gets the initial stepper parameters to use.

        Parameters
        ----------
        states : Dict[Hashable, BaseState]
            The state object(s) and identifiers to create the initial
            stepper parameters for.

        Returns
        -------
        Dict[Hashable, Dict[str, Any]]
            The initial parameters for the state stepper objects to use.

        """
        ret = {}
        for k in states:
            ret[k] = {}
        return ret

    def _update_stepper_params_pre(self, **kwargs: Any) -> None:
        """Updates the stepper parameters before each step."""
        return

    def _update_stepper_params_post(self, **kwargs: Any) -> None:
        """Updates the stepper parameters after each step."""
        for k in self._stepper_params:
            self._stepper_params[k].clear()
        return
