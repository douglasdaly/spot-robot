from abc import ABCMeta, abstractmethod
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar


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
