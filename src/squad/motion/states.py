from collections.abc import Sequence
from datetime import datetime
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from squad.constants import Leg
from squad.exceptions import StateError
from squad.kinematics.base import BodyParameters
from squad.kinematics.forward import foot_xyz
from squad.kinematics.inverse import leg_thetas

from .base import BaseState


L = TypeVar("L", bound="LegStates")
T_LegState = TypeVar("T_LegState", bound="LegState")


class LegState(BaseState):
    """
    Leg state data for one leg of the robot.
    """

    __slots__ = (
        "_leg",
        "_x",
        "_y",
        "_z",
        "_hip_theta",
        "_femur_theta",
        "_leg_theta",
    )

    def __init__(
        self,
        leg: Leg,
        x: float,
        y: float,
        z: float,
        hip_theta: float,
        femur_theta: float,
        leg_theta: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._leg = leg
        self._x = x
        self._y = y
        self._z = z
        self._hip_theta = hip_theta
        self._femur_theta = femur_theta
        self._leg_theta = leg_theta

    @property
    def leg(self) -> Leg:
        """Leg: The leg with this state object."""
        return self._leg

    @property
    def x(self) -> float:
        """float: The current X-coordinate of the foot."""
        return self._x

    @property
    def y(self) -> float:
        """float: The current Y-coordinate of the foot."""
        return self._y

    @property
    def z(self) -> float:
        """float: The current Z-coordinate of the foot."""
        return self._z

    @property
    def hip_theta(self) -> float:
        """float: The current hip angle for this leg."""
        return self._hip_theta

    @property
    def femur_theta(self) -> float:
        """float: The current femur angle for this leg."""
        return self._femur_theta

    @property
    def leg_theta(self) -> float:
        """float: The current leg angle for this leg."""
        return self._leg_theta

    def __str_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        s_args, s_kws = super().__str_args__()
        s_kws["x"] = self._x
        s_kws["y"] = self._y
        s_kws["z"] = self._z
        s_kws["hip_theta"] = self._hip_theta
        s_kws["femur_theta"] = self._femur_theta
        s_kws["leg_theta"] = self._leg_theta
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        r_kws["x"] = self._x
        r_kws["y"] = self._y
        r_kws["z"] = self._z
        r_kws["hip_theta"] = self._hip_theta
        r_kws["femur_theta"] = self._femur_theta
        r_kws["leg_theta"] = self._leg_theta
        return r_args, r_kws

    def __hash_params__(self) -> Tuple[Any, ...]:
        return super().__hash_params__() + (self.leg,)

    def update_position(
        self,
        x: float,
        y: float,
        z: float,
        **kwargs: Any,
    ) -> None:
        """Updates the leg's state for the given foot position.

        Parameters
        ----------
        x : float
            The new X-coordinate of the foot to set.
        y : float
            The new Y-coordinate of the foot to set.
        z : float
            The new Z-coordinate of the foot to set.
        **kwargs : optional
            Any additional parameters to pass to the :obj:`leg_thetas`
            function.

        """
        self._x = x
        self._y = y
        self._z = z
        self._hip_theta, self._femur_theta, self._leg_theta = leg_thetas(
            self._leg,
            x,
            y,
            z,
            **kwargs,
        )
        self._timestamp = datetime.now()

    def update_orientation(
        self,
        hip_theta: float,
        femur_theta: float,
        leg_theta: float,
        **kwargs: Any,
    ) -> None:
        """Updates the leg's state for the given servo angles.

        Parameters
        ----------
        hip_theta : float
            The new Hip-angle of the leg to set.
        femur_theta : float
            The new Femur-angle of the leg to set.
        leg_theta : float
            The new Leg-angle of the leg to set.
        **kwargs : optional
            Any additional parameters to pass to the :obj:`foot_xyz`
            function.

        """
        self._hip_theta = hip_theta
        self._femur_theta = femur_theta
        self._leg_theta = leg_theta
        self._x, self._y, self._z = foot_xyz(
            self._leg,
            hip_theta,
            femur_theta,
            leg_theta,
            **kwargs,
        )

    def distance(self, other: "LegState") -> float:
        super().distance(other)
        return (
            ((self._x - other._x) ** 2)
            + ((self._y - other._y) ** 2)
            + ((self._z - other._z) ** 2)
        ) ** 0.5

    @classmethod
    def from_position(
        cls,
        leg: Leg,
        x: float,
        y: float,
        z: float,
        *,
        timestamp: Optional[datetime] = None,
        **kwargs: Any,
    ) -> "LegState":
        """Creates a new LegState from the given foot position.

        Parameters
        ----------
        leg : Leg
            The leg to create the new state object for.
        x : float
            The X-coordinate of the foot to create the new state for.
        y : float
            The Y-coordinate of the foot to create the new state for.
        z : float
            The Z-coordinate of the foot to create the new state for.
        timestamp : datetime, optional
            The timestamp to use for the new state, if any.
        **kwargs : optional
            Additional keyword arguments to pass to the
            :obj:`leg_thetas` function.

        Returns
        -------
        LegState
            The leg state requested, initialized from the given foot
            position.

        """
        t_hip, t_femur, t_leg = leg_thetas(leg, x, y, z, **kwargs)
        return cls(leg, x, y, z, t_hip, t_femur, t_leg, timestamp=timestamp)

    @classmethod
    def from_thetas(
        cls,
        leg: Leg,
        hip_theta: float,
        femur_theta: float,
        leg_theta: float,
        *,
        timestamp: Optional[datetime] = None,
        **kwargs: Any,
    ) -> "LegState":
        """Creates a new LegState from the given servo angles.

        Parameters
        ----------
        leg : Leg
            The leg to create the new state object for.
        hip_theta : float
            The Hip-angle of the leg to create the new state for.
        femur_theta : float
            The Femur-angle of the leg to create the new state for.
        leg_theta : float
            The Leg-angle of the leg to create the new state for.
        timestamp : datetime, optional
            The timestamp to use for the new state, if any.
        **kwargs : optional
            Additional keyword arguments to pass to the :obj:`foot_xyz`
            function.

        Returns
        -------
        LegState
            The leg state requested, initialized from the given leg
            servo angles.

        """
        x, y, z = foot_xyz(leg, hip_theta, femur_theta, leg_theta, **kwargs)
        return cls(
            leg,
            x,
            y,
            z,
            hip_theta,
            femur_theta,
            leg_theta,
            timestamp=timestamp,
        )


class LegStates(Sequence[T_LegState], BaseState):
    """
    Composite state/wrapper for holding :obj:`LegState` objects.
    """

    __slots__ = ("_legs",)

    def __init__(self, *leg_states: T_LegState, **kwargs: Any) -> None:
        self._legs: List[T_LegState] = sorted(leg_states, key=lambda x: x.leg)
        if len(self._legs) != 4:
            raise StateError(
                "Invalid number of leg states given (requires 4), got:"
                f" {len(self._legs)}"
            )
        elif not all(x.leg == (i + 1) for i, x in enumerate(self._legs)):
            raise StateError("Not all legs represented in leg states given")

        if "timestamp" not in kwargs or kwargs["timestamp"] is None:
            kwargs["timestamp"] = max(x.timestamp for x in self._legs)
        super().__init__(**kwargs)

    @property
    def fl(self) -> T_LegState:
        """LegState: State of the front-left leg."""
        return self._legs[Leg.FL - 1]

    @property
    def fr(self) -> T_LegState:
        """LegState: State of the front-right leg."""
        return self._legs[Leg.FR - 1]

    @property
    def bl(self) -> T_LegState:
        """LegState: State of the back-left leg."""
        return self._legs[Leg.BL - 1]

    @property
    def br(self) -> T_LegState:
        """LegState: State of the back-right leg."""
        return self._legs[Leg.BR - 1]

    def __str_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        s_args, s_kws = super().__str_args__()
        _ = (s_args.append(x) for x in self._legs)
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        _ = (r_args.append(x) for x in self._legs)
        return r_args, r_kws

    def __iter__(self) -> Iterator[T_LegState]:
        return iter(self._legs)

    def __len__(self) -> int:
        return len(self._legs)

    def __getitem__(self, leg: Union[int, Leg]) -> T_LegState:
        if isinstance(leg, Leg):
            l_idx = leg.value - 1
        else:
            l_idx = leg
        return self._legs[l_idx]

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["legs"] = [x.__getstate__() for x in self._legs]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        legs: List[Dict[str, Any]] = state.pop("legs")
        state["legs"] = [LegState.from_dict(x) for x in legs]
        return super().__setstate__(state)

    def distance(self, other: "LegStates") -> float:
        super().distance(other)
        ret = 0.0
        for i, v in enumerate(self._legs):
            ret += v.distance(other._legs[i])
        return ret / len(self._legs)


class RobotState(BaseState):
    """
    Overall kinematic state data storage for the robot.
    """

    __slots__ = (
        "_x",
        "_y",
        "_z",
        "_roll",
        "_pitch",
        "_yaw",
        "_leg_states",
        "_body",
    )

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        legs: Sequence[LegState],
        *,
        body_params: Optional[BodyParameters] = None,
        **kwargs: Any,
    ) -> None:
        if body_params is None:
            self._body = BodyParameters(**kwargs)
        else:
            self._body = body_params
        super().__init__(**kwargs)
        self._x = x
        self._y = y
        self._z = z
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw
        self._leg_states = LegStates(*legs)

    @property
    def x(self) -> float:
        """float: The current X-coordinate of the body."""
        return self._x

    @property
    def y(self) -> float:
        """float: The current Y-coordinate of the body."""
        return self._y

    @property
    def z(self) -> float:
        """float: The current Z-coordinate of the body."""
        return self._z

    @property
    def roll(self) -> float:
        """float: The current Roll-angle of the body."""
        return self._roll

    @property
    def pitch(self) -> float:
        """float: The current Pitch-angle of the body."""
        return self._pitch

    @property
    def yaw(self) -> float:
        """float: The current Yaw-angle of the body."""
        return self._yaw

    @property
    def legs(self) -> LegStates[LegState]:
        """LegStates[LegState]: The state of each leg."""
        return self._leg_states

    @property
    def body(self) -> BodyParameters:
        """BodyParameters: The related body parameters for this state."""
        return self._body

    def __str_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        s_args, s_kws = super().__str_args__()
        s_kws["x"] = self._x
        s_kws["y"] = self._y
        s_kws["z"] = self._z
        s_kws["roll"] = self._roll
        s_kws["pitch"] = self._pitch
        s_kws["yaw"] = self._yaw
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        r_args.append(self._x)
        r_args.append(self._y)
        r_args.append(self._z)
        r_args.append(self._roll)
        r_args.append(self._pitch)
        r_args.append(self._yaw)
        r_args.append(tuple(self._leg_states))
        return r_args, r_kws

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["legs"] = self._leg_states.__getstate__()
        state["body"] = self._body.__getstate__()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state["legs"] = LegStates.from_dict(state.pop("legs"))
        state["body"] = BodyParameters.from_dict(state.pop("body"))
        return super().__setstate__(state)

    def distance(self, other: "RobotState") -> float:
        super().distance(other)
        return (
            ((self._x - other._x) ** 2)
            + ((self._y - other._y) ** 2)
            + ((self._z - other._z) ** 2)
        ) ** 0.5
