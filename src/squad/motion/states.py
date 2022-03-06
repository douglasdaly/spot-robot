from abc import ABCMeta
from collections.abc import Sequence
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

from squad.constants import AngleType, Leg
from squad.exceptions import StateError
from squad.kinematics.base import BodyParameters
from squad.kinematics.forward import foot_xyz
from squad.kinematics.inverse import leg_thetas

from .base import BaseState


L = TypeVar("L", bound="LegStates")
T_LegState = TypeVar("T_LegState", bound="LegState")


class LegState(BaseState, metaclass=ABCMeta):
    """
    Base class for Leg State data storage.
    """

    __slots__ = ("_leg",)

    def __init__(self, leg: Leg, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leg = leg

    @property
    def leg(self) -> Leg:
        """Leg: The leg with this state object."""
        return self._leg

    def __str_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        s_args, s_kws = super().__str_args__()
        s_args.append(self._leg.name)
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        r_args.append(self._leg)
        return r_args, r_kws

    def __hash_params__(self) -> Tuple[Any, ...]:
        return super().__hash_params__() + (self.leg,)


class LegServoState(LegState):
    """
    Leg servo state data for one leg of the robot.
    """

    __slots__ = (
        "_hip_theta",
        "_femur_theta",
        "_leg_theta",
    )

    def __init__(
        self,
        leg: Leg,
        hip_theta: float,
        femur_theta: float,
        leg_theta: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(leg, **kwargs)
        self._hip_theta = hip_theta
        self._femur_theta = femur_theta
        self._leg_theta = leg_theta

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
        s_kws["hip_theta"] = self._hip_theta
        s_kws["femur_theta"] = self._femur_theta
        s_kws["leg_theta"] = self._leg_theta
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        r_kws["hip_theta"] = self._hip_theta
        r_kws["femur_theta"] = self._femur_theta
        r_kws["leg_theta"] = self._leg_theta
        return r_args, r_kws


class LegFootState(LegState):
    """
    Leg foot state data for one leg of the robot.
    """

    __slots__ = (
        "_x",
        "_y",
        "_z",
    )

    def __init__(
        self,
        leg: Leg,
        x: float,
        y: float,
        z: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(leg, **kwargs)
        self._x = x
        self._y = y
        self._z = z

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

    def __str_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        s_args, s_kws = super().__str_args__()
        s_kws["x"] = self._x
        s_kws["y"] = self._y
        s_kws["z"] = self._z
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[Iterable[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        r_kws["x"] = self._x
        r_kws["y"] = self._y
        r_kws["z"] = self._z
        return r_args, r_kws


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
        if isinstance(leg, int):
            l_idx = leg
        else:
            l_idx = leg - 1
        return self._legs[l_idx]

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["legs"] = [x.__getstate__() for x in self._legs]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        legs: List[Dict[str, Any]] = state.pop("legs")
        if "x" in legs[0]:
            leg_cls = LegFootState
        else:
            leg_cls = LegServoState
        state["legs"] = [leg_cls.from_dict(x) for x in legs]
        return super().__setstate__(state)


class KinematicState(BaseState):
    """
    Overall kinematic state data storage for the robot.
    """

    __slots__ = (
        "_x",
        "_y",
        "_z",
        "_feet",
        "_servos",
        "_body",
    )

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        foot_states: Optional[Sequence[LegFootState]] = None,
        servo_states: Optional[Sequence[LegServoState]] = None,
        *,
        body_params: Optional[BodyParameters] = None,
        pre_compute: bool = False,
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

        if not foot_states and not servo_states:
            raise StateError("Must provide one of foot_states or servo_states")

        self._feet: Optional[LegStates[LegFootState]] = None
        self._servos: Optional[LegStates[LegServoState]] = None
        if foot_states:
            if isinstance(foot_states, LegStates):
                self._feet = foot_states
            else:
                self._feet = LegStates(*foot_states, **kwargs)
            if not servo_states and pre_compute:
                self._servos = LegStates(
                    *self._servos_from_feet(foot_states, self._body),
                    **kwargs,
                )
        if servo_states:
            if isinstance(servo_states, LegStates):
                self._servos = servo_states
            else:
                self._servos = LegStates(*servo_states, **kwargs)
            self._servos = LegStates(*servo_states, **kwargs)
            if not foot_states and pre_compute:
                self._feet = LegStates(
                    *self._feet_from_servos(servo_states, self._body),
                    **kwargs,
                )
        return

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
    def feet(self) -> LegStates[LegFootState]:
        """LegStates[LegFootState]: The foot state in each leg."""
        if self._feet is None:
            self._feet = LegStates(
                *self._feet_from_servos(self.servos, self._body),
                timestamp=self.servos.timestamp,
            )
        return self._feet

    @property
    def servos(self) -> LegStates[LegServoState]:
        """LegStates[LegServoState]: The servo states in each leg."""
        if self._servos is None:
            self._servos = LegStates(
                *self._servos_from_feet(self.feet, self._body),
                timestamp=self.feet.timestamp,
            )
        return self._servos

    @property
    def body(self) -> BodyParameters:
        """BodyParameters: The related body parameters for this state."""
        return self._body

    def __str_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        s_args, s_kws = super().__str_args__()
        s_kws["x"] = self._x
        s_kws["y"] = self._y
        s_kws["z"] = self._z
        return s_args, s_kws

    def __repr_args__(self) -> Tuple[List[Any], Dict[str, Any]]:
        r_args, r_kws = super().__repr_args__()
        r_args.append(self._x)
        r_args.append(self._y)
        r_args.append(self._z)
        if self._feet:
            r_kws["foot_states"] = tuple(self._feet)
        if self._servos:
            r_kws["servo_states"] = tuple(self._servos)
        return r_args, r_kws

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        if not self._feet and "feet" in state:
            del state["feet"]
        if not self._servos and "servos" in state:
            del state["servos"]
        state["body"] = self._body.__getstate__()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if "feet" in state:
            state["feet"] = LegStates.from_dict(state.pop("feet"))
        else:
            state["feet"] = None

        if "servos" in state:
            state["servos"] = LegStates.from_dict(state.pop("servos"))
        else:
            state["servos"] = None

        state["body"] = BodyParameters.from_dict(state.pop("body"))

        return super().__setstate__(state)

    @classmethod
    def _servos_from_feet(
        cls,
        legs: Sequence[LegFootState],
        body_params: BodyParameters,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> Sequence[LegServoState]:
        """Gets servo states corresponding to the given leg/foot states."""
        ret = []
        for leg in legs:
            t_hip, t_femur, t_leg = leg_thetas(
                leg.leg,
                leg.x,
                leg.y,
                leg.z,
                body_params=body_params,
                angle_type=angle_type,
            )
            t_servo = LegServoState(
                leg.leg,
                t_hip,
                t_femur,
                t_leg,
                timestamp=leg.timestamp,
            )
            ret.append(t_servo)
        return ret

    @classmethod
    def _feet_from_servos(
        cls,
        servos: Sequence[LegServoState],
        body_params: BodyParameters,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> Sequence[LegFootState]:
        """Gets the foot states corresponding to the given servo states."""
        ret = []
        for servo in servos:
            t_x, t_y, t_z = foot_xyz(
                servo.leg,
                servo.hip_theta,
                servo.femur_theta,
                servo.leg_theta,
                body_params=body_params,
                angle_type=angle_type,
            )
            t_foot = LegFootState(
                servo.leg,
                t_x,
                t_y,
                t_z,
                timestamp=servo.timestamp,
            )
            ret.append(t_foot)
        return ret
