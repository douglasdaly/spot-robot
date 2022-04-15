from abc import ABCMeta
from datetime import datetime
import math
from typing import Generic, Optional, TypeVar, Union

from euclid import Vector2, Vector3

from squad.config import config
from squad.constants import (
    HALF_PI,
    LENGTH_CONVERSION_FACTORS,
    TIME_CONVERSION_FACTORS,
    AngleType,
    LengthUnits,
    TimeUnit,
    VelocityUnits,
)
from squad.kinematics.core import coord_rotate_xyz
from squad.utils import convert_velocity


V = TypeVar("V", Vector2, Vector3)


class Heading(Generic[V], metaclass=ABCMeta):
    """
    Represents a direction of motion and speed.
    """

    __slots__ = (
        "_vector",
        "_velocity",
        "_ts",
        "_units",
    )

    def __init__(
        self,
        vector: V,
        speed: float,
        units: Optional[VelocityUnits] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._ts = timestamp or datetime.now()
        self._vector = vector.normalize()
        self._units = units or config.nav_velocity_units
        self._velocity = convert_velocity(
            speed,
            self._units,
            VelocityUnits.METERS_PER_SECOND,
        )

    @property
    def vector(self) -> V:
        """Vector: The (unit) vector of this heading's direction."""
        return self._vector

    @property
    def speed(self) -> float:
        """float: The speed of motion (in the desired units)."""
        return convert_velocity(
            self._velocity,
            VelocityUnits.METERS_PER_SECOND,
            self._units,
        )

    @property
    def velocity(self) -> float:
        """float: The speed of motion (in meters per second)."""
        return self._velocity

    @property
    def units(self) -> VelocityUnits:
        """VelocityUnits: The type/units of the speed for this heading."""
        return self._units

    @property
    def timestamp(self) -> datetime:
        """datetime: The timestamp of this heading."""
        return self._ts

    def delta(
        self,
        dt: Optional[Union[int, float, datetime]] = None,
        time_units: Optional[TimeUnit] = None,
        *,
        distance_units: Optional[LengthUnits] = None,
    ) -> V:
        """Calculates the delta offsets for the given time/interval.

        Parameters
        ----------
        dt : Union[int, float, datetime], optional
            The time delta (in seconds) or time to compute the deltas
            for, if not given the current datetime will be used.
        time_units : TimeUnit, optional
            The units the given `dt` (only if given as an ``int`` or
            ``float`` value) was specified in.  If not provided this is
            assumed to be the default time units the speed was
            originally given in.
        distance_units : LengthUnits, optional
            The units to return the distance deltas in (if not specified
            it will be determined from this instance's speed unit type).

        Returns
        -------
        Vector
            The distance deltas for the given `dt` based on this
            heading's direction and speed (in the `units` given, if
            any).

        """
        if dt is None:
            dt_s = (datetime.now() - self._ts).total_seconds()
        elif isinstance(dt, datetime):
            dt_s = (dt - self._ts).total_seconds()
        else:
            if time_units is None:
                t_factor = TIME_CONVERSION_FACTORS[self._units.time_unit]
            else:
                t_factor = TIME_CONVERSION_FACTORS[time_units]
            dt_s = t_factor * dt

        if distance_units is None:
            d_factor = LENGTH_CONVERSION_FACTORS[self._units.length_units]
        else:
            d_factor = LENGTH_CONVERSION_FACTORS[distance_units]

        return ((self._velocity * dt_s) / d_factor) * self._vector


class Heading2(Heading[Vector2]):
    """
    Represents a direction of motion and speed in 2D.

    Parameters
    ----------
    heading : float
        The angle, relative to the X-axis, of the direction of motion.
    speed : float
        The speed of the motion.
    timestamp : datetime, optional
        The timestamp to mark the initial heading with (if not given the
        current datetime will be used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/format of the given `heading` angle.

    """

    __slots__ = ("_yaw",)

    def __init__(
        self,
        heading: float,
        speed: float,
        units: Optional[VelocityUnits] = None,
        timestamp: Optional[datetime] = None,
        *,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> None:
        if angle_type == AngleType.DEGREES:
            yaw = math.radians(heading)
        else:
            yaw = heading
        self._yaw = yaw

        v_x = math.cos(self._yaw)
        v_y = math.sin(self._yaw)
        vector = Vector2(v_x, v_y)
        return super().__init__(
            vector,
            speed,
            units=units,
            timestamp=timestamp,
        )

    @property
    def heading(self) -> float:
        """float: The heading angle (in degrees)."""
        return math.degrees(self._yaw)

    @property
    def yaw(self) -> float:
        """float: The normalized yaw angle (in radians)."""
        return self._yaw

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(heading={self.heading:.2f},"
            f" speed={self.speed:.2f} {self.units:a})"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.heading:.2f}, {self.speed:.2f},"
            f" {self.units!r})"
        )


class Heading3(Heading[Vector3]):
    """
    Represents a direction and speed in 3D.
    """

    __slots__ = (
        "_yaw",
        "_pitch",
        "_roll",
    )

    def __init__(
        self,
        heading: float,
        elevation: float,
        bank: float,
        speed: float,
        units: Optional[VelocityUnits] = None,
        timestamp: Optional[datetime] = None,
        *,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> None:
        if angle_type == AngleType.DEGREES:
            yaw = math.radians(heading)
            pitch = math.radians(elevation)
            roll = math.radians(bank)
        else:
            yaw = heading
            pitch = elevation
            roll = bank

        self._yaw = yaw
        self._pitch = -pitch
        self._roll = roll + HALF_PI

        v_x, v_y, v_z = coord_rotate_xyz(
            1.0,
            0.0,
            0.0,
            self._roll,
            self._pitch,
            self._yaw,
            angle_type=AngleType.RADIANS,
        )
        vector = Vector3(v_x, v_y, v_z)
        return super().__init__(
            vector,
            speed,
            units=units,
            timestamp=timestamp,
        )

    @property
    def heading(self) -> float:
        """float: The heading angle (in degrees)."""
        return math.degrees(self._yaw)

    @property
    def elevation(self) -> float:
        """float: The elevation angle (in degrees)."""
        return math.degrees(-self._pitch)

    @property
    def bank(self) -> float:
        """float: The bank angle (in degrees)."""
        return math.degrees(self._roll - HALF_PI)

    @property
    def yaw(self) -> float:
        """float: The normalized yaw angle (in radians)."""
        return self._yaw

    @property
    def pitch(self) -> float:
        """float: The normalized pitch angle (in radians)."""
        return self._pitch

    @property
    def roll(self) -> float:
        """float: The normalized roll angle (in radians)."""
        return self._roll

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(heading={self.heading:.2f}, elevation="
            f"{self.elevation:.2f}, bank={self.bank:.2f},"
            f" speed={self.speed:.2f} {self.units:a})"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.heading:.2f},"
            f" {self.elevation:.2f}, {self.bank:.2f}, {self.speed:.2f},"
            f" {self.units!r})"
        )
