from collections import namedtuple
from enum import IntEnum
from math import pi
from typing import TYPE_CHECKING, Dict


# Math constants
PI = pi
HALF_PI = PI / 2.0

LIMIT_LT_1 = 1.0 - 1e-16


# Physical constants
GRAVITY = -9.8


# Enumerations


class Direction(IntEnum):
    """
    Enumeration for movement directions.
    """

    REVERSE = -1
    NONE = 0
    FORWARD = 1


class Leg(IntEnum):
    """
    Enumeration for leg identification.
    """

    FL = 1
    FR = 2
    BL = 3
    BR = 4


class AngleType(IntEnum):
    """
    Enumeration for angle formats.
    """

    DEGREES = 1
    RADIANS = 2

    @classmethod
    def from_string(cls, value: str) -> "AngleType":
        """Attempts to infer the angle type from the given string value.

        Parameters
        ----------
        value : str
            The value to infer the angle type from, can either be the
            name (singular or plural) or the abbreviation (e.g. "DEG"
            for degrees or "rad" for radians).

        Returns
        -------
        AngleType
            The angle type inferred from the given string `value`.

        Raises
        ------
        NotImplementedError
            If the given string `value` wasn't recognized or isn't
            supported.

        """
        v_fmt = value.strip().lower()
        if len(v_fmt) > 4:
            # - Name, get plural form and try to match
            v_single = v_fmt.upper()
            if v_single.endswith("S"):
                v_plural = v_single
                v_single = v_single[:-1]
            else:
                v_plural = f"{v_single}S"
            for a_type in cls:
                if a_type.name == v_plural:
                    return a_type
        else:
            if v_fmt.endswith("s"):
                v_fmt = v_fmt[:-1]
            if v_fmt in ANGLE_TYPE_SYMBOLS.values():
                for k, v in ANGLE_TYPE_SYMBOLS.items():
                    if v == v_fmt:
                        return k
        raise NotImplementedError(value)


ANGLE_TYPE_SYMBOLS: Dict[AngleType, str] = {
    AngleType.DEGREES: "deg",
    AngleType.RADIANS: "rad",
}


class TimeUnit(IntEnum):
    """
    Enumeration for time units.
    """

    MICROSECOND = 1
    MILLISECOND = 2
    SECOND = 3
    MINUTE = 4
    HOUR = 5
    DAY = 6

    def __format__(self, format_spec: str) -> str:
        if format_spec and format_spec[-1] == "a":
            return TIME_UNIT_SYMBOLS[self].__format__(format_spec[:-1])
        return self.name.__format__(format_spec)

    @classmethod
    def from_string(cls, value: str) -> "TimeUnit":
        """Attempts to infer the time unit/type from the given string.

        Parameters
        ----------
        value : str
            The string value to attempt to infer the time units from.
            Can either be the full-name (in singular or plural form) to
            match against (e.g. "second" or "seconds") or the unit
            abbreviation (e.g. "s" for seconds, "us" for microseconds,
            "min" for minutes, etc.).

        Returns
        -------
        TimeUnit
            The time unit/type inferred from the given string `value`.

        Raises
        ------
        NotImplementedError
            If the given `value` string was not recognized or is not
            supported.

        """
        v_fmt = value.strip().lower()
        if len(v_fmt) >= 3:
            # - Name, get singular and check
            v_single = v_fmt.upper()
            if v_single.endswith("S"):
                v_single = v_single[:-1]
            for t_type in cls:
                if t_type.name == v_single:
                    return t_type
        if v_fmt in TIME_UNIT_SYMBOLS.values():
            # - Known abbreviation value
            for k, v in TIME_UNIT_SYMBOLS.items():
                if v == v_fmt:
                    return k
        raise NotImplementedError(value)


TIME_UNIT_SYMBOLS: Dict[TimeUnit, str] = {
    TimeUnit.MICROSECOND: "us",
    TimeUnit.MILLISECOND: "ms",
    TimeUnit.SECOND: "s",
    TimeUnit.MINUTE: "min",
    TimeUnit.HOUR: "h",
    TimeUnit.DAY: "d",
}

TIME_CONVERSION_FACTORS: Dict[TimeUnit, float] = {
    TimeUnit.MICROSECOND: 1.0 / 1000000.0,
    TimeUnit.MILLISECOND: 1.0 / 1000.0,
    TimeUnit.SECOND: 1.0,
    TimeUnit.MINUTE: 60.0,
    TimeUnit.HOUR: 60.0 * 60.0,
    TimeUnit.DAY: 24.0 * 60.0 * 60.0,
}


class LengthUnits(IntEnum):
    """
    Enumeration for length units.
    """

    MILLIMETERS = 1
    CENTIMETERS = 2
    METERS = 3
    KILOMETERS = 4
    INCHES = 5
    FEET = 6
    YARDS = 7
    MILES = 8

    def __format__(self, format_spec: str) -> str:
        if format_spec and format_spec[-1] == "a":
            return LENGTH_UNIT_SYMBOLS[self].__format__(format_spec[:-1])
        return self.name.__format__(format_spec)

    @classmethod
    def from_string(cls, value: str) -> "LengthUnits":
        """Attempts to infer a length type/unit from a given string.

        Parameters
        ----------
        value : str
            The string-value to attempt to infer the length units from.
            Either the plural or singular name can be given (e.g. "inch"
            or "inches", "foot" or "feet", "meter" or "meters", etc.),
            or the abbreviation (e.g. "ft" for feet, "mm" for
            millimeters, "mi" for miles, etc.).

        Returns
        -------
        LengthUnits
            The length units inferred from the given value.

        Raises
        ------
        NotImplementedError
            If the given `value` string is not supported/recognized.

        """
        v_fmt = value.strip().lower()
        if len(v_fmt) > 2:
            # - A unit name, get the plural version
            v_single = v_fmt.upper()
            if v_single.endswith("S"):
                v_single = v_single[:-1]
            if v_single == "FOOT":
                v_plural = "FEET"
            elif v_single == "FEET":
                v_plural = "FEET"
                v_single = "FOOT"
            else:
                v_plural = f"{v_single}S"

            # - See if it matches one of the enums
            for d_type in cls:
                if d_type.name == v_plural:
                    return d_type
        elif v_fmt in LENGTH_UNIT_SYMBOLS.values():
            # - Unit symbol (known)
            for k, v in LENGTH_UNIT_SYMBOLS.items():
                if v == v_fmt:
                    return k
        raise NotImplementedError(value)


LENGTH_UNIT_SYMBOLS: Dict[LengthUnits, str] = {
    LengthUnits.MILLIMETERS: "mm",
    LengthUnits.CENTIMETERS: "cm",
    LengthUnits.METERS: "m",
    LengthUnits.KILOMETERS: "km",
    LengthUnits.INCHES: "in",
    LengthUnits.FEET: "ft",
    LengthUnits.YARDS: "yd",
    LengthUnits.MILES: "mi",
}


LENGTH_CONVERSION_FACTORS: Dict[LengthUnits, float] = {
    LengthUnits.MILLIMETERS: 1.0 / 1000.0,
    LengthUnits.CENTIMETERS: 1.0 / 100.0,
    LengthUnits.METERS: 1.0,
    LengthUnits.KILOMETERS: 1000.0,
    LengthUnits.INCHES: 1.0 / 39.37,
    LengthUnits.FEET: 12.0 / 39.37,
    LengthUnits.YARDS: 36.0 / 39.37,
    LengthUnits.MILES: 63360.0 / 39.37,
}


class VelocityUnits(namedtuple("VelocityType", ["length_units", "time_unit"])):
    """
    Represents a velocity type/units.
    """

    if TYPE_CHECKING:
        length_units: LengthUnits
        time_unit: TimeUnit

    @classmethod
    @property
    def METERS_PER_SECOND(cls) -> "VelocityUnits":
        return cls(LengthUnits.METERS, TimeUnit.SECOND)

    @classmethod
    @property
    def FEET_PER_SECOND(cls) -> "VelocityUnits":
        return cls(LengthUnits.FEET, TimeUnit.SECOND)

    @classmethod
    @property
    def MILES_PER_HOUR(cls) -> "VelocityUnits":
        return cls(LengthUnits.MILES, TimeUnit.HOUR)

    @classmethod
    @property
    def KILOMETERS_PER_HOUR(cls) -> "VelocityUnits":
        return cls(LengthUnits.KILOMETERS, TimeUnit.HOUR)

    def __format__(self, format_spec: str) -> str:
        if format_spec and format_spec[-1] == "a":
            ret_s = f"{self.length_units:a}/{self.time_unit:a}"
            format_spec = format_spec[:-1]
        else:
            ret_s = f"{self.length_units}/{self.time_unit}"
        return ret_s.__format__(format_spec)

    @classmethod
    def from_string(cls, value: str) -> "VelocityUnits":
        """Creates a new VelocityUnits object from a given string value.

        The allowed formats for the given `value` must be of one of the
        types shown in these examples:

        - meters/second
        - km/h
        - feet per minute
        - MILES_PER_HOUR

        Spacing is important if using a format with the work "per".  You
        also cannot use short-hand abbreviations for per (aside from
        "/"), such as "mph".

        Parameters
        ----------
        value : str
            The string-value to get the associated velocity type for, in
            one of the allowed formats.

        Returns
        -------
        VelocityUnits
            The velocity units determined from the given string value.

        Raises
        ------
        ValueError
            If the given string was not formatted in one of the allowed
            ways.
        NotImplementedError
            If any of the units in the given `value` string are not
            supported.

        """
        # - Format input into distance and time string values
        v_fmt = (
            value.strip()
            .replace("_", " ")
            .lower()
            .replace(" per ", "/")
            .replace(" ", "")
        )
        v_splat = v_fmt.split("/", maxsplit=2)
        if len(v_splat) != 2:
            raise ValueError(
                f"Unable to extract distance and time string from: {value}"
            )

        # - Attempt to get distance/time types and return
        l_type = LengthUnits.from_string(v_splat[0])
        t_type = TimeUnit.from_string(v_splat[1])
        return cls(l_type, t_type)
