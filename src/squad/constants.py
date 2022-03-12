from enum import IntEnum
from math import pi


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
