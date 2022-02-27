from enum import IntEnum
from math import pi


# Math constants
PI = pi
HALF_PI = PI / 2.0


# Physical constants
GRAVITY = -9.8


# Enumerations


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
