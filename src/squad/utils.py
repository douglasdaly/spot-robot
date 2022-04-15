from squad.constants import (
    LENGTH_CONVERSION_FACTORS,
    TIME_CONVERSION_FACTORS,
    LengthUnits,
    TimeUnit,
    VelocityUnits,
)


def _time_conv_factor_seconds(t_type: TimeUnit) -> float:
    """Gets the conversion factor to use for the given time type/units."""
    if t_type not in TIME_CONVERSION_FACTORS:
        raise NotImplementedError(t_type)
    return TIME_CONVERSION_FACTORS[t_type]


def convert_time(
    t_in: float,
    from_units: TimeUnit,
    to_units: TimeUnit,
) -> float:
    """Converts the given length of time from one basis to another.

    Parameters
    ----------
    t_in : float
        The length of time to convert.
    from_units : TimeUnit
        The units/type of the given `t_in` value to convert from.
    to_units : TimeUnit
        The units/type of the desired output value to convert to.

    Returns
    -------
    float
        The converted value of `t_in` in the desired units specified.

    """
    from_factor = _time_conv_factor_seconds(from_units)
    to_factor = _time_conv_factor_seconds(to_units)
    return (from_factor / to_factor) * t_in


def _length_conv_factor_meters(l_type: LengthUnits) -> float:
    """Gets the conversion factor for the given distance type/units."""
    if l_type not in LENGTH_CONVERSION_FACTORS:
        raise NotImplementedError(l_type)
    return LENGTH_CONVERSION_FACTORS[l_type]


def convert_length(
    l_in: float,
    from_units: LengthUnits,
    to_units: LengthUnits,
) -> float:
    """Converts the given distance from one basis to another.

    Parameters
    ----------
    l_in : float
        The length to convert.
    from_units : LengthUnits
        The units/type of the given `l_in` value to convert from.
    to_units : LengthUnits
        The units/type of the desired output value to convert to.

    Returns
    -------
    float
        The converted value of `l_in` in the desired units specified.

    """
    from_factor = _length_conv_factor_meters(from_units)
    to_factor = _length_conv_factor_meters(to_units)
    return (from_factor / to_factor) * l_in


def convert_velocity(
    v_in: float,
    from_units: VelocityUnits,
    to_units: VelocityUnits,
) -> float:
    """Converts the given velocity from one basis to another.

    Parameters
    ----------
    v_in : float
        The velocity to convert.
    from_units : VelocityUnits
        The units/type of the given `v_in` value to convert from.
    to_units : VelocityUnits
        The units/type of the desired output value to convert to.

    Returns
    -------
    float
        The converted value of `v_in` in the desired units specified.

    """
    from_l_factor = _length_conv_factor_meters(from_units.length_units)
    from_t_factor = _time_conv_factor_seconds(from_units.time_unit)
    to_l_factor = _length_conv_factor_meters(to_units.length_units)
    to_t_factor = _time_conv_factor_seconds(to_units.time_unit)
    return (
        (from_l_factor / from_t_factor) / (to_l_factor / to_t_factor)
    ) * v_in
