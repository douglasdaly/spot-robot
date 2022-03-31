from squad.constants import TIME_CONVERSION_FACTORS, TimeType


def _time_conv_factor_seconds(t_type: TimeType) -> float:
    """Gets the conversion factor to use for the given time type/units."""
    if t_type not in TIME_CONVERSION_FACTORS:
        raise NotImplementedError(t_type)
    return TIME_CONVERSION_FACTORS[t_type]


def convert_time(t_in: float, from_type: TimeType, to_type: TimeType) -> float:
    """Converts the given length of time from one basis to another.

    Parameters
    ----------
    t_in : float
        The length of time to convert.
    from_type : TimeType
        The units/type of the given `t_in` value to convert from.
    to_type : TimeType
        The units/type of the desired output value to convert to.

    Returns
    -------
    float
        The converted value of `t_in` in the desired units specified.

    """
    from_factor = _time_conv_factor_seconds(from_type)
    to_factor = _time_conv_factor_seconds(to_type)
    return (from_factor / to_factor) * t_in
