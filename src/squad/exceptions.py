class SquadException(Exception):
    pass


class ConfigError(SquadException):
    """
    Exception thrown if a configuration error occurs.
    """

    pass


class StateError(SquadException):
    """
    Exception thrown if an error occurs with a State object.
    """

    pass


class FrozenError(SquadException):
    """
    Exception thrown if an object is frozen and cannot be modified.
    """

    pass
