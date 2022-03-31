from typing import Any


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


# - Graphs


class GraphException(SquadException):
    """
    Base class for graph-related exceptions.
    """

    pass


class NodeNotFound(GraphException):
    """
    Exception thrown if a node could not be found in a graph.
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        return super().__init__(*args, **kwargs)


class EdgeNotFound(GraphException):
    """
    Exception thrown if an edge could not be found in a graph.
    """

    def __init__(
        self,
        u_name: str,
        v_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.u_name = u_name
        self.v_name = v_name
        return super().__init__(*args, **kwargs)


class NodeAlreadyExists(GraphException):
    """
    Exception thrown if a node already exists in a graph.
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        return super().__init__(*args, **kwargs)


class EdgeAlreadyExists(GraphException):
    """
    Exception thrown if an edge already exists in a graph.
    """

    def __init__(
        self,
        u_name: str,
        v_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.u_name = u_name
        self.v_name = v_name
        return super().__init__(*args, **kwargs)
