from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Type

from squad.graphs import Edge, Graph, Node


class MdpNode(Node):
    """
    Markov decision process node.
    """

    def __init__(
        self,
        name: str,
        reward: Optional[float] = None,
        **data: Any,
    ) -> None:
        self._reward = reward
        return super().__init__(name, **data)

    @property
    def reward(self) -> float:
        """float: The current reward value associated with this edge."""
        if self._reward is None:
            self._reward = self.get_reward()
        return self._reward

    @reward.setter
    def reward(self, value: float) -> None:
        self._reward = value

    @abstractmethod
    def get_reward(self, *args: Any, **kwargs: Any) -> float:
        """Computes the reward value associated with this edge.

        Returns
        -------
        float
            The current reward value associated with this edge.

        """
        raise NotImplementedError


class MdpEdge(Edge, metaclass=ABCMeta):
    """
    Markov Decision Process edge.
    """

    def __init__(
        self,
        u: MdpNode,
        v: MdpNode,
        weight: float = 1,
        *,
        learning: bool = False,
        **data: Any,
    ) -> None:
        self._learn = learning
        self._reward: Optional[float] = None
        return super().__init__(u, v, weight, **data)

    @property
    def learning(self) -> bool:
        """bool: Whether or not this edge is in learning mode."""
        return self._learn

    @learning.setter
    def learning(self, value: bool) -> None:
        self._learn = value


class MdpGraph(Graph):
    """
    Markov Decision Process graph.
    """

    if TYPE_CHECKING:
        _node_cls: Type[MdpNode]
        _nodes: List[MdpNode]
        _edge_cls: Type[MdpEdge]
        _edges: List[MdpEdge]

    def __init__(
        self,
        node_cls: Type[MdpNode],
        edge_cls: Type[MdpEdge],
        *,
        discount: float = 0.0,
        learning: bool = False,
    ) -> None:
        if not (0.0 <= discount <= 1.0):
            raise ValueError("Discount factor must be in: [0, 1]")
        self._discount = discount
        self._learn = learning
        return super().__init__(node_cls, edge_cls)

    @property
    def discount(self) -> float:
        """float: The current discount factor in use."""
        return self._discount

    @discount.setter
    def discount(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("Discount factor must be in: [0, 1]")
        self._discount = value

    @property
    def learning(self) -> bool:
        """bool: Whether or not this graph is in learning mode."""
        return self._learn

    @learning.setter
    def learning(self, value: bool) -> None:
        for edge in self._edges:
            edge.learning = value
        self._learn = value

    def add_node(
        self,
        name: str,
        reward: Optional[float] = None,
        **data: Any,
    ) -> None:
        return super().add_node(name, reward=reward, **data)

    def add_nodes(
        self,
        *names: str,
        reward: Optional[float] = None,
        **data: Any,
    ) -> None:
        return super().add_nodes(*names, reward=reward, **data)

    def add_edge(
        self,
        u_name: str,
        v_name: str,
        weight: float = 1,
        **data: Any,
    ) -> None:
        return super().add_edge(
            u_name,
            v_name,
            weight,
            learning=self._learn,
            **data,
        )

    def add_edges(
        self,
        u_name: str,
        *v_names: str,
        weight: float = 1,
        **data: Any,
    ) -> None:
        return super().add_edges(
            u_name,
            *v_names,
            weight=weight,
            learning=self._learn,
            **data,
        )
