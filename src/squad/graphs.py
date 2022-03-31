from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

import numpy as np

from squad.exceptions import (
    EdgeAlreadyExists,
    EdgeNotFound,
    NodeAlreadyExists,
    NodeNotFound,
)


class Node:
    """
    Single node in a graph.
    """

    def __init__(self, name: str, **data: Any) -> None:
        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        """str: The name of this node."""
        return self._name

    @property
    def data(self) -> Dict[str, Any]:
        """Dict[str, Any]: The data stored in this node (if any)."""
        return self._data.copy()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._name))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Node):
            return self._name == __o._name
        elif isinstance(__o, str):
            return self._name == __o
        raise ValueError(
            f"Cannot compare {self.__class__.__name__} with"
            f" {type(__o).__name__}"
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def update(self, **data: Any) -> None:
        """Updates the data stored on this node.

        Parameters
        ----------
        **data : Any, optional
            The data parameters to update on this node.

        """
        self._data.update(data)


class Edge:
    """
    Single edge in a graph.
    """

    def __init__(
        self,
        u: Node,
        v: Node,
        weight: float = 1.0,
        **data: Any,
    ) -> None:
        self._u = u
        self._v = v
        self._wgt = weight
        self._value: Optional[float] = None
        self._data = data

    @property
    def u(self) -> Node:
        """Node: The first node in this edge."""
        return self._u

    @property
    def v(self) -> Node:
        """Node: The second node in this edge."""
        return self._v

    @property
    def weight(self) -> float:
        """float: The weight of this edge."""
        return self._wgt

    @weight.setter
    def weight(self, value: float) -> None:
        self._wgt = value

    @property
    def value(self) -> float:
        """float: The value of this edge."""
        if self._value is None:
            self._value = self.get_value()
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value

    @property
    def weighted_value(self) -> float:
        """float: The weighted-value of this edge."""
        return self._wgt * self.value

    @property
    def data(self) -> Dict[str, Any]:
        """Dict[str, Any]: The data associated with this edge (if any)."""
        return self._data.copy()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._u.name}, {self._v.name})"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._u, self._v))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Edge):
            return self._u == __o._u and self._v == __o._v
        elif isinstance(__o, tuple):
            return self._u._name == __o[0] and self._v._name == __o[1]
        raise ValueError(
            f"Cannot compare {self.__class__.__name__} with"
            f" {type(__o).__name__}"
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __call__(self, **kwargs: Any) -> float:
        self.update(**kwargs)
        return self._wgt * self.value

    def update(self, **data: Any) -> None:
        """Updates this edge's state.

        Parameters
        ----------
        **data : Any
            Any named-parameters to update the edge's data with.

        """
        if data:
            self._data.update(data)
        self._value = self.get_value()

    def get_value(self) -> float:
        """Gets the value associated with this edge.

        Returns
        -------
        float
            The computed value for this edge.

        """
        return 1.0


def remove_square_matrix_index(matrix: np.ndarray, index: int) -> np.ndarray:
    """Removes the row & column of the specified index from the given
    square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The square matrix to remove the specified `index` row and column
        from.
    index : int
        The index of the row & column to remove from the given `matrix`.

    Returns
    -------
    np.ndarray
        The new matrix, from the original `matrix` given, with the
        desired row & column `index` removed.

    Raises
    ------
    ValueError
        If the given `matrix` is not a square matrix.
    IndexError
        If the given `index` is invalid for the bounds of the given
        `matrix`.

    """
    if matrix.ndim < 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Invalid matrix given, shape: {matrix.shape}")
    elif abs(index) > (matrix.shape[0] - 1):
        raise IndexError(index)
    return np.delete(np.delete(matrix, index, axis=0), index, axis=1)


class Graph:
    """
    Directed graph.
    """

    def __init__(
        self,
        node_cls: Optional[Type[Node]] = None,
        edge_cls: Optional[Type[Edge]] = None,
    ) -> None:
        self._node_cls = node_cls or Node
        self._nodes: List[Node] = []
        self._node_lookup: Dict[str, int] = {}

        self._edge_cls = edge_cls or Edge
        self._edges: List[Edge] = []
        self._edge_lookup: Dict[Tuple[str, str], int] = {}

        self._adj_mat = np.array([], dtype=float)
        self._con_mat = self._adj_mat.copy()

    @property
    def nodes(self) -> Dict[str, Node]:
        """Dict[str, Node]: The nodes contained in this graph."""
        return {x.name: x for x in self._nodes}

    @property
    def edges(self) -> Dict[str, Dict[str, Edge]]:
        """Dict[str, Dict[str, Edge]]: The edges in this graph."""
        ret = {x.name: {} for x in self._nodes}
        for x in self._edges:
            ret[x.u.name][x.v.name] = x
        return ret

    def __getitem__(
        self,
        key: Union[str, Tuple[str, str]],
    ) -> Union[Edge, Node]:
        if isinstance(key, str):
            if key not in self._node_lookup:
                raise NodeNotFound(key)
            return self._nodes[self._node_lookup[key]]
        else:
            if key not in self._edge_lookup:
                raise EdgeNotFound(*key)
            return self._edges[self._edge_lookup[key]]

    def add(self, obj: Union[Edge, Node]) -> None:
        """Adds an edge or node to this graph.

        Parameters
        ----------
        obj : Union[Edge, Node]
            The node or edge object to add to this graph.

        Raises
        ------
        EdgeAlreadyExists
            If the given edge `obj` is already in this graph.
        NodeAlreadyExists
            If the given node `obj` is already in this graph.
        NodeNotFound
            If one or both of the nodes in the given edge `obj` is not
            in this graph.

        """
        if isinstance(obj, Edge):
            if obj in self._edges:
                raise EdgeAlreadyExists(obj.u.name, obj.v.name)
            elif obj.u.name not in self._node_lookup:
                raise NodeNotFound(obj.u.name)
            elif obj.v.name not in self._node_lookup:
                raise NodeNotFound(obj.v.name)
            self._add_edge_obj(obj)
        else:
            if obj in self._nodes:
                raise NodeAlreadyExists(obj.name)
            self._add_node_obj(obj)
        return

    def remove(self, obj: Union[Edge, Node]) -> None:
        """Removes the given edge or node from this graph.

        Parameters
        ----------
        obj : Union[Edge, Node]
            The edge or node object to remove from this graph.

        Raises
        ------
        EdgeNotFound
            If the given edge `obj` could not be found.
        NodeNotFound
            If the given node `obj` could not be found.

        """
        if isinstance(obj, Edge):
            if obj not in self._edges:
                raise EdgeNotFound(obj.u.name, obj.v.name)
            self._remove_edge_obj(obj.u.name, obj.v.name)
        else:
            if obj not in self._nodes:
                raise NodeNotFound(obj.name)
            self._remove_node_obj(obj.name)
        return

    def clear(self) -> None:
        """Clears all nodes and edges from this graph."""
        self._node_lookup.clear()
        self._nodes.clear()
        self._edge_lookup.clear()
        self._edges.clear()
        self._adj_mat = np.array([], dtype=self._adj_mat.dtype)
        self._con_mat = self._adj_mat.copy()

    def _add_edge_obj(self, edge: Edge) -> None:
        """Adds a new edge object to this graph."""
        self._edges.append(edge)
        new_n_edges = len(self._edges)
        self._edge_lookup[(edge.u.name, edge.v.name)] = new_n_edges - 1

        idx_u = self._nodes.index(edge.u)
        idx_v = self._nodes.index(edge.v)
        self._adj_mat[idx_u, idx_v] = 1.0
        self._con_mat[idx_u, idx_v] = 1.0
        if idx_u != idx_v:
            self._con_mat[idx_v, idx_u] = 1.0
        return

    def _remove_edge_obj(self, u_name: str, v_name: str) -> None:
        """Removes the specified edge from this graph."""
        # - Update adjacency/connection matrices
        u_idx = self._node_lookup[u_name]
        v_idx = self._node_lookup[v_name]

        self._adj_mat[u_idx, v_idx] = 0.0
        if u_idx == v_idx:
            self._con_mat[u_idx, v_idx] = 0.0
        elif (v_name, u_name) not in self._edge_lookup:
            self._con_mat[u_idx, v_idx] = 0.0
            self._con_mat[v_idx, u_idx] = 0.0

        # - Remove edge
        edge_idx = self._edge_lookup.pop((u_name, v_name))
        self._edges.pop(edge_idx)

        # - Update lookup table for relevant edges
        edge_names_to_update = [
            (x.u.name, x.v.name) for x in self._edges[edge_idx:]
        ]
        for edge_name in edge_names_to_update:
            self._edge_lookup[edge_name] -= 1
        return

    def _add_node_obj(self, node: Node) -> None:
        """Adds a new node object to this graph."""
        orig_n_nodes = len(self._nodes)
        self._nodes.append(node)
        self._node_lookup[node.name] = orig_n_nodes
        new_n_nodes = orig_n_nodes + 1

        upd_adj_mat = np.zeros(
            (new_n_nodes, new_n_nodes),
            dtype=self._adj_mat.dtype,
        )
        upd_con_mat = upd_adj_mat.copy()

        if orig_n_nodes:
            upd_adj_mat[:orig_n_nodes, :orig_n_nodes] = self._adj_mat
            upd_con_mat[:orig_n_nodes, :orig_n_nodes] = self._con_mat

        self._adj_mat = upd_adj_mat
        self._con_mat = upd_con_mat

    def _remove_node_obj(self, node_name: str) -> None:
        """Removes an existing node object from this graph."""
        node_idx = self._node_lookup[node_name]

        # Update the adjacency/connection matrices
        self._adj_mat = remove_square_matrix_index(self._adj_mat, node_idx)
        self._con_mat = remove_square_matrix_index(self._con_mat, node_idx)

        # - Remove any edge objects connected to the node
        def _edge_filter(x: Tuple[str, str]) -> bool:
            return node_name in x

        edge_idxs_to_remove = sorted(
            (
                self._edge_lookup[k]
                for k in filter(_edge_filter, self._edge_lookup.keys())
            ),
            reverse=True,
        )
        edge_names_to_remove = [
            (x.u.name, x.v.name)
            for x in (self._edges[i] for i in edge_idxs_to_remove)
        ]

        for i, n in zip(edge_idxs_to_remove, edge_names_to_remove):
            del self._edge_lookup[n]
            self._edges.pop(i)

        # - Remove the node object
        self._nodes.pop(node_idx)

        # - Update the lookup tables
        for node in self._nodes[node_idx:]:
            self._node_lookup[node.name] -= 1

        for i, edge in enumerate(self._edges):
            self._edge_lookup[(edge.u.name, edge.v.name)] = i
        return

    def add_node(self, name: str, **data: Any) -> None:
        """Creates and adds a new node to this graph.

        Parameters
        ----------
        name : str
            The name of the node to add to this graph.
        **data : Any
            The data of the node to add to this graph (if any).

        Raises
        ------
        NodeAlreadyExists
            If a node with the same `name` given already exists in this
            graph.

        """
        if name in (x.name for x in self._nodes):
            raise NodeAlreadyExists(name)
        new_node = self._node_cls(name, **data)
        self._add_node_obj(new_node)

    def add_nodes(self, *names: str, **data: Any) -> None:
        """Creates and adds new node(s) to this graph.

        Parameters
        ----------
        *names : str
            The name(s) of the new nodes to create and add.
        **data : Any, optional
            The data (if any) to associate with each of the new nodes.

        Raises
        ------
        NodeAlreadyExists
            If any of the nodes from the given `names` already exist in
            this graph.
        ValueError
            If no `names` are provided.

        """
        for name in names:
            if name in self._node_lookup:
                raise NodeAlreadyExists(name)

        for name in names:
            new_node = self._node_cls(name, **data)
            self._add_node_obj(new_node)
        return

    def remove_node(self, name: str) -> None:
        """Removes the specified node from this graph.

        Parameters
        ----------
        name : str
            The name of the node to remove.

        Raises
        ------
        NodeNotFound
            If the node with the given `name` could not be found.

        """
        if name not in self._node_lookup:
            raise NodeNotFound(name)
        self._remove_node_obj(name)

    def add_edge(
        self,
        u_name: str,
        v_name: str,
        weight: float = 1.0,
        **data: Any,
    ) -> None:
        """Creates and adds a new edge to this graph.

        Parameters
        ----------
        u_name : str
            The name of the (existing) node to set as the first node for
            the new edge to add.
        v_name : str
            The name of the (existing) node to set as the second node
            for the new edge to add.
        weight : float, default=1.0
            The weight to use for the new edge to add.
        **data : Any, optional
            The data (if any) to store on the new edge.

        Raises
        ------
        EdgeAlreadyExists
            If an edge for the given nodes specified already exists in
            this graph.
        NodeNotFound
            If either of the given nodes specified could not be found.

        """
        if (u_name, v_name) in ((x.u.name, x.v.name) for x in self._edges):
            raise EdgeAlreadyExists(u_name, v_name)

        u = None
        v = None
        for node in self._nodes:
            if node.name == u_name:
                u = node
            if node.name == v_name:
                v = node
            if u is not None and v is not None:
                break

        if u is None:
            raise NodeNotFound(u_name)
        if v is None:
            raise NodeNotFound(v_name)

        new_edge = self._edge_cls(u, v, weight=weight, **data)
        self._add_edge_obj(new_edge)

    def add_edges(
        self,
        u_name: str,
        *v_names: str,
        weight: float = 1.0,
        **data: Any,
    ) -> None:
        """Adds multiple edges from `u_name` to this graph.

        Parameters
        ----------
        u_name : str
            The name of the (existing) node to set as the first node for
            the new edges to add.
        *v_names : str
            The names of the (existing) nodes to set as the second node
            for the new edge to add.
        weight : float, default=1.0
            The weight to use for each new edge to add.
        **data : Any, optional
            The data (if any) to store on each new edge.

        Raises
        ------
        EdgeAlreadyExists
            If any edge for the given nodes specified already exists in
            this graph.
        NodeNotFound
            If any of the given nodes specified could not be found.
        ValueError
            If no `v_names` are provided.

        """
        if not v_names:
            raise ValueError("You must provide at least one v node name")

        if u_name not in self._node_lookup:
            raise NodeNotFound(u_name)
        else:
            for v in v_names:
                if v not in self._node_lookup:
                    raise NodeNotFound(v)

        for e in ((u_name, v) for v in v_names):
            if e in self._edge_lookup:
                raise EdgeAlreadyExists(e[0], e[1])

        u_node = self._nodes[self._node_lookup[u_name]]
        for v_name in v_names:
            v_node = self._nodes[self._node_lookup[v_name]]
            new_edge = self._edge_cls(u_node, v_node, weight=weight, **data)
            self._add_edge_obj(new_edge)
        return

    def remove_edge(self, u_name: str, v_name: str) -> None:
        """Removes the edge specified from this graph.

        Parameters
        ----------
        u_name : str
            The name of the first node in the edge to remove.
        v_name : str
            The name of the second node in the edge to remove.

        Raises
        ------
        EdgeNotFound
            If the specified edge could not be found.
        NodeNotFound
            If either node specified by the given `u_name` and `v_name`
            could not be found.

        """
        if u_name not in self._node_lookup:
            raise NodeNotFound(u_name)
        elif v_name not in self._node_lookup:
            raise NodeNotFound(v_name)
        elif (u_name, v_name) not in self._edge_lookup:
            raise EdgeNotFound(u_name, v_name)
        self._remove_edge_obj(u_name, v_name)

    def update_nodes(self, *names: str, **data: Any) -> None:
        """Updates the node(s) in this graph.

        Parameters
        ----------
        *names : str, optional
            The specific node(s) to update (if not given then all nodes
            will be updated).
        **data : Any, optional
            The data updates to push to all nodes in the graph for the
            update calls.

        """
        if names:
            nodes = (x for x in self._nodes if x.name in names)
        else:
            nodes = self._nodes

        for node in nodes:
            node.update(**data)
        return

    def update_edges(self, *names: str, **data: Any) -> None:
        """Updates all the edges in this graph.

        Parameters
        ----------
        *names : str, optional
            The u-node (first node) names of the relevant edges to
            update (if not provided then all edges are updated).
        **data : Any, optional
            Any data updates to push to all edges in the graph for the
            update calls.

        """
        if names:
            edges = (x for x in self._edges if x.u.name in names)
        else:
            edges = self._edges

        for edge in edges:
            edge.update(**data)
        return

    @overload
    def adj_edges(self, u_name: str) -> List[Edge]:
        ...

    @overload
    def adj_edges(self, u_name: str, v_name: str) -> Edge:
        ...

    def adj_edges(
        self,
        u_name: str,
        v_name: Optional[str] = None,
    ) -> Union[Edge, List[Edge]]:
        """Gets the adjacenct edge(s) specified.

        Parameters
        ----------
        u_name : str
            The name of the node to get the adjacent edge(s) *from*.
        v_name : str, optional
            The name of the node to get the adjacent edge(s) *to* (if
            any).  If not specified (default) it will return all
            possible adjacent edges.

        Returns
        -------
        Edge or List[Edge]
            The adjacent edge(s) from the specified `u_name` (if
            `v_name` was not specified).  If `v_name` was given then
            it just returns the adjacent edge from the specified
            `u_name` node to the specified `v_name` node.

        Raises
        ------
        NodeNotFound
            If the specified `u_name` node (or `v_name` node, if given)
            could not be found.
        EdgeNotFound
            If the specified `u_name` to `v_name` (if given) edge could
            not be found.

        See Also
        --------
        adj, adj_values, adj_weights

        """
        u_idx = None
        v_idx = None
        for i, node in enumerate(self._nodes):
            if node.name == u_name:
                u_idx = i
            if v_name is not None:
                if node.name == v_name:
                    v_idx = i
                if u_idx is not None and v_idx is not None:
                    break
            elif u_idx is not None:
                break

        if u_idx is None:
            raise NodeNotFound(u_name)
        if v_name is not None and v_idx is None:
            raise NodeNotFound(v_name)

        if v_name is None:
            # - All adjacent edges
            adj_edges: List[Edge] = []
            for i, v in enumerate(self._adj_mat[u_idx]):
                if v == 0.0:
                    continue
                v_node = self._nodes[i]
                t_edge = self._edges[self._edge_lookup[(u_name, v_node.name)]]
                adj_edges.append(t_edge)

            return adj_edges
        else:
            # - Single edge
            try:
                adj_edge = self._edges[self._edge_lookup[(u_name, v_name)]]
            except KeyError:
                raise EdgeNotFound(u_name, v_name)
            return adj_edge

    @overload
    def adj_values(
        self,
        u_name: str,
    ) -> Dict[str, float]:
        ...

    @overload
    def adj_values(
        self,
        u_name: str,
        v_name: str,
    ) -> float:
        ...

    def adj_values(
        self,
        u_name: str,
        v_name: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """Gets the adjacency edge value(s) for the specified node/edge.

        Parameters
        ----------
        u_name : str
            The name of the node to get the adjacency data *from*.
        v_name : str, optional
            The name of the node to get the adjacency data *to* (if
            any).  If not specified (default) it will return all
            possible adjacent nodes and values.

        Returns
        -------
        float or Dict[str, float]
            The adjacent edges and values from the specified `u_name`
            (if `v_name` was not specified).  If `v_name` was given then
            it just returns the value of the adjacency edge from the
            specified `u_name` node to the specified `v_name` node.

        See Also
        --------
        adj, adj_edges, adj_weights

        """
        # - Single edge value
        if v_name is not None:
            edge = self.adj_edges(u_name, v_name)
            return edge.value

        # - All adjacent edge values
        edges = self.adj_edges(u_name)
        ret = {x.v.name: x.value for x in edges}
        return ret

    @overload
    def adj(
        self,
        u_name: str,
    ) -> Dict[str, float]:
        ...

    @overload
    def adj(
        self,
        u_name: str,
        v_name: str,
    ) -> float:
        ...

    def adj(
        self,
        u_name: str,
        v_name: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """Gets the adjacency edge weighted-value(s) for the specified
        node/edge.

        Parameters
        ----------
        u_name : str
            The name of the node to get the adjacency data *from*.
        v_name : str, optional
            The name of the node to get the adjacency data *to* (if
            any).  If not specified (default) it will return all
            possible adjacent nodes and values.

        Returns
        -------
        float or Dict[str, float]
            The adjacent edges and weighted-values from the specified
            `u_name` (if `v_name` was not specified).  If `v_name` was
            given then it just returns the weighted-value of the
            adjacent edge from the specified `u_name` node to the
            specified `v_name` node.

        See Also
        --------
        adj_edges, adj_values, adj_weights

        """
        # - Single edge value
        if v_name is not None:
            edge = self.adj_edges(u_name, v_name)
            return edge.weighted_value

        # - All adjacent edge values
        edges = self.adj_edges(u_name)
        ret = {x.v.name: x.weighted_value for x in edges}
        return ret

    @overload
    def adj_weights(
        self,
        u_name: str,
    ) -> Dict[str, float]:
        ...

    @overload
    def adj_weights(
        self,
        u_name: str,
        v_name: str,
    ) -> float:
        ...

    def adj_weights(
        self,
        u_name: str,
        v_name: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """Gets the adjacency edge weight(s) of the specified node/edge.

        Parameters
        ----------
        u_name : str
            The name of the node to get the adjacency data *from*.
        v_name : str, optional
            The name of the node to get the adjacency data *to* (if
            any).  If not specified (default) it will return all
            possible adjacent nodes and values.

        Returns
        -------
        float or Dict[str, float]
            The adjacent edges and weight(s) from the specified `u_name`
            node (if `v_name` was not specified).  If `v_name` was given
            then it just returns the raw value of the adjacent edge from
            the specified `u_name` node to the specified `v_name` node.

        See Also
        --------
        adj, adj_edges, adj_values

        """
        # - Single edge value
        if v_name is not None:
            edge = self.adj_edges(u_name, v_name)
            return edge.weight

        # - All adjacent edge values
        edges = self.adj_edges(u_name)
        ret = {x.v.name: x.weight for x in edges}
        return ret
