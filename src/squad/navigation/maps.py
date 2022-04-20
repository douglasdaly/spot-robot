from typing import Any, ClassVar, List, Optional, Tuple, Type, Union

from euclid import Line2, LineSegment2, Point2, Ray2, Vector2

from squad.compute.geometry import Polygon2, ccw2
from squad.config import config
from squad.constants import LengthUnits


class Boundary2:
    """
    Represents a 2D map boundary.
    """

    __slots__ = (
        "_name",
        "_bound",
        "_direction",
    )

    def __init__(
        self,
        name: str,
        bound: Union[Line2, LineSegment2, Ray2],
        direction: Vector2,
    ) -> None:
        self._name = name
        self._bound = bound
        self._direction = direction.normalized()

    @property
    def name(self) -> str:
        """str: The name of this boundary."""
        return self._name

    @property
    def bound(self) -> Union[Line2, LineSegment2, Ray2]:
        """Union[Line2, LineSegment2, Ray2]: Line defining the boundary."""
        return self._bound

    @property
    def direction(self) -> Vector2:
        """Vector2: Direction to which the boundary applies."""
        return self._direction

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name},"
            f" bound={self._bound}, direction={self._direction})"
        )

    def is_valid(self, point: Union[Tuple[float, float], Point2]) -> bool:
        """Checks if the given point is valid w.r.t. this boundary.

        Parameters
        ----------
        point : Tuple[float, float] or Point2
            The point to check for validity w.r.t. this boundary.

        Returns
        -------
        bool
            Whether or not the given point is valid w.r.t. this
            boundary.

        """
        if isinstance(point, tuple):
            pt = Point2(*point)
        else:
            pt = point
        return self._point_is_valid(pt)

    def _point_is_valid(self, point: Point2) -> bool:
        """Checks if the given point is valid for this boundary."""
        chk_ray = Ray2(point, -self._direction)
        return chk_ray.intersect(self._bound) is not None

    def _line_is_valid(self, line: Union[Line2, LineSegment2, Ray2]) -> bool:
        """Checks if the given line is valid for this boundary."""
        if line.intersect(self._bound) is not None:
            return False
        return self._point_is_valid(line.p)


class Obstacle2:
    """
    Represents a 2D obstacle on a map.
    """

    __slots__ = (
        "_name",
        "_shape",
        "_collision",
        "_bbox",
    )

    def __init__(
        self,
        name: str,
        shape: Polygon2,
        collision_shape: Optional[Polygon2] = None,
    ) -> None:
        self._name = name
        self._shape = shape
        self._collision = collision_shape or shape
        self._bbox = self._compute_bounding_box(self._collision)

    @property
    def name(self) -> str:
        """str: The name of this obstacle."""
        return self._name

    @property
    def shape(self) -> Polygon2:
        """Polygon2: The shape of this obstacle."""
        return self._shape

    @property
    def collision(self) -> Polygon2:
        """Polygon2: The collision shape of this obstacle."""
        return self._collision

    @property
    def bounding_box(self) -> Polygon2:
        """Polygon2: The bounding-box shape to avoid for this obstacle."""
        return self._bbox

    @classmethod
    def _compute_bounding_box(cls, shape: Polygon2) -> Polygon2:
        """Computes the bounding box to use for this obstacle."""
        min_x, min_y = max_x, max_y = shape._points[0]
        for point in shape._points[1:]:
            min_x = max(min_x, point.x)
            max_x = min(max_x, point.x)
            min_y = max(min_y, point.y)
            max_y = min(max_y, point.y)
        return Polygon2(
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
        )


class Map2:
    """
    Represents a map of an area in 2D.
    """

    __boundary_cls__: ClassVar[Type[Boundary2]] = Boundary2
    __obstacle_cls__: ClassVar[Type[Obstacle2]] = Obstacle2

    def __init__(
        self,
        name: str,
        *,
        length_units: Optional[LengthUnits] = None,
    ) -> None:
        self._name = name
        self._units = length_units or config.nav_length_units
        self._bounds: List[Boundary2] = []
        self._obstacles: List[Obstacle2] = []

    @property
    def name(self) -> str:
        """str: The name of this map."""
        return self._name

    @property
    def units(self) -> LengthUnits:
        """LengthUnits: The length units used for this map."""
        return self._units

    @property
    def bounds(self) -> List[Boundary2]:
        """List[Boundary2]: The boundaries for this map."""
        return self._bounds

    @property
    def obstacles(self) -> List[Obstacle2]:
        """List[Obstacle2]: The obstacles for this map."""
        return self._obstacles

    def add(self, obj: Union[Boundary2, Obstacle2]) -> None:
        """Adds a new boundary or obstacle to this map.

        Parameters
        ----------
        obj : Union[Boundary2, Obstacle2]
            The boundary or obstacle to add.

        """
        if isinstance(obj, Boundary2):
            self._bounds.append(obj)
        else:
            self._obstacles.append(obj)

    def add_bound(
        self,
        point_a: Union[Tuple[float, float], Point2],
        point_b: Union[Tuple[float, float], Point2],
        bound_type: Optional[str] = None,
        *,
        away: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Boundary2:
        """Adds a new boundary line to this map.

        Parameters
        ----------
        point_a : Union[Tuple[float, float], Point2]
            The first point defining the new boundary to add.
        point_b : Union[Tuple[float, float], Point2]
            The second point defining the new boundary to add.
        bound_type : str, optional
            The type of boundary line to add, can be ``line`` (for a
            line extending to infinity), ``ray`` (for a ray directed
            from `point_a` to `point_b` to infinity), or a ``segment``
            (for a finite line segment from `point_a` to `point_b`,
            default).
        away : bool, default=True
            Specifies the direction to which the boundary applies.  If
            ``True`` (default) this means the area beyond the boundary
            (in the direction of the center of the map/origin toward the
            boundary) is off-limits.  If ``False`` it's the opposite.
        name : str, optional
            The name to set for the new boundary to add, if not provided
            a default will be used.
        **kwargs : Any, optional
            Any additional keyword-arguments to pass to the boundary
            class constructor.

        """
        if isinstance(point_a, tuple):
            pt_a = Point2(*point_a)
        else:
            pt_a = point_a
        if isinstance(point_b, tuple):
            pt_b = Point2(*point_b)
        else:
            pt_b = point_b

        if bound_type is None:
            b_type = "segment"
        else:
            b_type = bound_type.strip().lower()

        if name is None:
            b_name = f"bound_{b_type[0]}_{len(self._bounds)+1}"
        else:
            b_name = name

        # - Create boundary geometric linear object
        if b_type == "line":
            bnd_line = Line2(pt_a, pt_b)
        elif b_type == "ray":
            bnd_line = Ray2(pt_a, pt_b)
        else:
            bnd_line = LineSegment2(pt_a, pt_b)

        # - Create boundary direction/vector
        is_ccw = ccw2((0.0, 0.0), pt_a, pt_b)
        if (away and is_ccw) or (not away and not is_ccw):
            bnd_vector = pt_b - pt_a
        else:
            bnd_vector = pt_a - pt_b
        bnd_vector.normalize()

        # - Create boundary object
        new_bnd = self.__boundary_cls__(
            b_name,
            bnd_line,
            bnd_vector,
            **kwargs,
        )
        self._bounds.append(new_bnd)
        return new_bnd

    def add_obstacle(
        self,
        location: Union[Tuple[float, float], Point2],
        *shape_points: Union[Tuple[float, float], Point2],
        buffer: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Obstacle2:
        """Adds a new obstacle to this map.

        Parameters
        ----------
        location : Union[Tuple[float, float], Point2]
            The point to place the obstacle at, also used as the
            reference origin for the given `shape_points`.
        *shape_points : Union[Tuple[float, float], Point2]
            The points defining the obstacle's shape, relative to the
            given `location`.
        buffer : float, optional
            The amount of space around the obstacles edges to use for
            a buffer area to avoid (if any).
        name : str, optional
            The name to use for the new obstacle, if not specified a
            default name will be given.
        **kwargs : Any, optional
            Any additional keyword-arguments to pass to the obstacle
            class constructor.

        Returns
        -------
        Obstacle2
            The newly-created and added obstacle object.

        """
        if name is None:
            obs_name = f"obstacle_{len(self._obstacles)+1}"
        else:
            obs_name = name

        obs_shape = Polygon2(*shape_points)

        if buffer is not None:
            buf_pts: List[Point2] = []
            for p, v in zip(obs_shape.points, obs_shape.vectors):
                buf_pts.append(p + (v.normalized() * buffer))  # type: ignore
            buf_shape = Polygon2(*buf_pts)
            buf_shape.translate(location[0], location[1])
        else:
            buf_shape = None

        obs_shape.translate(location[0], location[1])

        # - Create obstacle object & add to map
        new_obs = self.__obstacle_cls__(
            obs_name,
            obs_shape,
            collision_shape=buf_shape,
            **kwargs,
        )
        self._obstacles.append(new_obs)
        return new_obs
