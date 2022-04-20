import math
from typing import List, Optional, Tuple, Union

from euclid import Line2, LineSegment2, Matrix3, Point2, Ray2, Vector2

from squad.constants import AngleType


T_Point2 = Union[Tuple[float, float], Point2]


def ccw2(a: T_Point2, b: T_Point2, c: T_Point2) -> bool:
    """Determines if the turn defined by `a`, `b`, and `c` is CCW.

    Parameters
    ----------
    a : Union[Tuple[float, float], Point2]
        The first point in the turn.
    b : Union[Tuple[float, float], Point2]
        The second point in the turn.
    c : Union[Tuple[float, float], Point2]
        The third point in the turn.

    Returns
    -------
    bool
        Whether or not the turn is CCW (counter-clockwise).

    """
    return (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0])


class Path2:
    """
    2D Linear path.
    """

    __slots__ = (
        "_points",
        "_segments",
        "_steps",
        "_vectors",
    )

    def __init__(self, *points: T_Point2) -> None:
        if len(points) < 1:
            raise ValueError("Paths must have at least 2 points")

        self._points: List[Point2] = []
        for point in points:
            if isinstance(point, tuple):
                new_pt = Point2(*point)
            else:
                new_pt = point
            self._points.append(new_pt)
        self._vectors = [Vector2(x) for x in self._points]
        self._segments = self._construct_segments(self._points)
        self._steps: Optional[List[Vector2]] = None

    @property
    def points(self) -> List[Point2]:
        """List[Point2]: The points defining this path."""
        return self._points

    @property
    def segments(self) -> List[LineSegment2]:
        """List[LineSegment2]: The line segments making up this path."""
        return self._segments

    @property
    def vectors(self) -> List[Vector2]:
        """List[Vector2]: The vectors extending to this path's points."""
        return self._vectors

    @property
    def steps(self) -> List[Vector2]:
        """List[Vector2]: The path segment vectors from point-to-point."""
        if self._steps is None:
            self._steps = []
            for i in range(1, len(self._vectors)):
                v_prev = self._vectors[i - 1]
                v_curr = self._vectors[i]
                self._steps.append(v_curr - v_prev)
        return self._steps

    @property
    def total_length(self) -> float:
        """float: The total length of this path."""
        return sum(x.length for x in self._segments)

    @property
    def net_change(self) -> Vector2:
        """Vector2: The net vector from start to finish of this path."""
        return self._points[-1] - self._points[0]

    @property
    def net_length(self) -> float:
        """float: The net length from the start point to the end point."""
        return (self._points[-1] - self._points[0]).magnitude()

    def append(self, next_step: Union[T_Point2, Vector2]) -> "Path2":
        """Appends (and updates) this path with the next location given.

        Parameters
        ----------
        next_step : Tuple[float, float], Point2, or Vector2
            The next path endpoint (or step vector) to add to the end of
            this path.

        Returns
        -------
        Path2
            The updated path object (not a copy) with the next step
            given appended.

        """
        if isinstance(next_step, tuple):
            nxt_pt = Point2(*next_step)
        elif isinstance(next_step, Point2):
            nxt_pt = next_step
        else:
            nxt_pt: Point2 = self._points[-1] + next_step  # type: ignore
        return self._append_point(nxt_pt)

    def translate(self, dx: float, dy: float) -> "Path2":
        """Translates (and updates) this path by the given amounts.

        Parameters
        ----------
        dx : float
            The amount to translate this path in the X-direction.
        dy : float
            The amount to translate this path in the Y-direction.

        Returns
        -------
        Path2
            The updated path (not a copy) translated by the given
            amounts (so that operations can be chained).

        """
        m_trans: Matrix3 = Matrix3.new_translate(dx, dy)  # type: ignore
        return self._apply_transform(m_trans)

    def rotate(
        self,
        angle: float,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> "Path2":
        """Rotates (and updates) this path by the given angle.

        Parameters
        ----------
        angle : float
            The amount of rotation to apply to this path.
        angle_type : AngleType, default=AngleType.DEGREES
            The type/format of the given `angle`.

        Returns
        -------
        Path2
            The updated path (not a copy) with the rotation applied (so
            that operations can be chained).

        """
        if angle_type == AngleType.DEGREES:
            r_ang = math.radians(angle)
        else:
            r_ang = angle
        m_rot: Matrix3 = Matrix3.new_rotate(r_ang)  # type: ignore
        return self._apply_transform(m_rot)

    def scale(self, factor: float) -> "Path2":
        """Scales (and updates) this path uniformly by the given factor.

        Parameters
        ----------
        factor : float
            The scaling-factor to apply to this path, uniformly.

        Returns
        -------
        Path2
            The updated path (not a copy) with the scaling applied
            uniformly (so that operations can be chained).

        Raises
        ------
        ValueError
            If the scaling-factor given is less than or equal to
            ``0.0``.

        """
        if factor <= 0.0:
            raise ValueError("Cannot scale by values less than or equal to 0")
        m_scale: Matrix3 = Matrix3.new_scale(factor, factor)  # type: ignore
        return self._apply_transform(m_scale)

    def transform(
        self,
        *,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        angle: Optional[float] = None,
        scale: Optional[float] = None,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> "Path2":
        """Applies the given transformations to this path (and updates
        it).

        The order in which operations are applied is:
        - Translation
        - Rotation
        - Scaling

        Parameters
        ----------
        dx : float, optional
            The distance to translate this path in the X-direction.
        dy : float, optional
            The distance to translate this path in the Y-direction.
        angle : float, optional
            The angle to rotate this path by.
        scale : float, optional
            The scaling-factor to apply to this path (uniformly).
        angle_type : AngleType, default=AngleType.DEGREES
            The type/format of the given `angle`.

        Returns
        -------
        Path2
            The updated path (not a copy) after the transformations
            have been applied (so that operations can be chained).

        Raises
        ------
        ValueError
            If the given `scale` factor is less than or equal to
            ``0.0``.

        See Also
        --------
        translate, rotate, scale

        """
        m_comb: Matrix3 = Matrix3.new_identity()  # type: ignore
        if scale is not None:
            if scale <= 0.0:
                raise ValueError(
                    "Cannot scale by values less than or equal to 0"
                )
            m_comb *= Matrix3.new_scale(scale, scale)  # type: ignore
        if angle is not None:
            if angle_type == AngleType.DEGREES:
                r_ang = math.radians(angle)
            else:
                r_ang = angle
            m_comb *= Matrix3.new_rotate(r_ang)  # type: ignore
        if dx is not None or dy is not None:
            m_comb *= Matrix3.new_translate(dx or 0.0, dy or 0.0)  # type: ignore
        return self._apply_transform(m_comb)

    def contains(self, point: T_Point2) -> bool:
        """Whether or not this path contains the specified point.

        Parameters
        ----------
        point : Tuple[float, float] or Point2
            The point to check if this path contains.

        Returns
        -------
        bool
            Whether or not this path contains the specified `point`.

        """
        if isinstance(point, tuple):
            c_pt = Point2(*point)
        else:
            c_pt = point
        return self._contains_point(c_pt)

    def intersects(
        self,
        other: Union["Polygon2", "Path2", Line2, LineSegment2, Ray2],
    ) -> bool:
        """Whether or not this path intersects the other object given.

        Parameters
        ----------
        other : Union[Polygon2, Path2, Line2, LineSegment2, Ray2]
            The other object to check for intersection with.

        Returns
        -------
        bool
            Whether or not any part of this path intersects the other
            object given.

        """
        if isinstance(other, Polygon2):
            return self._intersects_polygon(other)
        elif isinstance(other, Path2):
            return self._intersects_path(other)
        return self._intersects_basic(other)

    def _intersects_polygon(self, shape: "Polygon2") -> bool:
        """Checks if this path intersects the given shape."""
        # - Check if any segments intersect edges
        for u_segment in self._segments:
            for v_edge in shape.edges:
                if u_segment.intersect(v_edge) is not None:
                    return True

        # - Check if any point on this path is contained in the shape
        return shape._contains_point(self._points[0])

    def _intersects_path(self, path: "Path2") -> bool:
        """Checks if this path intersects the other path given."""
        for u_segment in self._segments:
            for v_segment in path._segments:
                if u_segment.intersect(v_segment) is not None:
                    return True
        return False

    def _intersects_basic(
        self,
        other: Union[Line2, LineSegment2, Ray2],
    ) -> bool:
        """Checks if this path intersects another basic geometric item."""
        for u_segment in self._segments:
            if u_segment.intersect(other) is not None:
                return True
        return False

    def _contains_point(self, point: Point2) -> bool:
        """Checks if the given point is contained in this path."""
        for segment in self._segments:
            if segment.distance(point) == 0.0:
                return True
        return False

    def _append_point(self, next_point: Point2) -> "Path2":
        """Appends and returns the updated path with the point added."""
        self._segments.append(LineSegment2(self._points[-1], next_point))
        self._vectors.append(Vector2(next_point))
        self._points.append(next_point)
        return self

    def _apply_transform(self, m: Matrix3) -> "Path2":
        """Updates and returns the path with the given transformation."""
        for i in range(len(self._vectors)):
            self._vectors[i] *= m
        self._points = [Point2(v.x, v.y) for v in self._vectors]
        self._segments = self._construct_segments(self._points)
        self._steps = None
        return self

    @classmethod
    def _construct_segments(cls, points: List[Point2]) -> List[LineSegment2]:
        """Constructs the line segments between the path's points."""
        ret: List[LineSegment2] = []
        for i in range(1, len(points)):
            p_prev = points[i - 1]
            p_curr = points[i]
            ret.append(LineSegment2(p_prev, p_curr))
        return ret


class Polygon2:
    """
    2D Polygon shape.
    """

    __slots__ = (
        "_centroid",
        "_points",
        "_edges",
        "_vectors",
    )

    def __init__(self, *points: T_Point2) -> None:
        if len(points) < 3:
            raise ValueError("Polygons must have at least 3 points")

        self._points: List[Point2] = []
        for point in points:
            if isinstance(point, tuple):
                new_pt = Point2(*point)
            else:
                new_pt = point
            self._points.append(new_pt)
        self._vectors = [Vector2(p.x, p.y) for p in self._points]
        self._edges: List[LineSegment2] = self._construct_edges(self._points)
        self._centroid: Optional[Point2] = None

    @property
    def centroid(self) -> Point2:
        """Point2: The centroid point of this shape."""
        if self._centroid is None:
            c_x = 0.0
            c_y = 0.0
            n_p = len(self._points)
            for point in self._points:
                c_x += point.x
                c_y += point.y
            self._centroid = Point2(c_x / n_p, c_y / n_p)
        return self._centroid

    @property
    def points(self) -> List[Point2]:
        """List[Point2]: The points defining the edges of this shape."""
        return self._points

    @property
    def edges(self) -> List[LineSegment2]:
        """List[LineSegment2]: The edges of this shape."""
        return self._edges

    @property
    def vectors(self) -> List[Vector2]:
        """List[Vector2]: The vectors extending to this shape's points."""
        return self._vectors

    @property
    def perimeter(self) -> float:
        """float: The length of this shape's perimeter."""
        return sum(x.length for x in self._edges)

    def translate(self, dx: float, dy: float) -> "Polygon2":
        """Translates this shape (and updates it) by the amounts given.

        Parameters
        ----------
        dx : float
            The distance to translate this shape in the X-direction.
        dy : float
            The distance to translate this shape in the Y-direction.

        Returns
        -------
        Polygon2
            The updated shape (not a copy) after the translation
            operation (so that operations can be chained).

        See Also
        --------
        transform

        """
        m_trans: Matrix3 = Matrix3.new_translate(dx, dy)  # type: ignore
        return self._apply_transform(m_trans)

    def rotate(
        self,
        angle: float,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> "Polygon2":
        """Rotates this shape (and updates it) by the amount given.

        Parameters
        ----------
        angle : float
            The angle to rotate this shape by.
        angle_type : AngleType, default=AngleType.DEGREES
            The type/format of the given `angle`.

        Returns
        -------
        Polygon2
            The updated shape (not a copy) after the rotation operation
            (so that operations can be chained).

        See Also
        --------
        transform

        """
        if angle_type == AngleType.DEGREES:
            r_ang = math.radians(angle)
        else:
            r_ang = angle
        m_rot: Matrix3 = Matrix3.new_rotate(r_ang)  # type: ignore
        return self._apply_transform(m_rot)

    def scale(self, factor: float) -> "Polygon2":
        """Scales this shape (and updates it) uniformly.

        Parameters
        ----------
        factor : float
            The scaling-factor to use.

        Returns
        -------
        Polygon2
            The updated shape (not a copy) after the scaling operation
            (so that operations can be chained).

        Raises
        ------
        ValueError
            If the given `factor` for scaling is invalid (less than or
            equal to ``0.0``).

        See Also
        --------
        transform

        """
        if factor <= 0.0:
            raise ValueError("Cannot scale by values less than or equal to 0")
        m_scale: Matrix3 = Matrix3.new_scale(factor, factor)  # type: ignore
        return self._apply_transform(m_scale)

    def transform(
        self,
        *,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        angle: Optional[float] = None,
        scale: Optional[float] = None,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> "Polygon2":
        """Applies the given transformations to this shape (and updates
        it).

        The order in which operations are applied is:
        - Translation
        - Rotation
        - Scaling

        Parameters
        ----------
        dx : float, optional
            The distance to translate this shape in the X-direction.
        dy : float, optional
            The distance to translate this shape in the Y-direction.
        angle : float, optional
            The angle to rotate this shape by.
        scale : float, optional
            The scaling-factor to apply to this shape (uniformly).
        angle_type : AngleType, default=AngleType.DEGREES
            The type/format of the given `angle`.

        Returns
        -------
        Polygon2
            The updated shape (not a copy) after the transformations
            have been applied (so that operations can be chained).

        Raises
        ------
        ValueError
            If the given `scale` factor is less than or equal to
            ``0.0``.

        See Also
        --------
        translate, rotate, scale

        """
        m_comb: Matrix3 = Matrix3.new_identity()  # type: ignore
        if scale is not None:
            if scale <= 0.0:
                raise ValueError(
                    "Cannot scale by values less than or equal to 0"
                )
            m_comb *= Matrix3.new_scale(scale, scale)  # type: ignore
        if angle is not None:
            if angle_type == AngleType.DEGREES:
                r_ang = math.radians(angle)
            else:
                r_ang = angle
            m_comb *= Matrix3.new_rotate(r_ang)  # type: ignore
        if dx is not None or dy is not None:
            m_comb *= Matrix3.new_translate(dx or 0.0, dy or 0.0)  # type: ignore
        return self._apply_transform(m_comb)

    def contains(self, point: T_Point2) -> bool:
        """Whether or not this shape contains the given point.

        Parameters
        ----------
        point : Tuple[float, float] or Point2
            The point to check to see if it lies inside this shape.

        Returns
        -------
        bool
            Whether or not the given `point` lies inside this shape.

        """
        if isinstance(point, tuple):
            c_pt = Point2(*point)
        else:
            c_pt = point
        return self._contains_point(c_pt)

    def intersects(
        self,
        other: Union["Polygon2", Path2, Line2, LineSegment2, Ray2],
    ) -> bool:
        """Whether or not this shape and the other object intersect.

        Parameters
        ----------
        other : Union[Polygon2, Path2, Line2, LineSegment2, Ray2]
            The other shape or path to check for intersection with.

        Returns
        -------
        bool
            Whether or not there's any intersection between the two
            objects.

        """
        if isinstance(other, Polygon2):
            return self._intersects_polygon(other)
        elif isinstance(other, Path2):
            return self._intersects_path(other)
        return self._intersects_basic(other)

    def _intersects_polygon(self, shape: "Polygon2") -> bool:
        """Checks for intersection between this and another polygon."""
        # - First check edges for intersection
        for u_edge in self._edges:
            for v_edge in shape.edges:
                if u_edge.intersect(v_edge):
                    return True

        # - Check if a point from the shape is in this shape
        if self._contains_point(shape.points[0]):
            return True

        # - Check if a point from this shape is in the other shape
        if shape._contains_point(self._points[0]):
            return True

        return False

    def _intersects_path(self, path: Path2) -> bool:
        """Checks for intersection between this and a given path."""
        # - Check all edges and path segments
        for u_edge in self._edges:
            for v_segment in path.segments:
                if u_edge.intersect(v_segment) is not None:
                    return True

        # - Check if any path points are inside this shape
        return self._contains_point(path._points[0])

    def _intersects_basic(
        self,
        other: Union[Line2, LineSegment2, Ray2],
    ) -> bool:
        """Checks if a basic geometric line/segment/ray intersects this."""
        # - Check for edge intersections
        for u_edge in self._edges:
            if u_edge.intersect(other) is not None:
                return True

        # - If line segment, check if contained
        if isinstance(other, LineSegment2):
            return self._contains_point(other.p1)

        return False

    def _contains_point(self, point: Point2) -> bool:
        """Checks if the given point lies inside this polygon."""
        ray = Ray2(point, Vector2(1.0, 0.0))
        i_count = 0
        for edge in self._edges:
            if ray.intersect(edge) is not None:
                i_count += 1
        return i_count % 2 == 1

    def _apply_transform(self, m: Matrix3) -> "Polygon2":
        """Applies the given transformation matrix to this shape."""
        for i in range(len(self._vectors)):
            self._vectors[i] *= m
        self._points = [Point2(v.x, v.y) for v in self._vectors]
        self._edges = self._construct_edges(self._points)
        self._centroid = None
        return self

    @classmethod
    def _construct_edges(cls, points: List[Point2]) -> List[LineSegment2]:
        """Constructs the edges associated with the given end points."""
        edges = []
        p_curr = None
        for i in range(1, len(points)):
            p_prev = points[i - 1]
            p_curr = points[i]
            edges.append(LineSegment2(p_prev, p_curr))
        if p_curr is not None:
            edges.append(LineSegment2(p_curr, points[0]))
        return edges

    @classmethod
    def at_location(cls, origin: T_Point2, *points: T_Point2) -> "Polygon2":
        """Creates a shape with the given points at the given origin.

        Parameters
        ----------
        origin : Tuple[float, float] or Point2
            The origin location to place the new shape at.
        *points : Tuple[float, float] or Point2
            The points (relative to the given `origin`) to use to define
            the new shape.

        Returns
        -------
        Polygon2
            The newly-created shape at the given `origin` and defined by
            the specified `points`.

        """
        new_shape = cls(*points)
        if isinstance(origin, tuple):
            dx, dy = origin
        else:
            dx = origin.x
            dy = origin.y
        return new_shape.translate(dx, dy)
