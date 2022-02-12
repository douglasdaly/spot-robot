from math import cos, radians, sin
from typing import Optional, Tuple

import numpy as np

from squad.constants import AngleType


def _get_sin_cos_theta(theta: float) -> Tuple[float, float]:
    """Gets the :obj:`sin` and :obj:`cos` of the given `theta` (in
    radians).
    """
    return sin(theta), cos(theta)


def _x_rotation_mat(theta: float) -> np.ndarray:
    """Gets the X-axis rotation matrix for the given `theta` (in
    radians).
    """
    t_sin, t_cos = _get_sin_cos_theta(theta)

    r_x = np.identity(3)
    r_x[1, 1] = t_cos
    r_x[1, 2] = -t_sin
    r_x[2, 1] = t_sin
    r_x[2, 2] = t_cos

    return r_x


def _y_rotation_mat(theta: float) -> np.ndarray:
    """Gets the Y-axis rotation matrix for the given `theta` (in
    radians).
    """
    t_sin, t_cos = _get_sin_cos_theta(theta)

    r_y = np.identity(3)
    r_y[0, 0] = t_cos
    r_y[0, 2] = t_sin
    r_y[2, 0] = -t_sin
    r_y[2, 2] = t_cos

    return r_y


def _z_rotation_mat(theta: float) -> np.ndarray:
    """Gets the Z-axis rotation matrix for the given `theta` (in
    radians).
    """
    t_sin, t_cos = _get_sin_cos_theta(theta)

    r_z = np.identity(3)
    r_z[0, 0] = t_cos
    r_z[0, 1] = -t_sin
    r_z[1, 0] = t_sin
    r_z[1, 1] = t_cos

    return r_z


def rotation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Gets the rotation transformation matrix for the given angles (in
    radians).
    """
    if x:
        r_x = _x_rotation_mat(x)
    else:
        r_x = np.identity(3)

    if y:
        r_y = _y_rotation_mat(y)
    else:
        r_y = np.identity(3)

    if z:
        r_z = _z_rotation_mat(z)
    else:
        r_z = np.identity(3)

    r_xyz = r_z @ r_y @ r_x
    return r_xyz


def coord_rotate_xyz(
    x: float,
    y: float,
    z: float,
    x_rotation: Optional[float] = None,
    y_rotation: Optional[float] = None,
    z_rotation: Optional[float] = None,
    *,
    angle_format: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Apply a rotation transformation for the given position.

    Parameters
    ----------
    x : float
        The X-coordinate to apply the rotation(s) to.
    y : float
        The Y-coordinate to apply the rotation(s) to.
    z : float
        The Z-coordinate to apply the rotation(s) to.
    x_rotation : float, optional
        The rotation about the X-axis to apply (in radians, if any).
    y_rotation : float, optional
        The rotation about the Y-axis to apply (in radians, if any).
    z_rotation : float, optional
        The rotation about the Z-axis to apply (in radians, if any).
    angle_format : AngleType, default=AngleType.DEGREES
        The format of the given rotation angle inputs
        (:obj:`AngleType.DEGREES`, default) or
        (:obj:`AngleType.RADIANS`).

    Returns
    -------
    Tuple[float, float, float]
        The new coordinates, as a 3-element :obj:`tuple` of (x, y, z).

    """
    v0 = np.array((x, y, z))
    v1 = coord_rotate(
        v0,
        x_rotation=x_rotation,
        y_rotation=y_rotation,
        z_rotation=z_rotation,
        angle_format=angle_format,
    )
    return tuple(x for x in v1)


def coord_rotate(
    v: np.ndarray,
    x_rotation: Optional[float] = None,
    y_rotation: Optional[float] = None,
    z_rotation: Optional[float] = None,
    *,
    angle_format: AngleType = AngleType.DEGREES,
) -> np.ndarray:
    """Apply a rotation transformation for the given vector.

    Parameters
    ----------
    v : np.ndarray
        A 3D vector to rotate based on the given rotation angles.
    x_rotation : float, optional
        The rotation about the X-axis to apply (in radians, if any).
    y_rotation : float, optional
        The rotation about the Y-axis to apply (in radians, if any).
    z_rotation : float, optional
        The rotation about the Z-axis to apply (in radians, if any).
    angle_format : AngleType, default=AngleType.DEGREES
        The format of the given rotation angle inputs
        (:obj:`AngleType.DEGREES`, default) or
        (:obj:`AngleType.RADIANS`).

    Returns
    -------
    np.ndarray
        The new coordinates, as a 3-element vector, of (x, y, z).

    """
    if angle_format == AngleType.DEGREES:
        r_xyz = rotation_matrix(
            radians(x_rotation or 0.0),
            radians(y_rotation or 0.0),
            radians(z_rotation or 0.0),
        )
    else:
        r_xyz = rotation_matrix(
            x_rotation or 0.0,
            y_rotation or 0.0,
            z_rotation or 0.0,
        )
    return r_xyz @ v
