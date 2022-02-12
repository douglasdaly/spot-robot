import math
from typing import Optional, Tuple

import numpy as np

from squad.config import config
from squad.constants import HALF_PI, Leg


def get_rotation_x(roll: float) -> np.ndarray:
    """Gets the X-rotation matrix ($R_x$) based on the body's `roll`.

    Parameters
    ----------
    roll : float
        The roll angle ($\\omega$) of the main robot body to get the
        X-rotation matrix for.

    Returns
    -------
    np.ndarray
        The 4x4 $R_x$ matrix to use.

    """
    r_x = np.identity(4)
    r_x[1, 1] = math.cos(roll)
    r_x[1, 2] = -math.sin(roll)
    r_x[2, 1] = math.sin(roll)
    r_x[2, 2] = math.cos(roll)
    return r_x


def get_rotation_y(pitch: float) -> np.ndarray:
    """Gets the Y-rotation matrix ($R_y$) based on the body's `pitch`.

    Parameters
    ----------
    pitch : float
        The pitch angle ($\\psi$) of the main robot body to get the
        Y-rotation matrix for.

    Returns
    -------
    np.ndarray
        The 4x4 $R_y$ matrix to use.

    """
    r_y = np.identity(4)
    r_y[0, 0] = math.cos(pitch)
    r_y[0, 2] = math.sin(pitch)
    r_y[2, 0] = -math.sin(pitch)
    r_y[2, 2] = math.cos(pitch)
    return r_y


def get_rotation_z(yaw: float) -> np.ndarray:
    """Gets the Z-rotation matrix ($R_z$) based on the body's `yaw`.

    Parameters
    ----------
    yaw : float
        The yaw angle ($\\phi$) of the main robot body to get the
        Z-rotation matrix for.

    Returns
    -------
    np.ndarray
        The 4x4 $R_z$ matrix to use.

    """
    r_z = np.identity(4)
    r_z[0, 0] = math.cos(yaw)
    r_z[0, 1] = -math.sin(yaw)
    r_z[1, 0] = math.sin(yaw)
    r_z[1, 1] = math.cos(yaw)
    return r_z


def get_rotation_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Gets the combined XYZ-rotation matrix ($R_{xyz}$) from the body.

    Parameters
    ----------
    roll : float
        The roll angle ($\\omega$) of the main robot body to get the
        XYZ-rotation matrix for.
    pitch : float
        The pitch angle ($\\psi$) of the main robot body to get the
        XYZ-rotation matrix for.
    yaw : float
        The yaw angle ($\\phi$) of the main robot body to get the
        XYZ-rotation matrix for.

    Returns
    -------
    np.ndarray
        The 4x4 $R_{xyz}$ matrix to use.

    """
    r_x = get_rotation_x(roll)
    r_y = get_rotation_y(pitch)
    r_z = get_rotation_z(yaw)

    r_xyz = np.matmul(r_x, r_y)
    np.matmul(r_xyz, r_z, out=r_xyz)
    return r_xyz


def apply_rotation(
    pos: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Applies a rotation transformation to the given `pos` vector.

    Parameters
    ----------
    pos : np.ndarray
        A 3-element array representing (x, y, z) coordinates to apply
        the transformation to.
    roll : float
        The roll angle ($\\omega$) of the rotation to apply.
    pitch : float
        The pitch angle ($\\psi$) of the rotation to apply.
    yaw : float
        The yaw angle ($\\phi$) of the rotation to apply.

    Returns
    -------
    np.ndarray
        The transformed coordinates as an array representing (x, y, z).

    """
    pos_f = np.identity(4)
    pos_f[0:3, 3] = pos
    rot_t = get_rotation_xyz(roll, pitch, yaw)
    return np.outer(rot_t, pos_f)[:3, 0]


def get_cm_transform(
    roll: float,
    pitch: float,
    yaw: float,
    *,
    x_m: Optional[float] = None,
    y_m: Optional[float] = None,
    z_m: Optional[float] = None,
) -> np.ndarray:
    """Gets the main body transformation matrix ($T_m$) to use.

    Parameters
    ----------
    roll : float
        The roll angle ($\\omega$) of the main robot body to get the
        XYZ-rotation matrix for.
    pitch : float
        The pitch angle ($\\psi$) of the main robot body to get the
        XYZ-rotation matrix for.
    yaw : float
        The yaw angle ($\\phi$) of the main robot body to get the
        XYZ-rotation matrix for.
    x_m : float, optional
        The X-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    y_m : float, optioanl
        The Y-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    z_m : float, optional
        The Z-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).

    Returns
    -------
    np.ndarray
        The 4x4 $T_m$ matrix to use.

    """
    x = x_m if x_m is not None else config.cm_dx
    y = y_m if y_m is not None else config.cm_dy
    z = z_m if z_m is not None else config.cm_dz

    return apply_rotation(np.array([x, y, z]), roll, pitch, yaw)


def _compute_leg_transform(
    leg: Leg,
    t_m: np.ndarray,
    body_l2: float,
    body_w2: float,
) -> np.ndarray:
    """Helper function to do leg transform matrix calculation."""
    # - Get leg transform
    t_leg = np.identity(4)
    k_pi = HALF_PI if leg in (Leg.FR, Leg.BR) else -HALF_PI

    t_leg[0, 0] = math.cos(k_pi)
    t_leg[0, 2] = math.sin(k_pi)
    t_leg[2, 0] = -math.sin(k_pi)
    t_leg[2, 2] = math.cos(k_pi)

    if leg == Leg.FL:
        t_leg[0, 3] = body_l2
        t_leg[2, 3] = -body_w2
    elif leg == Leg.FR:
        t_leg[0, 3] = body_l2
        t_leg[2, 3] = body_w2
    elif leg == Leg.BL:
        t_leg[0, 3] = -body_l2
        t_leg[2, 3] = -body_w2
    elif leg == Leg.BR:
        t_leg[0, 3] = -body_l2
        t_leg[2, 3] = body_w2
    else:
        raise ValueError(f"Invalid leg specified: {leg}")

    # - Compute leg transformation matrix
    np.matmul(t_m, t_leg, out=t_leg)
    return t_leg


def get_leg_transform(
    roll: float,
    pitch: float,
    yaw: float,
    leg: Leg,
    *,
    x_m: Optional[float] = None,
    y_m: Optional[float] = None,
    z_m: Optional[float] = None,
    l_body: Optional[float] = None,
    w_body: Optional[float] = None,
    t_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gets a single leg's transformation matrix ($T_{leg}$).

    Parameters
    ----------
    roll : float
        The roll angle ($\\omega$) of the main robot body to get the
        XYZ-rotation matrix for.
    pitch : float
        The pitch angle ($\\psi$) of the main robot body to get the
        XYZ-rotation matrix for.
    yaw : float
        The yaw angle ($\\phi$) of the main robot body to get the
        XYZ-rotation matrix for.
    leg : Leg
        The leg for which to get the transformation matrix.
    x_m : float, optional
        The X-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    y_m : float, optioanl
        The Y-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    z_m : float, optional
        The Z-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    l_body : float, optional
        The total length of the robot body to use (if not given the
        value set in the :obj:`config` will be used).
    w_body : float, optional
        The total width of the robot body to use (if not given the value
        set in the :obj:`config` will be used).
    t_m : np.ndarray, optional
        The main body transformation matrix to use (if not given it will
        be computed).

    Returns
    -------
    np.ndarray
        The leg transformation matrix, $T_{leg}$, requested.

    """
    # - Get parameters to use
    if t_m is None:
        cm_x = x_m if x_m is not None else config.cm_dx
        cm_y = y_m if y_m is not None else config.cm_dy
        cm_z = z_m if z_m is not None else config.cm_dz
        t_cm = get_cm_transform(
            roll,
            pitch,
            yaw,
            x_m=cm_x,
            y_m=cm_y,
            z_m=cm_z,
        )
    else:
        t_cm = t_m

    body_l2 = (l_body if l_body is not None else config.l_body) / 2.0
    body_w2 = (w_body if w_body is not None else config.w_body) / 2.0

    return _compute_leg_transform(leg, t_cm, body_l2, body_w2)


def get_leg_transforms(
    roll: float,
    pitch: float,
    yaw: float,
    *legs: Leg,
    x_m: Optional[float] = None,
    y_m: Optional[float] = None,
    z_m: Optional[float] = None,
    l_body: Optional[float] = None,
    w_body: Optional[float] = None,
    t_m: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """Gets multiple leg transformation matrices ($T_{leg}$).

    Parameters
    ----------
    roll : float
        The roll angle ($\\omega$) of the main robot body to get the
        XYZ-rotation matrix for.
    pitch : float
        The pitch angle ($\\psi$) of the main robot body to get the
        XYZ-rotation matrix for.
    yaw : float
        The yaw angle ($\\phi$) of the main robot body to get the
        XYZ-rotation matrix for.
    legs: Leg
        The legs for which to get the transformation matrices.
    x_m : float, optional
        The X-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    y_m : float, optioanl
        The Y-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    z_m : float, optional
        The Z-coordinate of the main body's center-of-mass (if not given
        the value set in the :obj:`config` will be used).
    l_body : float, optional
        The total length of the robot body to use (if not given the
        value set in the :obj:`config` will be used).
    w_body : float, optional
        The total width of the robot body to use (if not given the value
        set in the :obj:`config` will be used).
    t_m : np.ndarray, optional
        The main body transformation matrix to use (if not given it will
        be computed).

    Returns
    -------
    Tuple[np.ndarray, ...]
        A tuple of the leg transformation matrices ($T_{leg}) in the
        same order as the `legs` were given.

    """
    # - Get parameters to use
    if t_m is None:
        cm_x = x_m if x_m is not None else config.cm_dx
        cm_y = y_m if y_m is not None else config.cm_dy
        cm_z = z_m if z_m is not None else config.cm_dz
        t_cm = get_cm_transform(
            roll,
            pitch,
            yaw,
            x_m=cm_x,
            y_m=cm_y,
            z_m=cm_z,
        )
    else:
        t_cm = t_m

    body_l2 = (l_body if l_body is not None else config.l_body) / 2.0
    body_w2 = (w_body if w_body is not None else config.w_body) / 2.0

    t_legs = tuple(
        _compute_leg_transform(x, t_cm, body_l2, body_w2) for x in legs
    )
    return t_legs
