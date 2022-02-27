import math
from typing import Optional, Tuple

from squad.config import config
from squad.constants import PI, AngleType, Leg

from .core import coord_rotate_xyz


def _compute_hip_angle(
    leg: Leg,
    y: float,
    z: float,
    w_body: float,
    l_hip: float,
) -> float:
    """Computes the hip angle (alpha)."""
    if leg in (Leg.FL, Leg.BL):
        y_t = y + (w_body / 2.0)
    else:
        y_t = y - (w_body / 2.0)

    r_alpha = (
        math.atan2(y_t, z)
        + math.acos(l_hip / (((z ** 2) + (y_t ** 2)) ** 0.5))
    ) + (PI / 2.0)
    return r_alpha


def _compute_femur_angle(
    leg: Leg,
    x: float,
    y: float,
    z: float,
    l_body: float,
    w_body: float,
    l_femur: float,
    l_leg: float,
) -> float:
    """Computes the femur angle (beta)."""
    if leg == Leg.FL:
        x_t = x - (l_body / 2.0)
        y_t = y - (w_body / 2.0)
    elif leg == Leg.FR:
        x_t = x - (l_body / 2.0)
        y_t = y + (w_body / 2.0)
    elif leg == Leg.BL:
        x_t = x + (l_body / 2.0)
        y_t = y - (w_body / 2.0)
    elif leg == Leg.BR:
        x_t = x + (l_body / 2.0)
        y_t = y + (w_body / 2.0)
    else:
        raise ValueError(f"Invalid value for leg: {leg}")

    t_yz2 = (y_t ** 2) + (z ** 2)
    t_xyz2 = (x_t ** 2) + t_yz2

    r_beta = math.acos(
        ((l_femur ** 2) + t_xyz2 - (l_leg ** 2)) / (2.0 * l_femur * t_xyz2)
    ) + math.atan2(x_t, (t_yz2 ** 0.5))
    return r_beta


def _compute_leg_angle(
    leg: Leg,
    x: float,
    y: float,
    z: float,
    l_body: float,
    w_body: float,
    l_femur: float,
    l_leg: float,
) -> float:
    """Computes the leg angle (gamma)."""
    if leg == Leg.FL:
        x_t = x - (l_body / 2.0)
        y_t = y - (w_body / 2.0)
    elif leg == Leg.FR:
        x_t = x - (l_body / 2.0)
        y_t = y + (w_body / 2.0)
    elif leg == Leg.BL:
        x_t = x + (l_body / 2.0)
        y_t = y - (w_body / 2.0)
    elif leg == Leg.BR:
        x_t = x + (l_body / 2.0)
        y_t = y + (w_body / 2.0)
    else:
        raise ValueError(f"Invalid value for leg: {leg}")

    t_xyz2 = (x_t ** 2) + (y_t ** 2) + (z ** 2)

    r_gamma = (
        math.acos(
            ((l_leg ** 2) + (l_femur ** 2) - t_xyz2) / (2.0 * l_leg * l_femur)
        )
        - PI
    )
    return r_gamma


def compute_leg_angles(
    leg: Leg,
    x: float,
    y: float,
    z: float,
    angle_type: AngleType = AngleType.DEGREES,
    *,
    l_body: Optional[float] = None,
    w_body: Optional[float] = None,
    l_hip: Optional[float] = None,
    l_femur: Optional[float] = None,
    l_leg: Optional[float] = None,
    x_m: Optional[float] = None,
    y_m: Optional[float] = None,
    z_m: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Computes the hip ($\\alpha$), femur ($\\beta$), and leg
    ($\\gamma$) angles for the desired position.

    Parameters
    ----------
    leg : Leg
        The leg (of the foot) for which to compute the angles.
    x : float
        The X coordinate to compute the angles for.
    y : float
        The Y coordinate to compute the angles for.
    z : float
        The Z coordinate to compute the angles for.
    angle_type : AngleType, default=AngleType.DEGREES
        The units of the angle to return (degrees or radians).
    l_body : float, optional
        The total length of the robot body to use (if not given the
        value set in the :obj:`config` will be used).
    w_body : float, optional
        The total width of the robot body to use (if not given the value
        set in the :obj:`config` will be used).
    l_hip : float, optional
        The length of the hip assembly (from hip rotation axis center to
        femur rotation axis center).
    l_femur : float, optional
        The length of the femur assembly (from femur rotation axis
        center to leg rotation axis center).
    l_leg : float, optional
        The length of the leg assembly (from leg rotation axis center to
        foot).
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
    Tuple[float, float, float]
        The angles computed for the servos of the specified `leg` based
        on the given position and parameters.

    """
    # - Handle inputs
    l_b = l_body if l_body is not None else config.l_body
    w_b = w_body if w_body is not None else config.w_body

    l_h = l_hip if l_hip is not None else config.l_hip
    l_f = l_femur if l_femur is not None else config.l_femur
    l_l = l_leg if l_leg is not None else config.l_leg

    dm_x = x_m if x_m is not None else config.cm_dx
    dm_y = y_m if y_m is not None else config.cm_dy
    dm_z = z_m if z_m is not None else config.cm_dz

    # - Compute common variables
    x_0 = x - dm_x
    y_0 = y - dm_y
    z_0 = z - dm_z

    # - Hip (alpha)
    r_alpha = _compute_hip_angle(leg, y_0, z_0, w_b, l_h)

    # - Coordinate transform
    x_1, y_1, z_1 = coord_rotate_xyz(
        x_0,
        y_0,
        z_0,
        x_rotation=-r_alpha,
        angle_type=AngleType.RADIANS,
    )

    # - Femur (beta)
    r_beta = _compute_femur_angle(leg, x_1, y_1, z_1, l_b, w_b, l_f, l_l)

    # - Leg (gamma)
    r_gamma = _compute_leg_angle(leg, x_1, y_1, z_1, l_b, w_b, l_f, l_l)

    if angle_type == AngleType.DEGREES:
        r_alpha = math.degrees(r_alpha)
        r_beta = math.degrees(r_beta)
        r_gamma = math.degrees(r_gamma)
    return (r_alpha, r_beta, r_gamma)


def _compute_foot_x(
    leg: Leg,
    alpha: float,
    beta: float,
    gamma: float,
    l_body: float,
    w_body: float,
    l_hip: float,
    l_femur: float,
    l_leg: float,
) -> float:
    """Computes the foot X-coordinate for the given angles."""
    return 0.0


def _compute_foot_y(
    leg: Leg,
    alpha: float,
    beta: float,
    gamma: float,
    l_body: float,
    w_body: float,
    l_hip: float,
    l_femur: float,
    l_leg: float,
) -> float:
    """Computes the foot Y-coordinate for the given angles."""
    return 0.0


def _compute_foot_z(
    leg: Leg,
    alpha: float,
    beta: float,
    gamma: float,
    l_body: float,
    w_body: float,
    l_hip: float,
    l_femur: float,
    l_leg: float,
) -> float:
    """Computes the foot Z-coordinate for the given angles."""
    return 0.0


def compute_foot_position(
    leg: Leg,
    alpha: float,
    beta: float,
    gamma: float,
    angle_type: AngleType = AngleType.DEGREES,
    *,
    l_body: Optional[float] = None,
    w_body: Optional[float] = None,
    l_hip: Optional[float] = None,
    l_femur: Optional[float] = None,
    l_leg: Optional[float] = None,
    x_m: Optional[float] = None,
    y_m: Optional[float] = None,
    z_m: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Computes the foot position from the given leg angles.

    Parameters
    ----------
    leg : Leg
        The leg (of the foot) for which to compute the angles.
    alpha : float
        The hip angle ($\\alpha$) of the leg.
    beta : float
        The femur angle ($\\beta$) of the leg.
    gamma : float
        The leg angle ($\\gamma$) of the leg.
    angle_type : AngleType, default=AngleType.DEGREES
        The units of the angle to return (degrees or radians).
    l_body : float, optional
        The total length of the robot body to use (if not given the
        value set in the :obj:`config` will be used).
    w_body : float, optional
        The total width of the robot body to use (if not given the value
        set in the :obj:`config` will be used).
    l_hip : float, optional
        The length of the hip assembly (from hip rotation axis center to
        femur rotation axis center).
    l_femur : float, optional
        The length of the femur assembly (from femur rotation axis
        center to leg rotation axis center).
    l_leg : float, optional
        The length of the leg assembly (from leg rotation axis center to
        foot).
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
    Tuple[float, float, float]
        The X, Y, and Z-coordinates computed for the servos of the
        specified `leg` based on the given angles and parameters.

    """
    # - Handle inputs
    if angle_type == AngleType.DEGREES:
        alpha = math.radians(alpha)
        beta = math.radians(beta)
        gamma = math.radians(gamma)

    l_b = l_body if l_body is not None else config.l_body
    w_b = w_body if w_body is not None else config.w_body

    l_h = l_hip if l_hip is not None else config.l_hip
    l_f = l_femur if l_femur is not None else config.l_femur
    l_l = l_leg if l_leg is not None else config.l_leg

    dm_x = x_m if x_m is not None else config.cm_dx
    dm_y = y_m if y_m is not None else config.cm_dy
    dm_z = z_m if z_m is not None else config.cm_dz

    # - Compute coordinates
    r_x = _compute_foot_x(leg, alpha, beta, gamma, l_b, w_b, l_h, l_f, l_l)
    r_y = _compute_foot_y(leg, alpha, beta, gamma, l_b, w_b, l_h, l_f, l_l)
    r_z = _compute_foot_z(leg, alpha, beta, gamma, l_b, w_b, l_h, l_f, l_l)

    return (r_x + dm_x, r_y + dm_y, r_z + dm_z)
