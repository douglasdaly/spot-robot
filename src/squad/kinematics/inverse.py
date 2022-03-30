import math
from typing import Optional, Tuple

import numpy as np

from squad.constants import HALF_PI, PI, AngleType, Leg

from .base import BodyParameters
from .core import rotation_matrix
from .forward import hip_xyz


def _leg_thetas_hip_frame(
    x: float,
    y: float,
    z: float,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Gets the leg thetas from the foot position in the hip frame."""
    b_ps = body_params if body_params is not None else BodyParameters()

    # - Rotate coordinates back to hip frame
    r_x = y
    r_y = z
    r_z = -x

    # - Precompute common values for speed
    d_h = ((r_x ** 2) + (r_y ** 2) - (b_ps.l_hip ** 2)) ** 0.5
    d_a = (
        (r_x ** 2)
        + (r_y ** 2)
        + (r_z ** 2)
        - (b_ps.l_hip ** 2)
        - (b_ps.l_femur ** 2)
        - (b_ps.l_leg ** 2)
    ) / (2.0 * b_ps.l_femur * b_ps.l_leg)

    # - Compute thetas (+ adjustments for our reference frames)
    t_hip = -math.atan2(-r_y, r_x) - math.atan2(d_h, -b_ps.l_hip) + PI
    t_leg = math.atan2(-((1.0 - (d_a ** 2)) ** 0.5), d_a)
    t_femur = (
        math.atan2(r_z, d_h)
        - math.atan2(
            b_ps.l_leg * math.sin(t_leg),
            b_ps.l_femur + (b_ps.l_leg * math.cos(t_leg)),
        )
    ) - HALF_PI

    t_leg += HALF_PI

    # - Return
    t_ret = (t_hip, t_femur, t_leg)
    if angle_type == AngleType.DEGREES:
        return tuple(math.degrees(v) for v in t_ret)
    return t_ret


def body_thetas(
    leg: Leg,
    x_hip: float,
    y_hip: float,
    z_hip: float,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Gets the body orientation based on the given position relative to
    the specified leg's hip.

    Parameters
    ----------
    leg : Leg
        The leg the hip's position is given for.
    x_hip : float
        The X position of the hip's origin (relative to the body's
        origin).
    y_hip : float
        The Y position of the hip's origin (relative to the body's
        origin).
    z_hip : float
        The Z position of the hip's origin (relative to the body's
        origin).
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    Tuple[float, float, float]
        The body's orientation (Roll, Pitch, Yaw) based on the position
        of the given leg's hip.

    """
    b_ps = body_params if body_params is not None else BodyParameters()

    # - Get neutral positions and radial distance based off leg
    if leg > 2:
        n_x = -b_ps.l_body / 2.0
    else:
        n_x = b_ps.l_body / 2.0

    if leg % 2 == 0:
        n_y = -b_ps.w_body / 2.0
    else:
        n_y = b_ps.w_body / 2.0

    n_x -= b_ps.cm_dx
    n_y -= b_ps.cm_dy
    n_z = -b_ps.cm_dz

    # - Get unit vectors
    a = np.array([n_x, n_y, n_z])
    a /= np.linalg.norm(a)

    b = np.array(
        [(x_hip - b_ps.cm_dx), (y_hip - b_ps.cm_dy), (z_hip - b_ps.cm_dz)]
    )
    b /= np.linalg.norm(b)

    # - Compute rotation axis and angle
    v = np.cross(a, b)
    v /= np.linalg.norm(v)

    v_r = math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # - Compute rotation matrix
    rot_m = rotation_matrix(v[0] * v_r, v[1] * v_r, v[2] * v_r)

    # - Compute angles with Tait-Bryan convention
    d = (1.0 - (rot_m[2, 0] ** 2)) ** 0.5

    r_yaw = -math.asin(rot_m[1, 0] / d)
    r_pitch = -math.asin(-rot_m[2, 0])
    r_roll = math.asin(rot_m[2, 1] / d)

    r_rpy = (r_roll, r_pitch, r_yaw)
    if angle_type == AngleType.DEGREES:
        return tuple(math.degrees(r) for r in r_rpy)
    return r_rpy


def body_orn(
    leg: Leg,
    hip_pos: np.ndarray,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> np.ndarray:
    """Gets the body orientation based on the given position relative to
    the specified leg's hip.

    Parameters
    ----------
    leg : Leg
        The leg the hip's position is given for.
    hip_pos : np.ndarray
        The coordinates of the hip's origin for the given `leg` as a
        (X, Y, Z) array/vector.
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    np.ndarray
        The body's orientation (Roll, Pitch, Yaw) based on the position
        of the given leg's hip.

    """
    return np.array(
        body_thetas(
            leg,
            hip_pos[0],
            hip_pos[1],
            hip_pos[2],
            body_params=body_params,
            angle_type=angle_type,
        )
    )


def leg_thetas(
    leg: Leg,
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Computes the thetas for the given leg and body orientation.

    Parameters
    ----------
    leg : Leg
        The leg to compute the thetas for.
    x : float
        The X-coordinate of the foot to compute the thetas for.
    y : float
        The Y-coordinate of the foot to compute the thetas for.
    z : float
        The Z-coordinate of the foot to compute the thetas for.
    roll : float, default=0.0
        The body's Roll orientation to compute the thetas for.
    pitch : float, default=0.0
        The body's Pitch orientation to compute the thetas for.
    yaw : float, default=0.0
        The body's Yaw orientation to compute the thetas for.
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    Tuple[float, float, float]
        The thetas (Hip, Femur, Leg) for the given leg's foot position
        and body orientation.

    """
    # - Get hip coordinates to adjust for hip frame
    x_h, y_h, z_h = hip_xyz(
        leg,
        roll,
        pitch,
        yaw,
        body_params=body_params,
        angle_type=angle_type,
    )

    if leg % 2 == 0:
        y_f = y_h - y
    else:
        y_f = y - y_h

    r_h, r_f, r_l = _leg_thetas_hip_frame(
        x - x_h,
        y_f,
        z - z_h,
        body_params=body_params,
        angle_type=angle_type,
    )
    return r_h, r_f, r_l


def knee_angle_to_leg_servo(
    leg_theta: float,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> float:
    """Converts the given leg/knee angle to the corresponding servo
    angle.

    Parameters
    ----------
    leg_theta : float
        The leg/knee joint angle to convert to the corresponding servo
        angle.
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    float
        The corresponding servo angle based on the given `leg_angle`.

    """
    b_ps = body_params if body_params is not None else BodyParameters()
    if angle_type == AngleType.DEGREES:
        tl = math.radians(90.0 - leg_theta)
    else:
        tl = HALF_PI - leg_theta

    # - Pre-compute trig values for speed
    s_tl = math.sin(tl)
    c_tl = math.cos(tl)

    # - Compute angle
    h_adj = math.atan2(-b_ps.h_rod_femur, b_ps.l_rod_femur)

    beta2 = (
        (b_ps.l_rod_leg ** 2)
        + (b_ps.l_rod_femur ** 2)
        - (2.0 * b_ps.l_rod_leg * b_ps.l_rod_femur * c_tl)
    )
    beta = beta2 ** 0.5

    phi_opp = math.acos(
        np.sign(leg_theta)
        * (
            (beta2 - (b_ps.l_rod ** 2) - (b_ps.l_rod_arm ** 2))
            / (-2.0 * b_ps.l_rod * b_ps.l_rod_arm)
        )
    )

    phi_1 = math.asin((b_ps.l_rod_leg * s_tl) / beta)
    phi_2 = math.asin((b_ps.l_rod * math.sin(phi_opp)) / beta)

    ret = phi_1 + phi_2 - h_adj - HALF_PI
    if angle_type == AngleType.DEGREES:
        return math.degrees(ret)
    return ret
