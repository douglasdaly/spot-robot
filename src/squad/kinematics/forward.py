import math
from typing import Optional, Tuple

import numpy as np

from squad.constants import HALF_PI, AngleType, Leg

from .base import BodyParameters
from .core import coord_rotate_xyz


def _foot_xyz_hip_frame(
    hip_theta: float,
    femur_theta: float,
    leg_theta: float,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Gets the X, Y, and Z coordinates of the foot in the hip frame."""
    b_ps = body_params if body_params is not None else BodyParameters()
    if angle_type == AngleType.DEGREES:
        r_h = math.radians(hip_theta)
        r_f = math.radians(femur_theta + 90.0)
        r_l = math.radians(leg_theta - 90.0)
    else:
        r_h = hip_theta
        r_f = femur_theta + HALF_PI
        r_l = leg_theta - HALF_PI

    # - Precompute common ones for speed
    s_h = math.sin(r_h)
    c_h = math.cos(r_h)
    c_f = math.cos(r_f)
    c_fl = math.cos(r_f + r_l)

    # - Compute coordinates
    x_h = (
        (b_ps.l_leg * s_h * c_fl)
        + (b_ps.l_femur * s_h * c_f)
        + (b_ps.l_hip * c_h)
    )
    y_h = (
        -(b_ps.l_leg * c_h * c_fl)
        - (b_ps.l_femur * c_h * c_f)
        + (b_ps.l_hip * s_h)
    )
    z_h = -(b_ps.l_leg * math.sin(r_f + r_l)) - (b_ps.l_femur * math.sin(r_f))

    # - Return in the body's coordinate system (hence the ordering)
    return z_h, x_h, y_h


def hip_xyz(
    leg: Leg,
    roll: float,
    pitch: float,
    yaw: float,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Gets the current origin of the hip based on the body orientation.

    Parameters
    ----------
    leg : Leg
        The leg to compute the hip coordinate for.
    roll : float
        The roll of the main body to compute the hip coordinates for.
    pitch : float
        The pitch of the main body to compute the hip coordinates for.
    yaw : float
        The yaw of the main body to compute the hip coordinates for.
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    Tuple[float, float, float]
        The X, Y, and Z coordinates of the hip for the specified `leg`,
        based on the given angles, in the body's coordinate frame.

    """
    b_ps = body_params if body_params is not None else BodyParameters()
    if angle_type == AngleType.DEGREES:
        r_r = math.radians(roll)
        r_p = math.radians(-pitch)
        r_y = math.radians(-yaw)
    else:
        r_r = roll
        r_p = -pitch
        r_y = -yaw

    # - Modify for leg
    if leg > 2:
        d_x = -b_ps.l_body / 2.0
    else:
        d_x = b_ps.l_body / 2.0

    if leg % 2 == 0:
        d_y = -b_ps.w_body / 2.0
    else:
        d_y = b_ps.w_body / 2.0

    # - Adjust for orientation
    return coord_rotate_xyz(
        d_x + b_ps.cm_dx,
        d_y + b_ps.cm_dy,
        b_ps.cm_dz,
        r_r,
        r_p,
        r_y,
        angle_type=AngleType.RADIANS,
    )


def hip_pos(
    leg: Leg,
    orientation: np.ndarray,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> np.ndarray:
    """Gets the current origin of the hip based on the body orientation.

    Parameters
    ----------
    leg : Leg
        The leg to compute the hip coordinate for.
    orientation : np.ndarray
        An orientation vector of the form (roll, pitch, yaw) for the
        main body to compute the hip position for.
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    np.ndarray
        The position vector of the form (X, Y, Z) of the hip for the
        specified `leg`, based on the given `orientation`, in the body's
        coordinate frame.

    """
    return np.array(
        hip_xyz(
            leg,
            orientation[0],
            orientation[1],
            orientation[2],
            body_params=body_params,
            angle_type=angle_type,
        )
    )


def foot_xyz(
    leg: Leg,
    hip_theta: float,
    femur_theta: float,
    leg_theta: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> Tuple[float, float, float]:
    """Gets the X, Y, and Z coordinates of the foot for the given `leg`
    in the main/body frame.

    Parameters
    ----------
    leg : Leg
        The leg to compute the foot position for.
    hip_theta : float
        The rotation angle of the hip joint.
    femur_theta : float
        The rotation angle of the femur joint.
    leg_theta : float
        The rotation angle of the leg joint.
    roll : float, default=0.0
        The current roll of the main body (if any).
    pitch : float, default=0.0
        The current pitch of the main body (if any).
    yaw : float, default=0.0
        The current yaw of the main body (if any).
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    Tuple[float, float, float]
        The X, Y, and Z coordinates of the foot, for the given `leg` and
        angles, in the body's coordinate frame.

    """
    b_ps = body_params if body_params is not None else BodyParameters()
    if angle_type == AngleType.DEGREES:
        r_h = math.radians(hip_theta)
        r_f = math.radians(femur_theta)
        r_l = math.radians(leg_theta)
        r_r = math.radians(roll)
        r_p = math.radians(pitch)
        r_y = math.radians(yaw)
    else:
        r_h = hip_theta
        r_f = femur_theta
        r_l = leg_theta
        r_r = roll
        r_p = pitch
        r_y = yaw

    # - Get hip relative to body
    x_h, y_h, z_h = hip_xyz(
        leg,
        r_r,
        r_p,
        r_y,
        body_params=b_ps,
        angle_type=AngleType.RADIANS,
    )

    # - Get foot relative to hip
    x_f, y_f, z_f = _foot_xyz_hip_frame(
        r_h,
        r_f,
        r_l,
        body_params=b_ps,
        angle_type=AngleType.RADIANS,
    )

    # - Format and return appropriate result
    if leg % 2 == 0:
        y_f = -y_f
    return (x_h + x_f, y_h + y_f, z_h + z_f)


def foot_pos(
    leg: Leg,
    thetas: np.ndarray,
    orientation: Optional[np.ndarray] = None,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> np.ndarray:
    """Gets the position vector of the foot for the given `leg` in the
    main/body frame.

    Parameters
    ----------
    leg : Leg
        The leg to compute the position vector for.
    thetas : np.ndarray
        The rotation angles of the hip, femur, and leg joints.
    orientation : np.ndarray
        An orientation vector of the form (roll, pitch, yaw) for the
        main body to compute the foot position for.
    body_params : BodyParameters, optional
        The parameters describing the robot body (if not provided then
        the default values from the configuration are used).
    angle_type : AngleType, default=AngleType.DEGREES
        The type/units the `alpha`, `beta`, and `gamma` angles are given
        in, either ``DEGREES`` (default) or ``RADIANS``.

    Returns
    -------
    np.ndarray
        The position vector of X, Y, and Z coordinates of the foot, for
        the given `leg` and `thetas`, in the body's coordinate frame.

    """
    if orientation is None:
        orn = np.array([0.0, 0.0, 0.0])
    else:
        orn = orientation

    return np.array(
        foot_xyz(
            leg,
            thetas[0],
            thetas[1],
            thetas[2],
            roll=orn[0],
            pitch=orn[1],
            yaw=orn[2],
            body_params=body_params,
            angle_type=angle_type,
        )
    )


def leg_servo_to_knee_angle(
    servo_theta: float,
    body_params: Optional[BodyParameters] = None,
    *,
    angle_type: AngleType = AngleType.DEGREES,
) -> float:
    """Converts the given leg servo angle to the corresponding knee
    joint angle.

    Parameters
    ----------
    servo_theta : float
        The servo angle to convert to the corresponding knee joint
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
        The corresponding knee angle based on the given `servo_angle`.

    """
    b_ps = body_params if body_params is not None else BodyParameters()
    if angle_type == AngleType.DEGREES:
        ts = math.radians(servo_theta + 90.0)
    else:
        ts = servo_theta + HALF_PI

    # - Pre-compute trig values for speed
    s_ts = math.sin(ts)
    c_ts = math.cos(ts)

    # - Compute angle
    h_adj = math.atan2(-b_ps.h_rod_femur, b_ps.l_rod_femur)

    beta2 = (
        (b_ps.l_rod_arm ** 2)
        + (b_ps.l_rod_femur ** 2)
        - (2.0 * b_ps.l_rod_arm * b_ps.l_rod_femur * c_ts)
    )
    beta = beta2 ** 0.5

    phi_opp = math.acos(
        (beta2 - (b_ps.l_rod_leg ** 2) - (b_ps.l_rod ** 2))
        / (-2.0 * b_ps.l_rod_leg * b_ps.l_rod)
    )

    phi_1 = math.asin((b_ps.l_rod_arm * s_ts) / beta)
    phi_2 = math.asin((b_ps.l_rod * math.sin(phi_opp)) / beta)

    ret = HALF_PI - (phi_1 + phi_2 - h_adj)
    if angle_type == AngleType.DEGREES:
        return math.degrees(ret)
    return ret
