from typing import Optional, Tuple

from squad.constants import AngleType, Leg

from .base import BodyParameters
from .forward import foot_xyz, hip_xyz, leg_servo_to_knee_angle
from .inverse import body_thetas, knee_angle_to_leg_servo, leg_thetas


class KinematicSolver:
    """
    Computation of kinematics for a particular robot body configuration.
    """

    def __init__(
        self,
        body_params: Optional[BodyParameters] = None,
        angle_type: AngleType = AngleType.DEGREES,
    ) -> None:
        if body_params is None:
            self._body = BodyParameters()
        else:
            self._body = body_params
        self._angle_type = angle_type

    def foot_fwd(
        self,
        leg: Leg,
        hip_theta: float,
        femur_theta: float,
        leg_theta: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Computes the X, Y, and Z-coordinates of the given foot.

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

        Returns
        -------
        Tuple[float, float, float]
            The X, Y, and Z coordinates of the foot, for the given `leg` and
            angles, in the body's coordinate frame.

        """
        knee_theta = leg_servo_to_knee_angle(
            leg_theta,
            body_params=self._body,
            angle_type=self._angle_type,
        )
        return foot_xyz(
            leg,
            hip_theta,
            femur_theta,
            knee_theta,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            body_params=self._body,
            angle_type=self._angle_type,
        )

    def foot_inv(
        self,
        leg: Leg,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Computes the servo thetas for the given foot position.

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

        Returns
        -------
        Tuple[float, float, float]
            The servo thetas (Hip, Femur, Leg) for the given leg's foot
            position and body orientation.

        """
        t_hip, t_femur, t_knee = leg_thetas(
            leg,
            x,
            y,
            z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            body_params=self._body,
            angle_type=self._angle_type,
        )
        t_leg = knee_angle_to_leg_servo(
            t_knee,
            body_params=self._body,
            angle_type=self._angle_type,
        )
        return (t_hip, t_femur, t_leg)

    def hip_fwd(
        self,
        leg: Leg,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Tuple[float, float, float]:
        """Gets the position of the hip based on the body orientation.

        Parameters
        ----------
        leg : Leg
            The leg to compute the hip coordinate for.
        roll : float
            The roll of the main body to compute the hip coordinates
            for.
        pitch : float
            The pitch of the main body to compute the hip coordinates
            for.
        yaw : float
            The yaw of the main body to compute the hip coordinates for.

        Returns
        -------
        Tuple[float, float, float]
            The X, Y, and Z coordinates of the hip for the specified
            `leg`, based on the given angles, in the body's coordinate
            frame.

        """
        return hip_xyz(
            leg,
            roll,
            pitch,
            yaw,
            body_params=self._body,
            angle_type=self._angle_type,
        )

    def hip_inv(
        self,
        leg: Leg,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
        """Gets the body orientation based on the given position
        relative to the specified leg's hip.

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

        Returns
        -------
        Tuple[float, float, float]
            The body's orientation (Roll, Pitch, Yaw) based on the
            position of the given leg's hip.

        """
        return body_thetas(
            leg,
            x,
            y,
            z,
            body_params=self._body,
            angle_type=self._angle_type,
        )
