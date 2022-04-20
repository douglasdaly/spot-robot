from squad.constants import Leg

from .base import Message


class ServoAngleMessage(Message):
    """
    Servo angle data message.
    """

    leg: Leg
    theta_hip: float
    theta_femur: float
    theta_leg: float


class ImuDataMessage(Message):
    """
    IMU data message.
    """

    accel_x: float
    accel_y: float
    accel_z: float
    d_roll: float
    d_pitch: float
    d_yaw: float
