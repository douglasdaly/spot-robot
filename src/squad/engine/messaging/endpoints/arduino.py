import time
from typing import Callable, Dict, List, Optional, Union

from squad.communication.arduino import Arduino

from ..base import Command, Endpoint
from ..constants import Group
from ..messages import ImuDataMessage, ServoAngleMessage


class ArduinoEndpoint(Endpoint[ServoAngleMessage, ImuDataMessage]):
    """
    Arduino IO endpoint.
    """

    def __init__(
        self,
        name: str,
        message_bus_addr: Union[int, str],
        *,
        arduino_port: Optional[str] = None,
        arduino_interval: Optional[float] = None,
        arduino_baud_rate: Optional[int] = None,
        arduino_timeout: Optional[int] = None,
        arduino_little_endian: Optional[bool] = None,
        message_encoding: Optional[str] = None,
        command_bus_addr: Optional[Union[int, str]] = None,
        group_addresses: Dict[str, Union[int, str]] = ...,
        timeout: Optional[int] = None,
        sleep_wait: Optional[int] = None,
        command_callbacks: List[Callable[[Command], None]] = ...,
        inbound_callbacks: List[Callable[[ServoAngleMessage], None]] = ...,
        outbound_callbacks: List[Callable[[ImuDataMessage], None]] = ...,
    ) -> None:
        self._arduino = Arduino(
            port=arduino_port,
            interval=arduino_interval,
            baud_rate=arduino_baud_rate,
            timeout=arduino_timeout,
            little_endian=arduino_little_endian,
        )
        self._angles = [0.0 for _ in range(12)]
        self._last_write = 0.0

        return super().__init__(
            name,
            message_bus_addr,
            message_encoding=message_encoding,
            command_bus_addr=command_bus_addr,
            group_addresses=group_addresses,
            timeout=timeout,
            sleep_wait=sleep_wait,
            command_callbacks=command_callbacks,
            inbound_callbacks=inbound_callbacks,
            outbound_callbacks=outbound_callbacks,
        )

    @property
    def arduino(self) -> Arduino:
        """Arduino: The Arduino IO object used by this endpoint."""
        return self._arduino

    def _post_loop(self) -> None:
        super()._post_loop()
        c_time = time.time() * 1000.0
        if (c_time - self._last_write) >= self._arduino.interval:
            # - Send angles out
            self._arduino.send(self._angles)  # type: ignore
            self._last_write = c_time

            # - Get IMU data in & send out
            imu_data = self._arduino.receive()
            imu_mesg = self._generate_msg(Group.IMU, *imu_data)
            self._send_msg(imu_mesg)
        return

    def _pre_run(self) -> None:
        super()._pre_run()
        self._last_write = time.time() * 1000.0

    def _post_run(self) -> None:
        super()._post_run()
        self._arduino.close()

    def handle(self, msg: ServoAngleMessage) -> Optional[List[ImuDataMessage]]:
        s_idx = msg.leg.value - 1
        self._angles[s_idx] = msg.theta_hip
        self._angles[s_idx + 1] = msg.theta_femur
        self._angles[s_idx + 2] = msg.theta_leg
