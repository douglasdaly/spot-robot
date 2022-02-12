from typing import Optional, Tuple

import serial

from squad.config import config

from .base import BaseIO


T_Out = Tuple[float, float, float, float, float, float, float, float, float]
T_In = Tuple[float, float, float, float, float, float]


class SerialIO(BaseIO[T_Out, T_In]):
    """
    Serial communication for Arduino I/O.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        interval: Optional[float] = None,
        *,
        baud_rate: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self._port = port or config.serial_port
        self._interval = interval or config.serial_interval
        self._baud_rate = baud_rate or config.serial_baud_rate
        self._timeout = config.serial_timeout if timeout is None else timeout

        self._serial = serial.Serial(
            port=self._port,
            baud_rate=self._baud_rate,
            timeout=self._timeout,
        )

    @property
    def port(self) -> str:
        """str: The serial port this bus is connected to."""
        return self._port

    @property
    def interval(self) -> float:
        """str: The serial communication interval (in ms)."""
        return self._interval

    @property
    def baud_rate(self) -> int:
        """int: The baud rate of this serial connection."""
        return self._baud_rate

    @property
    def timeout(self) -> int:
        """int: The timeout length (in ms) for this serial connection."""
        return self._timeout
