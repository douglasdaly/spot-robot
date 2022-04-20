import struct
from typing import Optional, Tuple

import serial

from squad.config import config

from .base import BaseIO


T_In = Tuple[float, float, float, float, float, float]
T_Out = Tuple[float, float, float, float, float, float]


class Arduino(BaseIO[T_Out, T_In]):
    """
    Serial communication for Arduino.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        interval: Optional[float] = None,
        *,
        baud_rate: Optional[int] = None,
        timeout: Optional[int] = None,
        little_endian: Optional[bool] = None,
    ) -> None:
        self._port = port or config.serial_port
        self._interval = interval or config.serial_interval
        self._baud_rate = baud_rate or config.serial_baud_rate
        self._timeout = config.serial_timeout if timeout is None else timeout
        self._little_endian = (
            config.board_little_endian
            if little_endian is None
            else little_endian
        )

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baud_rate,
            timeout=self._timeout,
        )

        pack_pre = "<" if self._little_endian else ">"
        self.__send_fmt = f"{pack_pre}fffffffff"
        self.__recv_fmt = f"{pack_pre}ffffff"

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

    @property
    def little_endian(self) -> bool:
        """bool: Whether or not the other end is little endian."""
        return self._little_endian

    def send(self, data: T_Out) -> None:
        to_send = struct.pack(self.__send_fmt, *data)
        self._serial.write(to_send)
        self._serial.flush()

    def receive(self) -> T_In:
        bytes_in = self._serial.read(24)
        self._serial.reset_input_buffer()
        return struct.unpack(self.__recv_fmt, bytes_in)

    def close(self) -> None:
        self._serial.close()
