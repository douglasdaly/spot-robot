from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T_Out = TypeVar("T_Out")
T_In = TypeVar("T_In")


class BaseIO(Generic[T_Out, T_In], ABC):
    """
    Base class for I/O communication objects.
    """

    @abstractmethod
    def send(self, data: T_Out) -> None:
        """Sends the given data over this communication object.

        Parameters
        ----------
        data : T_Out
            The data to send over this I/O channel.

        """
        raise NotImplementedError

    @abstractmethod
    def receive(self) -> T_In:
        """Receives data from this communication object.

        Returns
        -------
        T_In
            The data received over this I/O channel.

        """
        raise NotImplementedError
