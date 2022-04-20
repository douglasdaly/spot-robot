from abc import ABCMeta, abstractmethod
from datetime import datetime
import time
from typing import (
    Any,
    ByteString,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

import msgspec
import zmq

from squad.config import config
from squad.exceptions import AlreadySetupError, NotReadyError

from .constants import (
    DEFAULT_RETRIES,
    DEFAULT_SLEEP,
    DEFAULT_TIMEOUT,
    TZ,
    Sender,
    SystemCommand,
)
from .utils import get_msgspec_coders, get_msgspec_topic


class Message(msgspec.Struct, array_like=True):
    """
    Basic message class.
    """

    group: str
    sender: str
    id: UUID
    timestamp: datetime


class Command(msgspec.Struct, array_like=True):
    """
    Basic command message class.
    """

    recipient: str
    sender: str
    command: str
    id: UUID
    timestamp: datetime


M_In = TypeVar("M_In", bound=Message)
M_Out = TypeVar("M_Out", bound=Message)


class Endpoint(Generic[M_In, M_Out], metaclass=ABCMeta):
    """
    Base class for endpoints used to receive, handle, and send messages.

    Parameters
    ----------
    name : str
        The (unique) name to use for this endpoint.
    message_bus_addr : int or str
        The port or address of the message bus to use for transmitting
        outbound/generated messages from this endpoint.
    message_encoding : str, optional
        The encoding specification/codec to use for encoding messages to
        transmit and decoding received messages.
    command_bus_addr : int or str, optional
        The port or address of the main command bus to use for receiving
        commands (if any).
    group_addresses : Dict[str, Union[int, str]], optional
        Any group-to-port (or address/URL) subscriptions to set for this
        endpoint.
    timeout : int, optional
        The default timeout to use when attempting to pull messages or
        commands off the wire (in milliseconds).  If not provided a
        default value will be used.
    sleep_wait : int, optional
        The default interval (in milliseconds) to wait between loop
        iterations.  If not provided a default value will be used.
    command_callbacks : List[Callable[[Command], None]], optional
        Any callback(s) to invoke upon receiving commands from the main
        command bus (if any).
    inbound_callbacks : List[Callable[[M_In], None]], optional
        Any callback(s) to invoke upon receiving new messages from the
        wire (if any).
    outbound_callbacks : List[Callable[[M_Out], None]], optional
        Any callback(s) to invoke just prior to sending any response
        messages to the message bus (if any).

    """

    def __init__(
        self,
        name: str,
        message_bus_addr: Union[int, str],
        *,
        message_encoding: Optional[str] = None,
        command_bus_addr: Optional[Union[int, str]] = None,
        group_addresses: Dict[str, Union[int, str]] = {},
        timeout: Optional[int] = None,
        sleep_wait: Optional[int] = None,
        command_callbacks: List[Callable[[Command], None]] = [],
        inbound_callbacks: List[Callable[[M_In], None]] = [],
        outbound_callbacks: List[Callable[[M_Out], None]] = [],
    ) -> None:
        self._name = name
        if isinstance(message_bus_addr, int):
            self._msg_bus_addr = f"tcp://localhost:{message_bus_addr}"
        else:
            self._msg_bus_addr = message_bus_addr
        if isinstance(command_bus_addr, int):
            self._cmd_bus_addr = f"tcp://localhost:{command_bus_addr}"
        else:
            self._cmd_bus_addr = command_bus_addr
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._sleep: float = (sleep_wait or DEFAULT_SLEEP) / 1000.0

        self._groups: Dict[str, str] = {}
        for k, v in group_addresses:
            if isinstance(v, int):
                self._groups[k] = f"tcp://localhost:{v}"
            else:
                self._groups[k] = v

        self._cmd_cbs = command_callbacks
        self._in_cbs = inbound_callbacks
        self._out_cbs = outbound_callbacks

        # - Message/command encoders/decoders
        self._codec = (
            (message_encoding or config.message_encoding).strip().lower()
        )
        self._msg_encoder, self._msg_decoder = self._get_msg_coders(
            self._codec,
        )
        self._cmd_encoder, self._cmd_decoder = self._get_cmd_coders(
            self._codec,
        )

        # - ZMQ
        self._poller: Optional[zmq.Poller] = None
        self._msg_bus: Optional[zmq.Socket] = None
        self._cmd_bus: Optional[zmq.Socket] = None
        self._group_sockets: Dict[str, zmq.Socket] = {}

        self._ready = False
        self._shutdown = False

    @property
    def name(self) -> str:
        """str: The name of this endpoint."""
        return self._name

    @property
    def groups(self) -> Set[str]:
        """Set[str]: The set of groups this endpoint is a member of."""
        return set(self._groups.keys())

    @property
    def timeout(self) -> float:
        """float: The default timeout (in milliseconds)."""
        return self._timeout

    @property
    def sleep_wait(self) -> float:
        """float: The loop interval sleep time (in milliseconds)."""
        return self._sleep * 1000.0

    @property
    def ready(self) -> bool:
        """bool: Whether or not this endpoint is ready to start working."""
        return self._ready and not self._shutdown

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._name))

    def subscribe(self, group: str, address: str) -> None:
        """Adds a new group subscription to this endpoint.

        Parameters
        ----------
        group : str
            The name of the group to subscribe to messages for.
        address : str
            The address/URL of the group to subscribe to messages for.

        Raises
        ------
        AlreadySetupError
            If this object is already setup/configured and cannot modify
            subscriptions.

        """
        if self._ready:
            raise AlreadySetupError
        self._groups[group] = address

    def setup(self, ctx: zmq.Context) -> None:
        """Sets up this endpoint to begin working.

        Parameters
        ----------
        ctx : zmq.Context
            The ZMQ Context object to use to setup this endpoint.

        Raises
        ------
        AlreadySetupError
            If this endpoint has already been setup and this method is
            being called twice.  You must call :obj:`teardown` before
            calling this method again.

        """
        if self._ready:
            raise AlreadySetupError(f"{self.name} is already setup")

        if self._cmd_bus_addr:
            self._cmd_bus = self._cmd_bus_socket(ctx, self._cmd_bus_addr)

        self._msg_bus = self._msg_bus_socket(ctx, self._msg_bus_addr)

        rev_addr_grps = {}
        for group, address in self._groups.items():
            if address not in rev_addr_grps:
                rev_addr_grps[address] = []
            rev_addr_grps[address].append(group)

        self._poller = zmq.Poller()
        for address, groups in rev_addr_grps.items():
            socket = self._new_group_socket(ctx, address, *groups)
            self._poller.register(socket, zmq.POLLIN)
            self._group_sockets[address] = socket

        self._ready = True
        self._shutdown = False

    def teardown(self) -> None:
        """Tears down this endpoint to finish working.

        Raises
        ------
        NotReadyError
            If this endpoint isn't ready/setup and can't be torn down.
            You must call :obj:`setup` prior to this call.

        """
        if not self._ready:
            raise NotReadyError(f"{self.name} is not setup")

        self._ready = False

        # - Close group sockets
        for socket in self._group_sockets.values():
            self._poller.unregister(socket)  # type: ignore
            socket.close()
        self._group_sockets.clear()
        self._poller = None

        # - Close command socket and poller
        if self._cmd_bus is not None:
            self._cmd_bus.close()
        self._cmd_bus = None

        # - Close message bus socket
        if self._msg_bus is not None:
            self._msg_bus.close()
        self._msg_bus = None

    def _command_callbacks(self, cmd: Command) -> None:
        """Pass the given `cmd` to command callbacks."""
        for cb in self._cmd_cbs:
            cb(cmd)
        return

    def _inbound_callbacks(self, msg: M_In) -> None:
        """Pass the given `msg` to the receiver callbacks."""
        for cb in self._in_cbs:
            cb(msg)
        return

    def _outbound_callbacks(self, msg: M_Out) -> None:
        """Pass the given `msg` to the sender callbacks."""
        for cb in self._out_cbs:
            cb(msg)
        return

    def _pre_run(self) -> None:
        """Runs code once prior to starting main loop."""
        return

    def _post_run(self) -> None:
        """Runs code once after the main loop ends."""
        return

    def _pre_loop(self) -> None:
        """Runs any code prior to each main loop call."""
        return

    def _post_loop(self) -> None:
        """Runs any code after each main loop call."""
        return

    def _loop_once(self, timeout: Optional[int]) -> None:
        """Single iteration of the main run loop."""
        # - Get new command message and handle (if any)
        if self._cmd_bus:
            cmd_in = self._recv_cmd()
            if cmd_in:
                if self._cmd_cbs:
                    self._command_callbacks(cmd_in)
                cmd_out = self._handle_cmd(cmd_in)
                if cmd_out:
                    self._send_cmd(cmd_out)

        # - Get new group messages
        msg_in = self._recv(timeout=timeout)

        # - Inbound callbacks
        if msg_in and self._in_cbs:
            for m in msg_in:
                self._inbound_callbacks(m)

        # - Handle group messages
        msg_out = []
        for m in msg_in:
            m_out = self.handle(m)
            if m_out:
                msg_out.extend(m_out)

        # - Send response group messages
        for m in msg_out:
            self._send_msg(m)

        return

    def loop_once(self, timeout: Optional[int] = None) -> None:
        """Single iteration of the main run loop.

        Parameters
        ----------
        timeout : int, optional
            The amount of time to wait for messages in the call to
            :obj:`recv`, if not provided this endpoint's default timeout
            will be used.  If set to ``0`` or less no timeout will be
            used (blocking).

        Raises
        ------
        NotReadyError
            If this endpoint is not yet ready to receive messages.

        """
        if not self._ready:
            raise NotReadyError

        r_timeout = timeout or self._timeout
        if r_timeout <= 0:
            r_timeout = None

        return self._loop_once(r_timeout)

    def run(
        self,
        *,
        timeout: Optional[int] = None,
        sleep_wait: Optional[int] = None,
    ) -> None:
        """Runs this endpoint's main processing loop.

        Parameters
        ----------
        timeout : int, optional
            The timeout to use for getting requests (in milliseconds),
            if any.  If not provided the default timeout will be used.
            If a value less than or equal to zero is given then no
            timeout is used.
        sleep_wait : int, optional
            The sleep interval to wait between loop iterations, if any,
            in milliseconds.  If not provided the default sleep interval
            will be used.

        Raises
        ------
        NotReadyError
            If this endpoint is not ready to run yet.

        """
        if not self._ready:
            raise NotReadyError

        r_timeout = timeout or self._timeout
        if r_timeout <= 0:
            r_timeout = None

        if sleep_wait is not None:
            r_sleep = sleep_wait / 1000.0
        else:
            r_sleep = self._sleep

        # - Pre-loop code
        self._pre_run()

        # - Main loop
        while not self._shutdown:
            self._pre_loop()
            self._loop_once(r_timeout)
            self._post_loop()
            time.sleep(r_sleep)

        # - Post-loop code
        self._post_run()

    def start(
        self,
        ctx: zmq.Context,
        *,
        timeout: Optional[int] = None,
        sleep_wait: Optional[int] = None,
    ) -> None:
        """Runs the full endpoint process from setup to teardown.

        Parameters
        ----------
        ctx : zmq.Context
            The ZMQ context to setup this endpoint with.
        timeout : int, optional
            The timeout to use for getting requests (in milliseconds),
            if any.  If not provided the default timeout will be used.
            If a value less than or equal to zero is given then no
            timeout is used.
        sleep_wait : int, optional
            The sleep interval to wait between loop iterations, if any,
            in milliseconds.  If not provided the default sleep interval
            will be used.

        """
        self.setup(ctx)
        self.run(timeout=timeout, sleep_wait=sleep_wait)
        self.teardown()

    def _recv_cmd(self) -> Optional[Command]:
        """Receives a new command from the wire (if available)."""
        try:
            r_cmd: bytes = self._cmd_bus.recv(zmq.DONTWAIT)  # type: ignore
        except zmq.Again:
            return None
        cmd = self._cmd_decoder(r_cmd)
        return cmd

    def _recv(self, timeout: Optional[int]) -> List[M_In]:
        """Receives messages from the wire (if any are available)."""
        new_msgs = []
        socks: List[Tuple[zmq.Socket, int]] = self._poller.poll(timeout)  # type: ignore
        for sock, evt_type in socks:
            if sock in self._group_sockets.values() and evt_type == zmq.POLLIN:
                r_msg: bytes = sock.recv()  # type: ignore
                p_msg = self._msg_decoder(r_msg)
                new_msgs.append(p_msg)
        return new_msgs

    def recv(self, timeout: Optional[int] = None) -> List[M_In]:
        """Receives new message(s) from the wire.

        Parameters
        ----------
        timeout : int, optional
            The timeout to use (in milliseconds) for the receive call,
            if not provided the default timeout set on the class is
            used.  If given the value of ``-1`` no timeout is used.

        Returns
        -------
        List[M_In]
            The new message(s) received off the wire (if any).

        Raises
        ------
        NotReadyError
            If this endpoint is not yet ready to receive messages.

        """
        if not self._ready:
            raise NotReadyError

        r_timeout = timeout or self._timeout
        if r_timeout <= 0:
            r_timeout = None

        return self._recv(r_timeout)

    def _send_bytes(self, data: ByteString) -> None:
        """Transmits the given data to the main message bus."""
        self._msg_bus.send(data)  # type: ignore

    def _send_cmd(self, cmd: Command) -> None:
        """Sends the given `cmd` out to the main message bus."""
        r_cmd = self._cmd_encoder(cmd)
        return self._send_bytes(r_cmd)

    def _send_msg(self, msg: M_Out) -> None:
        """Sends the given `msg` out to the main message bus."""
        if self._out_cbs:
            self._outbound_callbacks(msg)
        r_msg = self._msg_encoder(msg)
        return self._send_bytes(r_msg)

    def send(self, msg: Union[Command, M_Out]) -> None:
        """Sends the given message or command to the main message bus.

        Parameters
        ----------
        msg : Command or M_Out
            The command or message object to send out.

        Raises
        ------
        NotReadyError
            If this endpoint is not yet ready to send messages.

        """
        if isinstance(msg, Command):
            return self._send_cmd(msg)
        else:
            return self._send_msg(msg)

    @abstractmethod
    def handle(self, msg: M_In) -> Optional[List[M_Out]]:
        """Handles the given `msg` and returns any responses to send.

        Parameters
        ----------
        msg : M_In
            The message to handle/process.

        Returns
        -------
        Optional[List[M_Out]]
            Any message(s) to send in response to the given `msg`.

        """
        raise NotImplementedError

    def _handle_other_cmd(self, cmd: Command) -> Optional[Command]:
        """Handles any additional commands this endpoint may receive."""
        return

    def _handle_system_cmd(self, cmd: Command) -> Optional[Command]:
        """Handles any system-related commands for this endpoint."""
        if cmd.command == SystemCommand.SHUTDOWN:
            self._shutdown = True
            ret_cmd = SystemCommand.ACKNOWLEDGE
        elif cmd.command == SystemCommand.HEARTBEAT:
            ret_cmd = SystemCommand.ACKNOWLEDGE
        else:
            ret_cmd = SystemCommand.UNKNOWN
        return self._generate_response_cmd(cmd, ret_cmd)

    def _handle_cmd(self, cmd: Command) -> Optional[Command]:
        """Logic for handling commands received."""
        if cmd.sender == Sender.SYSTEM:
            return self._handle_system_cmd(cmd)
        return self._handle_other_cmd(cmd)

    def handle_cmd(self, cmd: Command) -> Optional[Command]:
        """Handles the given `cmd` and returns a response (if needed).

        Parameters
        ----------
        cmd : Command
            The command message to handle/process.

        Returns
        -------
        Optional[Command]
            The reply command/message to send back in response (if any).

        """
        if not self._ready:
            raise NotReadyError
        return self._handle_cmd(cmd)

    def _generate_response_cmd(self, cmd_in: Command, command: str) -> Command:
        """Generates a response command to send."""
        return Command(
            self._name,
            cmd_in.sender,
            command,
            uuid4(),
            datetime.now(TZ),
        )

    def _generate_msg(self, group: str, *args: Any, **kwargs: Any) -> M_Out:
        """Generates a new message to send out."""
        return self.__class__.__orig_bases__[0].__args__[1](  # type: ignore
            group,
            self._name,
            uuid4(),
            datetime.now(TZ),
            *args,
            **kwargs,
        )

    def _generate_response_msg(
        self,
        msg_in: M_In,
        *args,
        **kwargs: Any,
    ) -> M_Out:
        """Generates a new messages in response to the given `msg_in`."""
        return self._generate_msg(msg_in.sender, *args, **kwargs)

    @classmethod
    def _get_cmd_coders(
        cls,
        codec: str,
    ) -> Tuple[Callable[[Command], bytes], Callable[[bytes], Command]]:
        """Gets the encoder/decoder pair to use for commands."""
        encoder, decoder = get_msgspec_coders(codec, Command)
        return encoder, decoder  # type: ignore

    @classmethod
    def _get_msg_coders(
        cls,
        codec: str,
    ) -> Tuple[Callable[[M_Out], bytes], Callable[[bytes], M_In]]:
        """Gets the encoder/decoder pair to use for messages."""
        enc_type, dec_type = cls.__orig_bases__[0].__args__  # type: ignore
        encoder, decoder = get_msgspec_coders(codec, enc_type, dec_type)
        return encoder, decoder  # type: ignore

    def _msg_bus_socket(self, ctx: zmq.Context, address: str) -> zmq.Socket:
        """Creates a new main message bus socket connection."""
        socket = ctx.socket(zmq.PUSH)
        socket.connect(address)
        return socket

    def _cmd_bus_socket(self, ctx: zmq.Context, address: str) -> zmq.Socket:
        """Creates a new main command bus socket connection."""
        socket = ctx.socket(zmq.PULL)
        socket.connect(address)
        return socket

    def _new_group_socket(
        self,
        ctx: zmq.Context,
        address: str,
        *groups: str,
    ) -> zmq.Socket:
        """Creates a new group socket connection to the given `address`."""
        socket = ctx.socket(zmq.SUB)
        socket.connect(address)
        for grp in groups:
            sub_grp = self._group_sub_topic(grp)
            socket.setsockopt(zmq.SUBSCRIBE, sub_grp)
        return socket

    def _group_sub_topic(self, group: str) -> bytes:
        """Generates the subscription topic to watch messages for."""
        m_cls = self.__class__.__orig_bases__[0].__args__[0]  # type: ignore
        return get_msgspec_topic(self._codec, m_cls, group)


class Bus:
    """
    Generic message bus/router.

    Parameters
    ----------
    name : str
        The (unique) name to use for this bus object.
    recv_addr : int or str
        The port or address of the receiver socket to bind to.
    pub_addr : int or str
        The port or address of the publisher socket to bind to.
    retries : int, optional
        The number of attempts to use when trying to pull new messages
        off the wire to transmit.  If not provided a default value will
        be used.
    timeout : int, optional
        The default timeout to use (in milliseconds) when attempting to
        pull messages off the wire.  If not provided a default value
        will be used.
    command_bus_addr : int or str, optional
        The port or address of the main command bus to use for receiving
        commands (if any).
    message_encoding : str, optional
        The encoding specification/codec to use when encoding messages
        and commands for transmission.
    command_callbacks : List[Callable[[Command], None]], optional
        Any callback(s) to use on commands received.
    inbound_callbacks : List[Callable[[bytes], None]], optional
        Any callback(s) to use on raw data messages received for
        relaying through the bus.

    """

    def __init__(
        self,
        name: str,
        recv_addr: Union[int, str],
        pub_addr: Union[int, str],
        *,
        retries: Optional[int] = None,
        timeout: Optional[int] = None,
        command_bus_addr: Optional[Union[int, str]] = None,
        message_encoding: Optional[str] = None,
        command_callbacks: List[Callable[[Command], None]] = [],
        inbound_callbacks: List[Callable[[bytes], None]] = [],
    ) -> None:
        self._name = name
        if isinstance(recv_addr, int):
            self._addr = f"tcp://*:{recv_addr}"
        else:
            self._addr = recv_addr
        if isinstance(pub_addr, int):
            self._pub_addr = f"tcp://*:{pub_addr}"
        else:
            self._pub_addr = pub_addr
        if isinstance(command_bus_addr, int):
            self._cmd_bus_addr = f"tcp://localhost:{command_bus_addr}"
        else:
            self._cmd_bus_addr = command_bus_addr
        self._retries = retries or DEFAULT_RETRIES
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._cmd_cbs = command_callbacks
        self._in_cbs = inbound_callbacks

        # - Encoding
        self._codec = (
            (message_encoding or config.message_encoding).strip().lower()
        )
        self._cmd_encoder, self._cmd_decoder = self._get_cmd_coders(
            self._codec,
        )

        # - ZMQ
        self._cmd_bus: Optional[zmq.Socket] = None
        self._receiver: Optional[zmq.Socket] = None
        self._publisher: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

        self._shutdown = False
        self._ready = False

    @property
    def name(self) -> str:
        """str: The name of this bus."""
        return self._name

    @property
    def address(self) -> str:
        """str: The address/URL of this bus's receiver."""
        return self._addr

    @property
    def pub_address(self) -> str:
        """str: The address/URL of this bus's publisher."""
        return self._pub_addr

    @property
    def ready(self) -> bool:
        """bool: Whether or not this bus is ready to route."""
        return self._ready and not self._shutdown

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._name))

    def setup(self, ctx: zmq.Context) -> None:
        """Sets up this bus to begin running.

        Parameters
        ----------
        ctx : zmq.Context
            The ZMQ context to use to setup this bus.

        Raises
        ------
        AlreadySetupError
            If this bus object has already been setup.

        """
        if self._ready:
            raise AlreadySetupError

        self._poller = zmq.Poller()

        # - Command bus
        if self._cmd_bus_addr:
            self._cmd_bus = ctx.socket(zmq.PULL)
            self._cmd_bus.connect(self._cmd_bus_addr)
            self._poller.register(self._cmd_bus, zmq.POLLIN)

        # - Message receiver
        self._receiver = ctx.socket(zmq.PULL)
        self._receiver.bind(self._addr)
        self._poller.register(self._receiver, zmq.POLLIN)

        # - Publisher
        self._publisher = ctx.socket(zmq.PUB)
        self._publisher.bind(self._pub_addr)

        self._shutdown = False
        self._ready = True

    def teardown(self) -> None:
        """Tears down this bus object.

        Raises
        ------
        NotReadyError
            If this bus object is not setup/ready in a state to be torn
            down.

        """
        if not self._ready:
            raise NotReadyError

        # - Publisher
        if self._publisher is not None:
            self._publisher.close()
        self._publisher = None

        # - Message receiver
        if self._receiver is not None:
            self._poller.unregister(self._receiver)  # type: ignore
            self._receiver.close()
        self._receiver = None

        # - Command bus
        if self._cmd_bus is not None:
            self._poller.unregister(self._cmd_bus)  # type: ignore
            self._cmd_bus.close()
        self._cmd_bus = None

        self._poller = None

        self._ready = False

    def _recv_cmd(self, timeout: float) -> Optional[Command]:
        """Attempts to pull a command message off the wire."""
        socks = dict(self._poller.poll(timeout))  # type: ignore
        if self._cmd_bus in socks and socks[self._cmd_bus] == zmq.POLLIN:
            r_cmd: bytes = self._cmd_bus.recv()  # type: ignore
            cmd = self._cmd_decoder(r_cmd)
            return cmd
        return None

    def _recv_msg(self, retries: int, timeout: float) -> List[bytes]:
        """Attempts to pull messages off the wire."""
        ret = []
        curr_i = 0
        while curr_i < retries:
            socks = dict(self._poller.poll(timeout))  # type: ignore
            if self._receiver in socks and socks[self._receiver] == zmq.POLLIN:
                new_msg = self._receiver.recv()  # type: ignore
                ret.append(new_msg)
            curr_i += 1
        return ret

    def _send_bytes(self, socket: zmq.Socket, data: bytes) -> None:
        """Sends the given data over the wire."""
        socket.send(data)

    def _send_cmd(self, cmd: Command) -> None:
        """Sends the given command to the command bus."""
        b_cmd = self._cmd_encoder(cmd)
        return self._send_bytes(self._cmd_bus, b_cmd)  # type: ignore

    def _loop_once(self, retries: int, timeout: float) -> None:
        """Single iteration of processing loop."""
        # - Get any command and process first
        if self._cmd_bus:
            cmd_in = self._recv_cmd(timeout)
            if cmd_in:
                if self._cmd_cbs:
                    for cb in self._cmd_cbs:
                        cb(cmd_in)
                cmd_out = self._handle_cmd(cmd_in)
                if cmd_out:
                    self._send_cmd(cmd_out)

        # - Get any received messages and relay them
        msg_in = self._recv_msg(retries, timeout)
        for r_msg in msg_in:
            if self._in_cbs:
                for cb in self._in_cbs:
                    cb(r_msg)
            self._send_bytes(self._publisher, r_msg)  # type: ignore
        return

    def loop_once(
        self,
        *,
        retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Runs a single iteration of the main processing loop.

        Parameters
        ----------
        retries : int, optional
            The number of attempts to make to fetch messages off the
            wire.  If not provided the default set for the object will
            be used.
        timeout : int, optional
            The timeout to use for pulling messages from the wire (in
            milliseconds).  If not provided then the default value set
            on this object will be used.

        Raises
        ------
        NotReadyError
            If this bus object is not yet ready to run the main loop.

        """
        if not self._ready:
            raise NotReadyError

        l_retries = retries or self._retries
        if timeout is not None:
            l_timeout = timeout
        else:
            l_timeout = self._timeout

        return self._loop_once(l_retries, l_timeout)

    def _pre_run(self) -> None:
        """Code to run once prior to starting the main processing loop."""
        return

    def _post_run(self) -> None:
        """Code to run once after the main processing loop ends."""
        return

    def _pre_loop(self) -> None:
        """Code to run before each iteration of the main loop."""
        return

    def _post_loop(self) -> None:
        """Code to run after each iteration of the main loop."""
        return

    def run(
        self,
        *,
        retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Main run loop for this bus object.

        Parameters
        ----------
        retries : int, optional
            The number of attempts to make to fetch messages off the
            wire.  If not provided the default set for the object will
            be used.
        timeout : int, optional
            The timeout to use for pulling messages from the wire (in
            milliseconds).  If not provided then the default value set
            on this object will be used.

        Raises
        ------
        NotReadyError
            If this bus object is not yet ready to run.

        """
        if not self._ready:
            raise NotReadyError

        l_retries = retries or self._retries
        if timeout is not None:
            l_timeout = timeout
        else:
            l_timeout = self._timeout

        self._pre_run()

        while not self._shutdown:
            self._pre_loop()
            self._loop_once(l_retries, l_timeout)
            self._post_loop()

        self._post_run()

    def start(
        self,
        ctx: zmq.Context,
        *,
        retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Runs the full bus process from setup to run to teardown.

        Parameters
        ----------
        ctx : zmq.Context
            The ZMQ context to use to setup this bus.
        retries : int, optional
            The number of attempts to make to fetch messages off the
            wire.  If not provided the default set for the object will
            be used.
        timeout : int, optional
            The timeout to use for pulling messages from the wire (in
            milliseconds).  If not provided then the default value set
            on this object will be used.

        """
        self.setup(ctx)
        self.run(retries=retries, timeout=timeout)
        self.teardown()

    def _handle_system_cmd(self, cmd: Command) -> Optional[Command]:
        """Handles a system-based command."""
        if cmd.command == SystemCommand.SHUTDOWN:
            self._shutdown = True
            ret_cmd = SystemCommand.ACKNOWLEDGE
        elif cmd.command == SystemCommand.HEARTBEAT:
            ret_cmd = SystemCommand.ACKNOWLEDGE
        else:
            ret_cmd = SystemCommand.UNKNOWN
        return self._generate_response_cmd(cmd, ret_cmd)

    def _handle_other_cmd(self, cmd: Command) -> Optional[Command]:
        """Handles all other commands."""
        return

    def _handle_cmd(self, cmd: Command) -> Optional[Command]:
        """Processes a single command received."""
        if cmd.sender == Sender.SYSTEM:
            return self._handle_system_cmd(cmd)
        return self._handle_other_cmd(cmd)

    def handle_cmd(self, cmd: Command) -> Optional[Command]:
        """Processes and handles a single command.

        Parameters
        ----------
        cmd : Command
            The command to process/handle.

        Returns
        -------
        Optional[Command]
            The response command (if any) generated from processing the
            given `cmd`.

        Raises
        ------
        NotReadyError
            If this bus object is not yet ready to handle commands.

        """
        if not self._ready:
            raise NotReadyError
        return self._handle_cmd(cmd)

    def _generate_response_cmd(self, cmd_in: Command, command: str) -> Command:
        """Generates a response command for the given `cmd_in`."""
        return Command(
            cmd_in.sender,
            self._name,
            command,
            uuid4(),
            datetime.now(TZ),
        )

    @classmethod
    def _get_cmd_coders(
        cls,
        codec: str,
    ) -> Tuple[Callable[[Command], bytes], Callable[[bytes], Command]]:
        """Gets the encoder/decoder pair for command objects."""
        return get_msgspec_coders(codec, Command)
