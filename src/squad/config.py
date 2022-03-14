import os
from typing import Any, Literal, Optional, Type, TypeVar, overload

from dotenv import load_dotenv

from .exceptions import ConfigError


T = TypeVar("T")

# Defaults
DEFAULT_BOARD_ARCH = 32
DEFAULT_BOARD_LITTLE_ENDIAN = True

DEFAULT_SERIAL_INTERVAL = 0.02
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_SERIAL_BAUD_RATE = 115200
DEFAULT_SERIAL_TIMEOUT = 1

DEFAULT_SERVO_MAX_ANGLE = 270.0


@overload
def get_env_value(
    name: str,
    default: Optional[T] = ...,
    *,
    type_: Optional[Type[T]] = ...,
    raise_ex: Literal[True] = ...,
) -> T:
    ...


@overload
def get_env_value(
    name: str,
    default: Optional[T] = ...,
    *,
    type_: Optional[Type[T]] = ...,
    raise_ex: Literal[False] = ...,
) -> Optional[T]:
    ...


def get_env_value(
    name: str,
    default: Optional[T] = None,
    *,
    type_: Optional[Type[T]] = None,
    raise_ex: bool = True,
) -> Optional[T]:
    """Gets the environment variable with the corresponding name.

    Parameters
    ----------
    name : str
        The name (excluding the prefix ``SR_``) of the environment
        variable to get (if it exists).
    default : T
        The default value to use if the variable with the given `name`
        isn't set.
    type_ : Type[T], optional
        The data type of the variable to get (will attempt to infer from
        the `default` value given, otherwise assumes ``float``).
    raise_ex : bool, default=True
        Whether or not to load an exception if the returned value would
        be ``None``.

    Returns
    -------
    T or None
        The value of the specified environment variable (if found),
        otherwise returns `default` (if provided) or ``None``.

    Raises
    ------
    ConfigError
        If no value is found for the given environment variable `name`
        and no `default` value is provided and `raise_ex` is ``True``.

    """
    if type_ is None:
        if default is not None:
            type_out = type(default)
        else:
            type_out = float
    else:
        type_out = type_

    env_name = f"SR_{name.strip().upper()}"
    env_val = os.getenv(env_name, default)
    if env_val is None:
        if raise_ex:
            raise ConfigError(f"Value required for: {env_name}")
        return None
    elif type_out == bool and isinstance(env_val, str):
        return env_val.strip().lower() in ("true", "1")  # type: ignore
    return type_out(env_val)  # type: ignore


class Config:
    """
    Configuration storage object.

    Parameters
    ----------
    kwargs : Any, optional
        Any configuration settings to overwrite the environment value
        for.

    """

    def __init__(self, **kwargs: Any) -> None:
        load_dotenv()

        # Robot specifications
        # - Body dimensions (in mm)
        self.l_body: float = get_env_value("L_BODY", kwargs.get("l_body"))
        self.w_body: float = get_env_value("W_BODY", kwargs.get("w_body"))
        self.h_body: float = get_env_value("H_BODY", kwargs.get("h_body"))

        # - Leg dimensions (in mm)
        self.l_hip: float = get_env_value("L_HIP", kwargs.get("l_hip"))
        self.l_femur: float = get_env_value("L_FEMUR", kwargs.get("l_femur"))
        self.l_leg: float = get_env_value("L_LEG", kwargs.get("l_leg"))

        # - Femur-to-Leg rod (in mm)
        self.l_rod: float = get_env_value("L_ROD", kwargs.get("l_rod"))
        self.l_rod_arm: float = get_env_value(
            "L_ROD_ARM",
            kwargs.get("l_rod_arm"),
        )
        self.l_rod_femur: float = get_env_value(
            "L_ROD_FEMUR",
            kwargs.get("l_rod_femur"),
        )
        self.h_rod_femur: float = get_env_value(
            "H_ROD_FEMUR",
            kwargs.get("h_rod_femur", 0.0),
        )
        self.l_rod_leg: float = get_env_value(
            "L_ROD_LEG",
            kwargs.get("l_rod_leg"),
        )

        # - Center of Mass offsets (in mm, from absolute center)
        self.cm_dx: float = get_env_value("CM_DX", kwargs.get("cm_dx", 0.0))
        self.cm_dy: float = get_env_value("CM_DY", kwargs.get("cm_dy", 0.0))
        self.cm_dz: float = get_env_value("CM_DZ", kwargs.get("cm_dz", 0.0))

        # - Servo max angle (in degrees)
        self.servo_max_angle: float = get_env_value(
            "SERVO_MAX_ANGLE",
            kwargs.get("servo_max_angle", DEFAULT_SERVO_MAX_ANGLE),
        )

        # Robot Kinematics
        # - Leg servo min/max angles (in degrees)
        # -- Alpha (hip servo)
        self.leg_alpha_min: float = get_env_value(
            "LEG_ALPHA_MIN",
            kwargs.get("leg_alpha_min"),
        )
        self.leg_alpha_max: float = get_env_value(
            "LEG_ALPHA_MAX",
            kwargs.get("leg_alpha_max"),
        )
        # -- Beta (femur servo)
        self.leg_beta_min: float = get_env_value(
            "LEG_BETA_MIN",
            kwargs.get("leg_beta_min"),
        )
        self.leg_beta_max: float = get_env_value(
            "LEG_BETA_MAX",
            kwargs.get("leg_beta_max"),
        )
        # -- Gamma (leg servo)
        self.leg_gamma_min: float = get_env_value(
            "LEG_GAMMA_MIN",
            kwargs.get("leg_gamma_min"),
        )
        self.leg_gamma_max: float = get_env_value(
            "LEG_GAMMA_MAX",
            kwargs.get("leg_gamma_max"),
        )

        # Control board parameters
        self.board_arch: int = get_env_value(
            "BOARD_ARCH",
            kwargs.get("board_arch", DEFAULT_BOARD_ARCH),
            type_=int,
        )
        self.board_little_endian: bool = get_env_value(
            "BOARD_LITTLE_ENDIAN",
            kwargs.get("board_little_endian", DEFAULT_BOARD_LITTLE_ENDIAN),
            type_=bool,
        )

        # - Serial connection
        self.serial_interval: float = get_env_value(
            "SERIAL_INTERVAL",
            kwargs.get("serial_interval", DEFAULT_SERIAL_INTERVAL),
        )
        self.serial_port: str = get_env_value(
            "SERIAL_PORT",
            kwargs.get("serial_port", DEFAULT_SERIAL_PORT),
            type_=str,
        )
        self.serial_baud_rate: int = get_env_value(
            "SERIAL_BAUD_RATE",
            kwargs.get("serial_baud_rate", DEFAULT_SERIAL_BAUD_RATE),
            type_=int,
        )
        self.serial_timeout: int = get_env_value(
            "SERIAL_TIMEOUT",
            kwargs.get("serial_timeout", DEFAULT_SERIAL_TIMEOUT),
        )


config = Config()


def reload_config(**kwargs: Any) -> None:
    """Reloads the configuration data from the environment.

    Parameters
    ----------
    kwargs : Any, optional
        Any configuration settings to overwrite the environment value
        for.

    """
    global config

    load_dotenv()
    config = Config(**kwargs)
