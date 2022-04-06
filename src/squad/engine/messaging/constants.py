import pytz

from squad.config import config


# Defaults
DEFAULT_TIMEOUT = 100
DEFAULT_RETRIES = 5
DEFAULT_SLEEP = 10


# Constants
TZ = pytz.timezone(config.timezone)


# - Messaging


class Sender:
    """
    Standard sender names.
    """

    SYSTEM = "SYS"


class SystemCommand:
    """
    System command codes.
    """

    ACKNOWLEDGE = "ACK"
    HEARTBEAT = "HBT"
    UNKNOWN = "UNK"
    SHUTDOWN = "SDN"
