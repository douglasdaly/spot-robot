from typing import Any, Callable, Optional, Tuple, Type
from uuid import UUID

import msgspec


def _ms_enc_hook(obj: Any) -> Any:
    """msgspec encoder hook for other data types."""
    if isinstance(obj, UUID):
        return obj.hex
    else:
        raise TypeError(f"Objects of type {type(obj)} are not supported")


def _ms_dec_hook(type: Type, obj: Any) -> Any:
    """msgspec decoder hook for other data types."""
    if type is UUID:
        return UUID(obj)
    else:
        raise TypeError(f"Objects of type {type} are not supported")


def get_msgspec_coders(
    codec: str,
    enc_type: Type[Any],
    dec_type: Optional[Type[Any]] = None,
) -> Tuple[Callable[[Any], bytes], Callable[[bytes], Any]]:
    """Gets the ``msgspec`` library encoder/decoder specified.

    Parameters
    ----------
    codec : str
        The name of the encoding to get the objects for (either ``json``
        or ``msgpack``).
    enc_type : type
        The object type to create the encoder for.
    dec_type : type, optional
        The object type to create the decoder for (if not given then the
        type specified by the given `enc_type` will be used).

    Returns
    -------
    Tuple[Encoder, Decoder]
        The requested encoder/decoder pair (if supported).

    Raises
    ------
    NotImplementedError
        If the specified `codec` is not implemented/supported.

    """
    if dec_type is None:
        dec_type = enc_type

    if codec == "json":
        encoder = msgspec.json.Encoder(enc_hook=_ms_enc_hook)
        decoder = msgspec.json.Decoder(type=dec_type, dec_hook=_ms_dec_hook)
    elif codec == "msgpack":
        encoder = msgspec.msgpack.Encoder(enc_hook=_ms_enc_hook)
        decoder = msgspec.msgpack.Decoder(type=dec_type, dec_hook=_ms_dec_hook)
    else:
        raise NotImplementedError(codec)
    return encoder.encode, decoder.decode


def get_msgspec_topic(
    codec: str,
    obj_type: Type[msgspec.Struct],
    match_first: str,
) -> bytes:
    """Gets the topic bytes to match for ZMQ subscribers.

    Parameters
    ----------
    codec : str
        The encoding framework being used on the transmitted data.
    obj_type : msgspec.Struct
        The class of the data objects being transmitted.
    match_first : str
        The first field value in the encoded object (must be of type
        ``str``) to match against.

    Returns
    -------
    bytes
        The bytes to use for subscription topic matching.

    Raises
    ------
    NotImplementedError
        If the specified `codec` is not supported.

    """
    if codec == "json":
        pre = '["'.encode("UTF-8")
    elif codec == "msgpack":
        n_elem = len(obj_type.__struct_fields__)
        if n_elem <= 15:
            arr_pre = (144 + n_elem).to_bytes(1, "big", signed=False)
        elif 16 <= n_elem <= ((2 ** 16) - 1):
            arr_pre = b"\xdc" + n_elem.to_bytes(16, "big", signed=False)
        else:
            arr_pre = b"\xdd" + n_elem.to_bytes(32, "big", signed=False)

        n_str = len(match_first)
        if n_str <= 31:
            str_pre = (160 + n_str).to_bytes(1, "big", signed=False)
        elif 32 <= n_str <= ((2 ** 8) - 1):
            str_pre = b"\xd9" + n_str.to_bytes(8, "big", signed=False)
        elif (2 ** 8) <= n_str <= ((2 ** 16) - 1):
            str_pre = b"\xda" + n_str.to_bytes(16, "big", signed=False)
        else:
            str_pre = b"\xdb" + n_str.to_bytes(32, "big", signed=False)

        pre = arr_pre + str_pre
    else:
        raise NotImplementedError(codec)

    return pre + match_first.encode("UTF-8")
