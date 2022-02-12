from typing import Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt


Vector = Union[Tuple[float, float, float], npt.NDArray[np.floating]]

VectorT = TypeVar(
    "VectorT",
    Tuple[float, float, float],
    npt.NDArray[np.floating],
)
