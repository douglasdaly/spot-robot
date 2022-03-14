from typing import Any, Dict

from squad.config import config
from squad.exceptions import FrozenError


class BodyParameters:
    """
    Storage class for (static) body data/parameters.
    """

    __slots__ = (
        "_frozen",
        "l_body",
        "w_body",
        "h_body",
        "l_hip",
        "l_femur",
        "l_leg",
        "l_rod",
        "l_rod_arm",
        "l_rod_femur",
        "h_rod_femur",
        "l_rod_leg",
        "cm_dx",
        "cm_dy",
        "cm_dz",
        "leg_alpha_min",
        "leg_alpha_max",
        "leg_beta_min",
        "leg_beta_max",
        "leg_gamma_min",
        "leg_gamma_max",
    )

    def __init__(self, **kwargs: float) -> None:
        self._frozen = False
        self.l_body = kwargs.pop("l_body", config.l_body)
        self.w_body = kwargs.pop("w_body", config.w_body)
        self.h_body = kwargs.pop("h_body", config.h_body)
        self.l_hip = kwargs.pop("l_hip", config.l_hip)
        self.l_femur = kwargs.pop("l_femur", config.l_femur)
        self.l_leg = kwargs.pop("l_leg", config.l_leg)
        self.l_rod = kwargs.pop("l_rod", config.l_rod)
        self.l_rod_arm = kwargs.pop("l_rod_arm", config.l_rod_arm)
        self.l_rod_femur = kwargs.pop("l_rod_femur", config.l_rod_femur)
        self.h_rod_femur = kwargs.pop("h_rod_femur", config.h_rod_femur)
        self.l_rod_leg = kwargs.pop("l_rod_leg", config.l_rod_leg)
        self.cm_dx = kwargs.pop("cm_dx", config.cm_dx)
        self.cm_dy = kwargs.pop("cm_dy", config.cm_dy)
        self.cm_dz = kwargs.pop("cm_dz", config.cm_dz)
        self.leg_alpha_min = kwargs.pop("leg_alpha_min", config.leg_alpha_min)
        self.leg_alpha_max = kwargs.pop("leg_alpha_max", config.leg_alpha_max)
        self.leg_beta_min = kwargs.pop("leg_beta_min", config.leg_beta_min)
        self.leg_beta_max = kwargs.pop("leg_beta_max", config.leg_beta_max)
        self.leg_gamma_min = kwargs.pop("leg_gamma_min", config.leg_gamma_min)
        self.leg_gamma_max = kwargs.pop("leg_gamma_max", config.leg_gamma_max)
        self._frozen = True

    def __repr__(self) -> str:
        return repr(self.__getstate__())

    def __setattr__(self, __name: str, __value: Any) -> None:
        if hasattr(self, "_frozen") and self._frozen:
            raise FrozenError(
                "BodyParameters objects are frozen and cannot be modified"
            )
        return super().__setattr__(__name, __value)

    def __getitem__(self, key: str) -> float:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __getstate__(self) -> Dict[str, Any]:
        state = {}
        for name in (x for x in self.__slots__ if x != "_frozen"):
            state[name] = getattr(self, name)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "_frozen", False)
        for k, v in state.items():
            setattr(self, k, v)
        object.__setattr__(self, "_frozen", True)

    def to_dict(self) -> Dict[str, float]:
        """Gets the parameters for this body in dictionary form.

        Returns
        -------
        dict
            The data dictionary representation of this object's data.

        """
        return self.__getstate__()

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BodyParameters":
        """Instantiates a new object from the given data.

        Parameters
        ----------
        data : dict
            The data to use to create the new body parameters object.

        Returns
        -------
        BodyParameters
            The new instance of the body parameters from the `data`
            given.

        """
        return cls(**data)
