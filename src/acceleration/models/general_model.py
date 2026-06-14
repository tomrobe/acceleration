from dataclasses import dataclass
import numpy as np
from .common import parameters

@dataclass
class InitialConditionsGeneralModel:
    rho_0: float
    drho_0: float
    theta_0: float
    dtheta_0: float

@dataclass
class ConfigurationGeneralModel:
    parameters: parameters
    init: InitialConditionsGeneralModel
    t_span: tuple[float,float]
    t_eval: np.ndarray
    name: str = "unnamed"

    def __post_init__(self):
        if self.t_eval == None:
            self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 1001)

@dataclass
class SolutionGeneralModel:
    t: np.ndarray
    rho: np.ndarray
    drho: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    config: ConfigurationGeneralModel
    success: bool
