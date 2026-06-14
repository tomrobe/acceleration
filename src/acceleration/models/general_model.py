from dataclasses import dataclass
import numpy as np
from .common import PhysicalParameters

@dataclass
class InitialConditionsGeneralModel:
    rho_0: float
    drho_0: float
    theta_0: float
    dtheta_0: float

@dataclass
class ConfigurationGeneralModel:
    parameters: PhysicalParameters
    init: InitialConditionsGeneralModel
    t_span: tuple[float,float]
    t_eval: np.ndarray
    name: str = "unnamed"

@dataclass
class SolutionGeneralModel:
    t: np.ndarray
    rho: np.ndarray
    drho: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    config: ConfigurationGeneralModel
    success: bool
