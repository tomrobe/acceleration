from dataclasses import dataclass
import numpy as np
from .common import PhysicalParameters

@dataclass
class InitialConditionsYFModel:
    y_0: float
    f_0: float

@dataclass
class ConfigurationYFModel:
    parameters: PhysicalParameters
    init: InitialConditionsYFModel
    t_span: tuple[float,float]
    t_eval: np.ndarray
    name: str = "unnamed"

@dataclass
class SolutionYFModel:
    t: np.ndarray
    y: np.ndarray
    f: np.ndarray
    config: ConfigurationYFModel
    success: bool
