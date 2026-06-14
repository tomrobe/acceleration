from .models import (
    PhysicalParameters,
    InitialConditionsGeneralModel,
    ConfigurationGeneralModel,
    SolutionGeneralModel,
)

from .solvers.general_solver import solve, solve_multiple

from .loader import load, run
