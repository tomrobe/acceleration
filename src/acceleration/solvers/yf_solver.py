import numpy as np
from scipy.integrate import solve_ivp

from ..models.common import PhysicalParameters
from ..models.yf_model import ConfigurationYFModel, SolutionYFModel

def eom(x, yf, par: PhysicalParameters):
    y, f = yf

    dy = (
        - 3 * y**2 / f
        + f
        + par.lamb * np.cos(par.phi + par.m*x) / f
    )
    df = (
        - 4 * y
        + par.lamb * np.sin(par.phi + par.m*x) / f
    )
    return [dy, df]

def solve(config: ConfigurationYFModel) -> SolutionYFModel:
    y0 = [config.init.y_0, config.init.f_0] # None

    sol = solve_ivp (
        fun = lambda t, yf: eom (t, yf, config.parameters),
        t_span = config.t_span,
        y0 = y0,
        t_eval = config.t_eval,
        method = 'Radau',
        dense_output = True
    )

    return SolutionYFModel(
        t = sol.t,
        y = sol.y[0],
        f = sol.y[1],
        config = config,
        success = sol.success
    )

def solve_multiple(configs):
    solutions = []
    for i,config in enumerate(configs):
        sol = solve(config)
        if not sol.success:
            print(f"Warning: solver did not converge for '{config.name}'")
        solutions.append(sol)
    return solutions
