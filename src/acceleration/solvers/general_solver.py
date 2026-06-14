import numpy as np
from scipy.integrate import solve_ivp

from ..models.common import PhysicalParameters
from ..models.general_model import ConfigurationGeneralModel, SolutionGeneralModel

def eom(t, y, par: PhysicalParameters):
    rho, drho, theta, dtheta = y

    ddrho = (
        (dtheta**2 + par.E**2) * rho
        + par.lamb * np.cos(par.phi + par.m * theta) * rho**par.l
    )

    ddtheta = (
        - 2 * drho * dtheta / rho
        + par.lamb * np.sin(par.phi + par.m * theta) * rho**(par.l - 1)
    )

    return [drho, ddrho, dtheta, ddtheta]

def solve(config: ConfigurationGeneralModel) -> SolutionGeneralModel:
    y0 = [config.init.rho_0, config.init.drho_0, config.init.theta_0, config.init.dtheta_0] # None

    sol = solve_ivp (
        fun = lambda t, y: eom (t, y, config.parameters),
        t_span = config.t_span,
        y0 = y0,
        t_eval = config.t_eval,
        method = 'Radau',
        dense_output = True
    )

    return SolutionGeneralModel(
        t = sol.t,
        rho = sol.y[0],
        drho = sol.y[1],
        theta = sol.y[2],
        dtheta = sol.y[3],
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
    return solutiions
