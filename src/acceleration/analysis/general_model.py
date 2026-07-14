import numpy as np

from ..models.general_model import SolutionGeneralModel

# Define desired quantities as function, then include them in the return of analyze()

def compute_w(sol: SolutionGeneralModel):
    p = sol.config.parameters
    temp = (sol.dtheta**2 + p.E**2) * sol.rho**2 + p.lamb * np.cos(p.phi + p.m * sol.theta) * sol.rho**(p.l + 1)
    return (2 - temp) / sol.drho**2

def compute_y(sol: SolutionGeneralModel):
    return sol.drho / sol.rho**3

def compute_redshift(sol: SolutionGeneralModel):
    return (sol.rho[-1] / sol.rho)**(2/3) - 1

def compute_redshift_q(sol: SolutionGeneralModel):
    return (sol.rho[-1])**(2/3) - 1

def compute_delta_w(sol: SolutionGeneralModel):
    a = compute_w(sol)
    b = compute_redshift(sol)
    c = compute_redshift_q(sol)
    return (a + 1) * ((1 + c)/(1 + b))**6

def analyze(sol: SolutionGeneralModel):
    return {
        "w": compute_w(sol),
        "y": compute_y(sol),
        "redshift": compute_redshift(sol),
        "redshift_q": compute_redshift_q(sol),
        "delta_w": compute_delta_w(sol)
    }
