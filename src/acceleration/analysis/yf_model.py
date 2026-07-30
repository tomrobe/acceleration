import numpy as np
from scipy.integrate import cumulative_trapezoid


from ..models.yf_model import SolutionYFModel
from ..solvers.yf_solver import eom

# Define desired quantities as function, then include them in the return of analyze()

def compute_derivative(sol: SolutionYFModel):
    p = sol.config.parameters
    dyf = np.array([
        eom(sol.t[i], [sol.y[i], sol.f[i]], p)
        for i in range(len(sol.t))
    ])
    return dyf[:,0], dyf[:,1]

def compute_delta_N(sol: SolutionYFModel):
    integrand = sol.y/sol.f
    temp = (3/2) * cumulative_trapezoid(integrand, sol.t, initial=0)
    return temp

def compute_epsilon1(sol: SolutionYFModel):
    dy = compute_derivative(sol)[0]
    temp = -(2/3) * sol.f * dy / (sol.y**2)
    return temp

def compute_epsilon2(sol: SolutionYFModel):
    p = sol.config.parameters
    dy,df = compute_derivative(sol)
    temp =  6 - sol.f * (2-p.m) * df / (dy * sol.y) + (sol.f * 4 * p.m / dy) + 2 * dy * sol.f / (sol.y**2)
    return -3 * temp / 2

def compute_field(sol: SolutionYFModel):
    eps1 = compute_epsilon1(sol)
    delta_N = compute_delta_N(sol)
    field_integrand = np.sqrt(2 * eps1 / 3)
    field = np.sqrt(3) * cumulative_trapezoid(field_integrand, delta_N, initial=0.0)
    return field

def compute_potential(sol: SolutionYFModel):
    eps1 = compute_epsilon1(sol)
    delta_N = compute_delta_N(sol)
    potential_integrand = compute_epsilon1(sol)
    potential = (1 - 3 * eps1/4) * np.exp(- 9 * cumulative_trapezoid(potential_integrand, delta_N, initial=0.0) / 2)
    return potential

def analyze(sol: SolutionYFModel):
    return {
        "delta_N": compute_delta_N(sol),
        "epsilon1": compute_epsilon1(sol),
        "epsilon2": compute_epsilon1(sol),
        "field": compute_field(sol),
        "potential": compute_potential(sol)
    }
