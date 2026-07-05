import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from acceleration.loader import run
from acceleration.analysis.general_model import compute_y
from acceleration.plotting import plot_subplots,plot_phasespace,plot_deltaw,plot_w

solutions = run("configurations/dummy_sweep.json")

plot_subplots(
    solutions,
    subplot_parameter = "parameters.m",
    line_parameters = ["init.drho_0","init.theta_0"],
    x_func = lambda sol: sol.dtheta,
    y_func = lambda sol: compute_y(sol)
)
plot_phasespace(solutions)
plot_deltaw(solutions)
plot_w(solutions)
