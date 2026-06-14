import json
import numpy as np
from pathlib import Path

from .models.common import PhysicalParameters
from .models.general_model import InitialConditionsGeneralModel, ConfigurationGeneralModel

from .solvers import general_solver

def load(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def config_dict_to_simulation(config_dict, solver_type):
    parameters = PhysicalParameters(**config_dict["base_parameters"])

    t_span = (
        config_dict["time_span"]["start"],
        config_dict["time_span"]["end"]
    )
    t_eval = np.linspace(t_span[0], t_span[1],config_dict["time_span"]["n_points"])

    if solver_type == "general_model":
        init = InitialConditionsGeneralModel(**config_dict["initial_conditions"])
        return ConfigurationGeneralModel(
            parameters = parameters,
            init = init,
            t_span = t_span,
            t_eval = t_eval,
            name = config_dict["name"],
        )
    else:
        raise ValueError(f"Unknown solver_type: '{solver_type}'")

def run(config_file, output_dir=Path("results")):
    # Load
    config = load(config_file)

    solver_type = config.get("solver_type")
    config_base = config_dict_to_simulation(config, solver_type)

    sweep_type = config["sweep_type"]

    if sweep_type == "single":
        configs = [config_base]
    else:
        raise ValueError(f"Unknown sweep_type: '{sweep_type}'")

    # Solve
    if solver_type == "general_model":
        solutions = general_solver.solve_multiple(configs)
    else:
        solutions = False

    return solutions
