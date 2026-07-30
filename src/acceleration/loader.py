import json
import numpy as np
from pathlib import Path
import itertools
import copy

from .models.common import PhysicalParameters
from .models.general_model import InitialConditionsGeneralModel, ConfigurationGeneralModel
from .models.yf_model import InitialConditionsYFModel,ConfigurationYFModel

from .solvers import general_solver, yf_solver

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
    elif solver_type == "yf_model":
        init = InitialConditionsYFModel(**config_dict["initial_conditions"])
        return ConfigurationYFModel(
            parameters = parameters,
            init = init,
            t_span = t_span,
            t_eval = t_eval,
            name = config_dict["name"],
        )
    else:
        raise ValueError(f"Unknown solver_type: '{solver_type}'")

def create_sweep(base_config, parameter_specs, paired_parameter_specs):
    def set_parameter(config, path, val):
        parts = path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)

    if paired_parameter_specs is not None:
        paired_paths = list(paired_parameter_specs.keys())
        paired_values = list(paired_parameter_specs.values())

        lengths = [len(v) for v in paired_values]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"all paired parameters must have the same number of values. "
                f"Got lengths: {dict(zip(paired_paths, lengths))}"
            )

        paired_combinations = list(zip(*paired_values))

    else:
        paired_paths = []
        paired_combinations = [()]

    if parameter_specs:
        independent_paths = list(parameter_specs.keys())
        independent_values = list(parameter_specs.values())
        independent_combinations = list(itertools.product(*independent_values))
    else:
        independent_paths = []
        independent_combinations = [()]

    configs = []
    for independent_combo in independent_combinations:
        for paired_combo in paired_combinations:
            config = copy.deepcopy(base_config)

            name_parts = []

            for path, val in zip(independent_paths, independent_combo):
                set_parameter(config, path, val)
                name_parts.append(f"{path.split('.')[-1]}={val:.3g}")

            for path, val in zip(paired_paths, paired_combo):
                set_parameter(config, path, val)
                name_parts.append(f"{path.split('.')[-1]}={val:.3g}")
            
            config.name = ", ".join(name_parts)
            configs.append(config)

    return configs

def run(config_file, output_dir=Path("results")):
    # Load
    config = load(config_file)

    solver_type = config.get("solver_type")
    base_config = config_dict_to_simulation(config, solver_type)

    sweep_type = config["sweep_type"]

    if sweep_type == "single":
        configs = [base_config]
    elif sweep_type == "sweep":
        configs = create_sweep(
            base_config, 
            parameter_specs=config["sweep_config"].get("parameters", {}),
            paired_parameter_specs=config["sweep_config"].get("paired_parameters", None)
        )
    else:
        raise ValueError(f"Unknown sweep_type: '{sweep_type}'")

    # Solve
    if solver_type == "general_model":
        solutions = general_solver.solve_multiple(configs)
    elif solver_type == "yf_model":
        solutions = yf_solver.solve_multiple(configs)
    else:
        solutions = False

    return solutions
