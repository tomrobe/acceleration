import matplotlib.pyplot as plt
import numpy as np

from .analysis import general_model as analysis_general

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "text.usetex": True,
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
import matplotlib as mpl

# Generic plot function
def plot_subplots(solutions, subplot_parameter, line_parameters, x_func, y_func, x_label="", y_label="", figsize=None, filename=None):

    def get_parameter(sol, parameter_path):
        parts = parameter_path.split(".")
        obj = sol.config
        for part in parts:
            obj = getattr(obj, part)
        return obj

    subplot_values = sorted(set(get_parameter(sol, subplot_parameter) for sol in solutions))

    groups = {}
    for sol in solutions:
        key = get_parameter(sol, subplot_parameter)
        if key not in groups:
            groups[key] = []
        groups[key].append(sol)

    n = len(subplot_values)
    if figsize is None:
        figsize = (4,1.5*n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharey=True)

    if n == 1:
        axes = [axes]

    for ax, subplot_val in zip(axes, subplot_values):

        colormap = plt.get_cmap("Greys_r")
        n_lines = len(groups[subplot_val])
        values = np.linspace(0.0,0.8,n_lines)
        colors = [colormap(v) for v in values]

        for sol, color in zip(groups[subplot_val], colors):
            label = ", ".join(
                f"{p.split('.')[-1]}={get_parameter(sol, p):.3g}"
                for p in line_parameters
            )
            
            ax.plot(x_func(sol), y_func(sol), label=label, color=color, alpha=0.8)

        subplot_name = subplot_parameter.split(".")[-1]

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=250, format='pdf')
    plt.show()

## Specialized plot functions

def plot_phasespace(solutions, figsize=None):
    subplot_parameter = "parameters.m"
    line_parameters = ["init.drho_0","init.theta_0"]
#    x_func = sol.theta
#    y_func = analysis_general.compute_y(sol)
    x_label = r'$x$'
    y_label = r'$y$'
    filename = "phasespace"

    def get_parameter(sol, parameter_path):
        parts = parameter_path.split(".")
        obj = sol.config
        for part in parts:
            obj = getattr(obj, part)
        return obj

    subplot_values = sorted(set(get_parameter(sol, subplot_parameter) for sol in solutions))

    groups = {}
    for sol in solutions:
        key = get_parameter(sol, subplot_parameter)
        if key not in groups:
            groups[key] = []
        groups[key].append(sol)

    n = len(subplot_values)
    if figsize is None:
        figsize = (4,2*n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharey=True)

    if n == 1:
        axes = [axes]

    for ax, subplot_val in zip(axes, subplot_values):

        colormap = plt.get_cmap("Greys_r")
        n_lines = len(groups[subplot_val])
        values = np.linspace(0.0,0.8,n_lines)
        colors = [colormap(v) for v in values]

        for sol, color in zip(groups[subplot_val], colors):
            label = ", ".join(
                f"{p.split('.')[-1]}={get_parameter(sol, p):.3g}"
                for p in line_parameters
            )
            
            x_func = sol.theta
            y_func = analysis_general.compute_y(sol)
            ax.plot(x_func, y_func, label=label, color=color, alpha=0.8)

        subplot_name = subplot_parameter.split(".")[-1]
        ax.set_ylabel(y_label)

    axes[-1].set_xlabel(x_label)

    plt.tight_layout()
    if filename:
        plt.savefig(filename+'.pdf', dpi=250, format='pdf')
    plt.show()

def plot_deltaw(solutions, figsize=None):
    subplot_parameter = "parameters.m"
    line_parameters = ["init.drho_0","init.theta_0"]
#    x_func = sol.theta
#    y_func = analysis_general.compute_y(sol)
    x_label = r'$z$'
    y_label = r''
    filename = "deltaw"

    def get_parameter(sol, parameter_path):
        parts = parameter_path.split(".")
        obj = sol.config
        for part in parts:
            obj = getattr(obj, part)
        return obj

    subplot_values = sorted(set(get_parameter(sol, subplot_parameter) for sol in solutions))

    groups = {}
    for sol in solutions:
        key = get_parameter(sol, subplot_parameter)
        if key not in groups:
            groups[key] = []
        groups[key].append(sol)

    n = len(subplot_values)
    if figsize is None:
        figsize = (4,2*n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    if n == 1:
        axes = [axes]

    for ax, subplot_val in zip(axes, subplot_values):

        colormap = plt.get_cmap("Greys_r")
        n_lines = len(groups[subplot_val])
        values = np.linspace(0.0,0.8,n_lines)
        colors = [colormap(v) for v in values]
        linestyle='--'

        for sol, color in zip(groups[subplot_val], colors):
            label = ", ".join(
                f"{p.split('.')[-1]}={get_parameter(sol, p):.3g}"
                for p in line_parameters
            )
            
            x_func = analysis_general.compute_redshift(sol)
            y_func = analysis_general.compute_delta_w(sol)
            ax.plot(x_func, y_func, label=label, linestyle=linestyle, color=color, alpha=0.8)

        subplot_name = subplot_parameter.split(".")[-1]
        ax.set_ylabel(y_label)

    axes[0].set_xlim(3.0,10.0)
    axes[0].set_ylim(-2 * 10**4, 2 * 10**4)
    axes[0].set_yticks([0])

    axes[1].set_xlim(3.0,10.0)
    axes[1].set_ylim(-2 * 10**4, 2 * 10**5)
    axes[1].set_yticks([0])

    axes[2].set_xlim(3.0,10.0)
    axes[2].set_ylim(-10**7, 10**6)
    axes[2].set_yticks([0])
    axes[-1].set_xlabel(x_label)

    plt.tight_layout()
    if filename:
        plt.savefig(filename+'.pdf', dpi=250, format='pdf')
    plt.show()

def plot_w(solutions, figsize=None):
    subplot_parameter = "parameters.m"
    line_parameters = ["init.drho_0","init.theta_0"]
#    x_func = sol.theta
#    y_func = analysis_general.compute_y(sol)
    x_label = r'Relational Time'
    y_label = r'$w$'
    filename = "w"

    def get_parameter(sol, parameter_path):
        parts = parameter_path.split(".")
        obj = sol.config
        for part in parts:
            obj = getattr(obj, part)
        return obj

    subplot_values = sorted(set(get_parameter(sol, subplot_parameter) for sol in solutions))

    groups = {}
    for sol in solutions:
        key = get_parameter(sol, subplot_parameter)
        if key not in groups:
            groups[key] = []
        groups[key].append(sol)

    n = len(subplot_values)
    if figsize is None:
        figsize = (4,1.5*n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    if n == 1:
        axes = [axes]

    for ax, subplot_val in zip(axes, subplot_values):

        colormap = plt.get_cmap("Greys_r")
        n_lines = len(groups[subplot_val])
        values = np.linspace(0.0,0.8,n_lines)
        colors = [colormap(v) for v in values]

        for sol, color in zip(groups[subplot_val], colors):
            label = ", ".join(
                f"{p.split('.')[-1]}={get_parameter(sol, p):.3g}"
                for p in line_parameters
            )
            
            x_func = sol.t
            y_func = analysis_general.compute_w(sol)
            ax.plot(x_func, y_func, label=label, color=color, alpha=0.8)

        subplot_name = subplot_parameter.split(".")[-1]
        ax.set_ylabel(y_label)
        ax.set_yticks([-1,1])
        axes[-1].set_xlabel(x_label)

    plt.tight_layout()
    if filename:
        plt.savefig(filename+'.pdf', dpi=250, format='pdf')
    plt.show()
