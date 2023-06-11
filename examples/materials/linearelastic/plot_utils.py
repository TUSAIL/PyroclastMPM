import matplotlib.pyplot as plt
import numpy as np


def plot_stress_strain_components(stresses, strain, i, j, file, title):
    """Plot the stress components vs strain components

    Args:
        stresses (np.array): array of stress tensors (nsteps, 3, 3)
        strain (np.array): array of strain tensors (nsteps, 3, 3)
        i (int): i index of the stress/strain component to plot
        j (int): j index of the stress/strain component to plot
        file (str): filename to save the plot to
        title (str): title of the plot
    """

    plt.scatter(strain[:, i, j], stresses[:, i, j], color="red")
    nx = i + 1  # x component of stress/strain component
    ny = j + 1  # y component of stress/strain component
    plt.xlabel(f"$\epsilon_{{{nx}{ny}}}$ ")
    plt.ylabel(f"$\sigma_{{{nx}{ny}}}$ ")
    plt.title(title)
    plt.grid()
    plt.savefig(file, dpi=600, transparent=False, bbox_inches="tight")
    plt.clf()


def q_p_plot(stresses, file, title):
    # Hyrdostatic stress
    p_values = (1 / 3.0) * (
        stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2]
    )  # hydrostatic pressure

    # Deviatoric stress
    dev_stress_values = (
        stresses - np.identity(3) * p_values[:, None, None]
    )  # deviatoric stress

    # effective stress q
    q_values = list(
        map(lambda s: np.sqrt(3 * np.trace(s @ s.T)), dev_stress_values)
    )

    plt.scatter(p_values, q_values, color="blue", marker="s")
    plt.title(title)
    plt.xlabel(r"$p$ ")
    plt.ylabel(r"$q$ ")
    plt.grid()
    plt.savefig(file, dpi=600, transparent=False, bbox_inches="tight")
