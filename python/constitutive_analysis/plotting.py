import matplotlib.pyplot as plt
import numpy as np


def plot_stress_strain_components(
    stresses, strain, i, j, file, title, dpi=300
):
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
    plt.savefig(
        file,
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
    )
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
    plt.xlabel(r"$p$")
    plt.ylabel(r"$q$ ")
    plt.grid()
    plt.savefig(
        file,
        # dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()


def give_implicit3D(fn, bbox=(-2.5, 2.5), res=100):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox * 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    A = np.linspace(xmin, xmax, res)  # resolution of the contour
    B = np.linspace(xmin, xmax, res)  # number of slices
    A1, A2 = np.meshgrid(A, A)  # grid on which the contour is plotted

    for z in B:  # plot contours in the XY plane
        X, Y = A1, A2
        Z = fn(X, Y, z)
        ax.contour(X, Y, Z + z, [z], zdir="z")

    for y in B:  # plot contours in the XZ plane
        X, Z = A1, A2
        Y = fn(X, y, Z)
        ax.contour(X, Y + y, Z, [y], zdir="y")

    for x in B:  # plot contours in the YZ plane
        Y, Z = A1, A2
        X = fn(x, Y, Z)
        ax.contour(X + x, Y, Z, [x], zdir="x")
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    return fig, ax


def plot_stress_subplot(stresses, steps, file, title):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=False)
    fig.suptitle(title)
    for i in range(3):
        for j in range(3):
            axs[i, j].scatter(steps, stresses[:, i, j], color="red")
            nx = i + 1  # x component of stress/strain component
            ny = j + 1  # y component of stress/strain component
            axs[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$ ")
            # axs[i,j].set_ylabel(f'$\varepsilon_{{{nx}{ny}}}$ ')
            axs[i, j].grid()

    plt.tight_layout()
    plt.savefig(
        file,
        # dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()
