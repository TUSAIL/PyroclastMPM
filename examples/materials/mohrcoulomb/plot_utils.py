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
    plt.xlabel(f"$\epsilon_{{{nx}{ny}}}$")
    plt.ylabel(f"$\sigma_{{{nx}{ny}}}$")
    plt.title(title)
    plt.grid()
    plt.savefig(
        file,
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
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()


def give_implicit(fig, ax, fn, bbox=(-2.5, 2.5), res=100, alpha=0.9):
    """create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval"""
    xmin, xmax, ymin, ymax, zmin, zmax = bbox * 3

    A = np.linspace(xmin, xmax, res)  # resolution of the contour
    B = np.linspace(xmin, xmax, res)  # number of slices
    A1, A2 = np.meshgrid(A, A)  # grid on which the contour is plotted

    for z in B:  # plot contours in the XY plane
        X, Y = A1, A2
        Z = fn(X, Y, z)
        ax.contour(X, Y, Z + z, [z], zdir="z", alpha=alpha)
        # [z] defines the only level to plot for this contour for this value of z

    for y in B:  # plot contours in the XZ plane
        X, Z = A1, A2
        Y = fn(X, y, Z)
        ax.contour(X, Y + y, Z, [y], zdir="y", alpha=alpha)

    for x in B:  # plot contours in the YZ plane
        Y, Z = A1, A2
        X = fn(x, Y, Z)
        ax.contour(X + x, Y, Z, [x], zdir="x", alpha=alpha)

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    return fig, ax


cohesion = 1e4
friction_angle = 15


def mohr_coulomb0(x, y, z):
    global friction_angle, cohesion
    phi = (
        x
        - z
        + (x + z) * np.sin(np.deg2rad(friction_angle))
        - 2 * cohesion * np.cos(np.deg2rad(friction_angle))
    )
    return phi


def mohr_coulomb1(x, y, z):
    global friction_angle, cohesion
    phi = (
        y
        - z
        + (y + z) * np.sin(np.deg2rad(friction_angle))
        - 2 * cohesion * np.cos(np.deg2rad(friction_angle))
    )
    return phi


def mohr_coulomb2(x, y, z):
    global friction_angle, cohesion
    phi = (
        y
        - x
        + (y + x) * np.sin(np.deg2rad(friction_angle))
        - 2 * cohesion * np.cos(np.deg2rad(friction_angle))
    )
    return phi


def mohr_coulomb3(x, y, z):
    global friction_angle, cohesion
    phi = (
        z
        - x
        + (z + x) * np.sin(np.deg2rad(friction_angle))
        - 2 * cohesion * np.cos(np.deg2rad(friction_angle))
    )
    return phi


def mohr_coulomb4(x, y, z):
    global friction_angle, cohesion
    phi = (
        z
        - y
        + (z + y) * np.sin(np.deg2rad(friction_angle))
        - 2 * cohesion * np.cos(np.deg2rad(friction_angle))
    )
    return phi


def mohr_coulomb5(x, y, z):
    global friction_angle, cohesion
    phi = (
        x
        - y
        + (x + y) * np.sin(np.deg2rad(friction_angle))
        - 2 * cohesion * np.cos(np.deg2rad(friction_angle))
    )
    return phi


def plot_principal(stresses, file, title, res=100):
    w_list, v_list = [], []
    for stress in stresses:
        eigen_values, eigen_vectors = np.linalg.eig(stress)

        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        w_list.append(eigen_values)
        v_list.append(eigen_vectors)

    w_list = np.array(w_list)
    v_list = np.array(v_list)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig, ax = give_implicit(
        fig, ax, mohr_coulomb0, bbox=(35000, -22000), res=res
    )
    fig, ax = give_implicit(
        fig, ax, mohr_coulomb1, bbox=(35000, -22000), res=res
    )
    fig, ax = give_implicit(
        fig, ax, mohr_coulomb2, bbox=(35000, -22000), res=res
    )
    fig, ax = give_implicit(
        fig, ax, mohr_coulomb3, bbox=(35000, -22000), res=res
    )
    fig, ax = give_implicit(
        fig, ax, mohr_coulomb4, bbox=(35000, -22000), res=res
    )
    fig, ax = give_implicit(
        fig, ax, mohr_coulomb5, bbox=(35000, -22000), res=res
    )
    ax.scatter(
        w_list[:, 0], w_list[:, 1], w_list[:, 2], color="blue", marker="s"
    )

    ax.set_zlim3d(30000, -30000)
    ax.set_xlim3d(30000, -30000)
    ax.set_ylim3d(30000, -30000)
    ax.view_init(31, 35)

    plt.title(title)
    plt.xlabel(r"$\sigma_1$")
    plt.ylabel(r"$\sigma_2$")
    plt.savefig(
        file,
        transparent=False,
        bbox_inches="tight",
    )

    ax.view_init(31, 45)

    plt.savefig(
        ".".join(file.split(".")[:2]) + "2.png",
        transparent=False,
        bbox_inches="tight",
    )
    ax.view_init(31, 50)

    plt.show()
    plt.clf()


def plot_stress_subplot(stresses, steps, file, title):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=False)
    fig.suptitle(title)
    for i in range(3):
        for j in range(3):
            axs[i, j].scatter(steps, stresses[:, i, j], color="red")
            nx = i + 1  # x component of stress/strain component
            ny = j + 1  # y component of stress/strain component
            axs[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$ ")
            axs[i, j].grid()

    plt.tight_layout()
    plt.savefig(
        file,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()
