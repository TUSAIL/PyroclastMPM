import matplotlib.pyplot as plt
import numpy as np


def flip_sign(arr):
    """Flip the sign of an array"""
    return -1.0 * arr


def plot_component_subplot(data, mask, file, title, ptype="stress", *args):
    """Plot the stress/strain/velocity gradient components in a subplot

    Note velocity gradient can either mean the actual velocity gradient
    (finite strain control) or strain increment (infinite strain control)

    Parameters
    ----------
    data : np.array (N,3,3)
        stress/strain/velocity gradient data
    mask : mask : np.array (3,3)
        mask to say if boundary is strain or stress control
    file : str
        output file name
    title : str
        title of the plot
    ptype : str, optional
        plot type, "stress", "strain", "velgrad", by default "stress"
    """
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=False)
    fig.suptitle(title)
    steps = np.arange(len(data))
    for i in range(3):
        for j in range(3):
            axs[i, j].plot(
                steps,
                data[:, i, j],
                color="r",
                marker="o",
                markevery=5,
            )
            nx = i + 1  # x component of stress/strain component
            ny = j + 1  # y component of stress/strain component
            if ptype == "stress":
                axs[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$", *args)
            elif ptype == "strain":
                axs[i, j].set_ylabel(f"$\epsilon_{{{nx}{ny}}}$", *args)
            elif ptype == "velgrad":
                axs[i, j].set_ylabel(f"$\Delta \epsilon_{{{nx}{ny}}}$", *args)

    plt.tight_layout()
    plt.savefig(
        file,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()


def plot_component_vs(
    data1, data2, mask, file, title, ptype1="stress", ptype2="strain", *args
):
    """Plot two components against each other

    Parameters
    ----------
    data1 : np.array
        First data set (N,3,3)
    data2 : np.array
        Second dataset (N,3,3)
    mask : np.array
        mask to say if boundary is strain or stress control (N,3,3)
    file : str
        output file name
    title : str
        title of the plot
    ptype1 : str, optional
        type of data for data1, by default "stress"
    ptype2 : str, optional
        type of data for data2, by default "strain"
    """
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=False)
    fig.suptitle(title)
    for i in range(3):
        for j in range(3):
            x = data1[:, i, j]
            y = data2[:, i, j]
            axs[i, j].plot(
                x,
                y,
                color="r",
                marker="o",
                markevery=5,
                markersize=2,
            )
            nx = i + 1  # x component of stress/strain component
            ny = j + 1  # y component of stress/strain component
            if ptype1 == "stress":
                axs[i, j].set_xlabel(f"$\sigma_{{{nx}{ny}}}$", *args)
            elif ptype1 == "strain":
                axs[i, j].set_xlabel(f"$\epsilon_{{{nx}{ny}}}$", *args)
            elif ptype1 == "F":
                axs[i, j].set_xlabel(f"$F_{{{nx}{ny}}}$", *args)
            elif ptype1 == "velgrad":
                axs[i, j].set_xlabel(f"$L_{{{nx}{ny}}}$", *args)

            if ptype2 == "stress":
                axs[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$", *args)
            elif ptype2 == "strain":
                axs[i, j].set_ylabel(f"$\epsilon_{{{nx}{ny}}}$", *args)
            elif ptype2 == "F":
                axs[i, j].set_ylabel(f"$F_{{{nx}{ny}}}$", *args)
            elif ptype2 == "velgrad":
                axs[i, j].set_ylabel(f"$L_{{{nx}{ny}}}$", *args)

    plt.tight_layout()
    plt.savefig(
        file,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()


def q_p_plot(stresses, file, title, *args, **kwargs):
    """Q-p plot

    Parameters
    ----------
    stresses : np.array
        (3,3) np array of stresses
    file : str
        output file name
    title : str
        title of the plot
    """

    # Hyrdostatic stress
    p_values = -(1 / 3.0) * (
        stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2]
    )  # hydrostatic pressure

    # Deviatoric stress
    dev_stress_values = (
        stresses + np.identity(3) * p_values[:, None, None]
    )  # deviatoric stress

    # effective stress q

    q_values = list(
        map(lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)), dev_stress_values)
    )

    plt.scatter(p_values, q_values, color="blue", marker="s", *args, **kwargs)
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


def volume_plot(stresses, strain, file, title, over="p", *args, **kwargs):
    """Volume plot over p or q"""
    # Hyrdostatic stress
    p_values = -(1 / 3.0) * (
        stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2]
    )  # hydrostatic pressure

    # Deviatoric stress
    dev_stress_values = (
        stresses - np.identity(3) * p_values[:, None, None]
    )  # deviatoric stress

    q_values = list(
        map(lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)), dev_stress_values)
    )

    vol = np.trace(strain, axis1=1, axis2=2) / 3.0

    if over == "p":
        plt.scatter(
            np.log(p_values), vol, color="blue", marker="s", *args, **kwargs
        )
        plt.xlabel(r"ln $p$ ")
    elif over == "q":
        plt.scatter(q_values, vol, color="blue", marker="s", *args, **kwargs)
        plt.xlabel(r"$q$ ")
    plt.ylabel(r"$\epsilon_v$ ")
    plt.title(title)
    plt.grid()
    plt.savefig(
        file,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()
