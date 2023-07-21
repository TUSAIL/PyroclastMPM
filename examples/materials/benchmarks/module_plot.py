import matplotlib.pyplot as plt
import numpy as np

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

linestyles = ["-.", "--", "-.", ":"]
markers = [
    "s",
    "o",
    "s",
    "v",
    "^",
    "+",
]

plt.style.use(["seaborn-whitegrid"])

plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 700})


plt.set_cmap("gist_rainbow")


def shear_volume_strain_plot(strain_list, names_list, file, title):
    """Plot shear strain over volumetric strain

    Parameters
    ----------
    strain_list : list
        List of strain tensors for several simulations
    names_list : list
        List of names for each simulation
    file : str
        Name of the output file
    title : _type_
        Title of the plot
    """ """"""
    _, ax = plt.subplots()

    for i, name in enumerate(names_list):
        strains = strain_list[i]

        # volumetric strain
        eps_v_values = strains[:, 0, 0] + strains[:, 1, 1] + strains[:, 2, 2]

        # Deviatoric stress
        dev_strain_values = (
            strains
            - (1.0 / 3.0) * np.identity(3) * eps_v_values[:, None, None]
        )

        shear_strain_values = list(
            map(
                lambda s: np.sqrt(1 * 0.5 * np.trace(s @ s.T)),
                dev_strain_values,
            )
        )

        ax.scatter(eps_v_values, shear_strain_values, label=name)

    ax.set_title(title)
    ax.set_xlabel(r"$\epsilon_v$")
    ax.set_ylabel(r"$\epsilon_d$ ")
    plt.legend()
    plt.savefig(file)
    plt.clf()


def q_p_plot(stress_list, names_list, file, title):
    """Q-p plot

    Parameters
    ----------
    stress_list : list
        List of stress tensors for several simulations
    file : str
        output file name
    title : str
        title of the plot
    """

    _, ax = plt.subplots()

    for i, name in enumerate(names_list):
        stresses = stress_list[i]
        # hydrostatic pressure
        # compressive pressure is negative
        p_values = (1 / 3.0) * (
            stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2]
        )

        # Deviatoric stress
        dev_stress_values = stresses - np.identity(3) * p_values[:, None, None]

        # von Mises effective stress
        q_values = list(
            map(
                lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                dev_stress_values,
            )
        )

        ax.scatter(p_values, q_values, label=name)
    ax.set_title(title)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$q$ ")
    plt.legend()
    plt.savefig(file)

    plt.clf()


def plot_component_step_subplot(
    data_list, names_list, style_id_tuples, file, title, ptype="stress"
):
    """Plot the stress/strain/velocity gradient components in a subplot

    Note velocity gradient can either mean the actual velocity gradient
    (finite strain control) or strain increment (infinite strain control)

    Parameters
    ----------
    data_list : list
        list of stress/strain/velocity gradient data for each simulation
    names_list : list
        list of names for each simulation
    file : str
        output file name
    title : str
        title of the plot
    ptype : str, optional
        plot type, "stress", "strain", "velgrad", by default "stress"
    """
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=False)
    fig.suptitle(title)

    for di, name in enumerate(names_list):
        data = np.array(data_list[di])
        steps = np.arange(len(data))
        marker_id, color_id = style_id_tuples[di]
        for i in range(3):
            for j in range(3):
                ax[i, j].plot(
                    steps,
                    data[:, i, j],
                    markevery=10,
                    markersize=4,
                    label=name,
                    ls=linestyles[marker_id],
                    color=colors[color_id],
                    marker=markers[marker_id],
                )
                nx = i + 1  # x component of stress/strain component
                ny = j + 1  # y component of stress/strain component
                if ptype == "stress":
                    ax[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$")
                elif ptype == "strain":
                    ax[i, j].set_ylabel(f"$\epsilon_{{{nx}{ny}}}$")
                elif ptype == "velgrad":
                    ax[i, j].set_ylabel(f"$\Delta \epsilon_{{{nx}{ny}}}$")

    plt.tight_layout()

    plt.locator_params(axis="both", nbins=5)
    plt.figlegend(names_list, loc="lower center", bbox_to_anchor=(0.55, -0.14))

    plt.savefig(file, bbox_inches="tight")
    plt.clf()


def plot_component_vs(
    data1_list,
    data2_list,
    names_list,
    style_id_tuples,
    file,
    title,
    ptype1="stress",
    ptype2="strain",
):
    """Plot two components against each other

    Parameters
    ----------
    data1_list : list
        First data sets [(N,3,3),...]
    data2_list : list
        Second dataset [(N,3,3),...] (same length as data1_list)
    names_list : list
        names for each simulation
    style_id_tuples : list
        list of tuples (marker_id, color_id) for each simulation
    file : str
        output file name
    title : str
        title of the plot
    ptype1 : str, optional
        type of data for data1, by default "stress"
    ptype2 : str, optional
        type of data for data2, by default "strain"
    """
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    fig.suptitle(title)

    for di, name in enumerate(names_list):
        data1 = np.array(data1_list[di])
        data2 = np.array(data2_list[di])
        marker_id, color_id = style_id_tuples[di]

        for i in range(3):
            for j in range(3):
                x = data1[:, i, j]
                y = data2[:, i, j]
                axs[i, j].plot(
                    x,
                    y,
                    markevery=10,
                    markersize=4,
                    label=name,
                    ls=linestyles[marker_id],
                    color=colors[color_id],
                    marker=markers[marker_id],
                )
                nx = i + 1  # x component of stress/strain component
                ny = j + 1  # y component of stress/strain component
                if ptype1 == "stress":
                    axs[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$")
                elif ptype1 == "strain":
                    axs[i, j].set_ylabel(f"$\epsilon_{{{nx}{ny}}}$")
                elif ptype1 == "F":
                    axs[i, j].set_ylabel(f"$F_{{{nx}{ny}}}$")
                elif ptype1 == "velgrad":
                    axs[i, j].set_ylabel(f"$L_{{{nx}{ny}}}$")

                if ptype2 == "stress":
                    axs[i, j].set_xlabel(f"$\sigma_{{{nx}{ny}}}$")
                elif ptype2 == "strain":
                    axs[i, j].set_xlabel(f"$\epsilon_{{{nx}{ny}}}$")
                elif ptype2 == "F":
                    axs[i, j].set_xlabel(f"$F_{{{nx}{ny}}}$")
                elif ptype2 == "velgrad":
                    axs[i, j].set_xlabel(f"$L_{{{nx}{ny}}}$")

    plt.tight_layout()

    plt.locator_params(axis="both", nbins=5)
    plt.figlegend(names_list, loc="lower center", bbox_to_anchor=(0.55, -0.14))

    plt.savefig(file, bbox_inches="tight")
    plt.clf()


def volume_plot(
    strain_list, stress_list, names_list, style_id_tuples, file, over="p"
):
    _, ax = plt.subplots()
    for di, name in enumerate(names_list):
        stresses = stress_list[di]
        strain = strain_list[di]
        marker_id, color_id = style_id_tuples[di]

        p_values = (1 / 3.0) * (
            stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2]
        )  # hydrostatic pressure

        # Deviatoric stress
        dev_stress_values = (
            stresses - np.identity(3) * p_values[:, None, None]
        )  # deviatoric stress

        q_values = list(
            map(
                lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                dev_stress_values,
            )
        )

        vol = np.trace(strain, axis1=1, axis2=2) / 3.0

        if over == "p":
            ax.plot(
                np.log(-1 * p_values),
                vol,
                label=name,
                markevery=10,
                markersize=4,
                ls=linestyles[marker_id],
                color=colors[color_id],
                marker=markers[marker_id],
            )
            ax.set_xlabel(r"ln $p$ ")
        elif over == "q":
            ax.scatter(
                q_values,
                vol,
                label=name,
                color=colors[color_id],
                marker=markers[marker_id],
            )
            ax.set_xlabel(r"$q$ ")

        ax.set_ylabel(r"$\epsilon_v$ ")
    plt.legend()
    plt.savefig(
        file,
        bbox_inches="tight",
    )
    plt.clf()
