import matplotlib.pyplot as plt
import numpy as np


def set_style():
    # print(plt.style.available)
    plt.style.use(["seaborn-v0_8-poster"])

    plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 700})

    plt.set_cmap("gist_rainbow")


def plot_tensor_components(tensor_list, *argv, **kwargs):
    axes = kwargs.pop("axes", None)
    if axes is None:
        _, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)

    steps = np.arange(len(tensor_list))
    dim = kwargs.pop("dim", 3)
    for i in range(dim):
        for j in range(dim):
            axes[i, j].plot(steps, tensor_list[:, i, j], *argv, **kwargs)
    return axes


def plot_triax_soilmechanics_set(
    stress_list, strain_list, control_index, *argv, **kwargs
):
    axes = kwargs.pop("axes", None)

    if axes is None:
        _, axes = plt.subplots(nrows=2, ncols=2)

    p_values = (1 / 3.0) * (
        stress_list[:, 0, 0] + stress_list[:, 1, 1] + stress_list[:, 2, 2]
    )

    # Deviatoric stress
    dev_stress_values = stress_list - np.identity(3) * p_values[:, None, None]

    # von Mises effective stress
    q_values = np.array(
        list(
            map(
                lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                dev_stress_values,
            )
        )
    )

    q_values /= 1e3
    p_values /= -1e3  # swap sign (since we use tension as negative)
    axes[0, 0].plot(p_values, q_values, *argv, **kwargs)

    axes[0, 0].set_xlabel("p [kPa]")
    axes[0, 0].set_ylabel("q [kPa]")

    axes[0, 1].plot(
        -strain_list[:, control_index, control_index],
        q_values,
        *argv,
        **kwargs
    )
    axes[0, 1].set_xlabel(r"$\epsilon_{11}$")
    axes[0, 1].set_ylabel("q [kPa]")

    volume = np.trace(strain_list, axis1=1, axis2=2)
    axes[1, 0].plot(p_values, volume, *argv, **kwargs)
    axes[1, 0].set_xlabel(r"$p [kPa]$")
    axes[1, 0].set_ylabel(r"$v$")

    axes[1, 1].plot(
        -strain_list[:, control_index, control_index], volume, *argv, **kwargs
    )
    axes[1, 1].set_xlabel("\sigma_{11} [kPa]")
    axes[1, 1].set_ylabel(r"$v$")
    return axes


def plot_stress_plastic_strain(
    stress_list,
    strain_list,
    elastic_strain_list,
    control_index,
    *argv,
    **kwargs
):
    axes = kwargs.pop("axes", None)
    if axes is None:
        _, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    p_values = (1 / 3.0) * (
        stress_list[:, 0, 0] + stress_list[:, 1, 1] + stress_list[:, 2, 2]
    )

    # Deviatoric stress
    dev_stress_values = stress_list - np.identity(3) * p_values[:, None, None]

    # von Mises effective stress
    q_values = np.array(
        list(
            map(
                lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                dev_stress_values,
            )
        )
    )

    q_values /= 1e3
    p_values /= -1e3  # swap sign (since we use tension as negative)

    # eps_p_list = strain_list - elastic_strain_list
    # eps_p = np.trace(eps_p_list, axis1=1, axis2=2)
    # eps_v = np.trace(strain_list, axis1=1, axis2=2)
    eps_v = np.trace(strain_list, axis1=1, axis2=2)
    axes[0].plot(
        -strain_list[:, control_index, control_index],
        q_values,
        *argv,
        **kwargs
    )
    axes[0].set_ylabel("q [kPa]")
    axes[0].set_xlabel(r"$\epsilon_1$")

    axes[1].plot(
        -strain_list[:, control_index, control_index], eps_v, *argv, **kwargs
    )
    axes[1].set_xlabel(r"$\epsilon_1$")
    axes[1].set_ylabel(r"$\epsilon_v$")
    return axes
