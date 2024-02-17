import numpy as np


def get_a(Pt, beta, Pc):
    a = (Pc + Pt) / (1 + beta)
    return a


def get_b(Pt, beta, a, P):
    if P >= Pt - a:
        return 1.0

    return beta


def yield_surface_q_of_p(M, Pt, beta, Pc, P):
    a = get_a(Pt, beta, Pc)

    b = get_b(Pt, beta, a, P)
    # print(a,b)

    # Yield surface as q of function p
    # (q/M)**2
    pow_q_M = a**2 - (1.0 / (b**2)) * (P - Pt + a) ** 2

    # taking q ~ P so positive roots only
    q = M * np.sqrt(pow_q_M)
    return q


def get_CSL_Line(P, M, Pt):
    q = -M * (P - Pt)
    return q


def plot_strain_grid(
    self, fig_ax=None, file=None, plot_options={}, savefig_option={}
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    else:
        fig, axes = fig_ax

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                self.step_list, self.strain_list[:, i, j], **plot_options
            )

            if i == 2:
                axes[i, j].set_xlabel("step")
            if j == 0:
                axes[i, j].set_ylabel(f"$\\varepsilon_{{{i+1}{j+1}}}$")

    plt.tight_layout()
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_stress_grid(
    self,
    fig_ax=None,
    file=None,
    normalize_stress=1.0e6,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    else:
        fig, axes = fig_ax

    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                self.step_list,
                self.stress_list[:, i, j] / normalize_stress,
                **plot_options,
            )

            if i == 2:
                axes[i, j].set_xlabel("step")
            if j == 0:
                axes[i, j].set_ylabel(f"$\sigma_{{{i+1}{j+1}}}$ {units}")

    plt.tight_layout()
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_q_p(
    self,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        self.pressure_list / normalize_stress,
        self.q_vm_list / normalize_stress,
        **plot_options,
    )

    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel(f"$p$ {units}")
    ax.set_ylabel(f"$q$ {units}")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_q_eps1(
    self,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        -self.strain_list[:, 0, 0],
        self.q_vm_list / normalize_stress,
        **plot_options,
    )

    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel("$\\varepsilon_1$ (-)")
    ax.set_ylabel(f"$q$ {units}")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_tau_gamma(
    self,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        self.gamma_list,
        self.tau_list / normalize_stress,
        **plot_options,
    )

    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel("$\gamma$ (-)")
    ax.set_ylabel(f"$\\tau$ {units}")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_sigma1_eps1(
    self,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        -self.strain_list[:, 0, 0],
        -self.stress_list[:, 0, 0] / normalize_stress,
        **plot_options,
    )
    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_ylabel(f"$\sigma_1$ {units}")
    ax.set_xlabel("$\\varepsilon_1$ (-)")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_dvs_ln_p(
    self,
    vs0=2.0,
    fig_ax=None,
    normalize_stress=1.0e6,
    file=None,
    plot_options={},
    savefig_option={},
):
    import matplotlib.pyplot as plt

    if fig_ax is None:
        fig, ax = plt.subplots()

    vs_prev = vs0
    eps_v_0 = 0.0
    de_list = []
    for eps_v in self.volumetric_strain_list:
        delta_eps = eps_v - eps_v_0

        vs = vs_prev * (1.0 - delta_eps)
        # print(vs,eps_v)

        void_ratio = vs - 1.0
        de_list.append(-((vs0 - 1.0) - void_ratio))
        eps_v_0 = eps_v
        vs_prev = vs

    ax.plot(
        np.log(-self.stress_list[:, 0, 0] / normalize_stress),
        de_list,
        **plot_options,
    )
    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel(f"ln $\sigma_1$ {units}")
    ax.set_ylabel("$\Delta e$")
    if file is not None:
        plt.savefig(file, **savefig_option)


def plot_density_ln_p(
    self,
    normalize_stress=1.0e6,
    ax=None,
    file=None,
    label=None,
    density=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        np.log(-self.stress_list[:, 0, 0] / normalize_stress),
        -self.volumetric_strain_list,
        lw=3,
        ls="--",
        label=label,
    )
    units = ""
    if np.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    ax.set_xlabel(f"ln $\sigma_1$ {units}")
    ax.set_ylabel("$\Delta V / V$")
