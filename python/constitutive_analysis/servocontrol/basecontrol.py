import numpy as np
import pyroclastmpm.MPM3D as pm


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


class BaseControl:
    def __init__(
        self,
        particles,
        material,
        tolerance=1e-7,
        optimize_method="krylov",
        is_finite_strain=False,
        is_staged=False,
        output_step=100,
        verbose=False,
    ):
        self.particles = particles
        self.material = material
        self.tolerance = tolerance
        self.optimize_method = optimize_method
        self.is_finite_strain = is_finite_strain
        self.is_staged = is_staged
        self.mode = None
        self.output_step = output_step
        self.verbose = verbose

        self.sign_flip = 1.0  # or -1.0

        self.prestress = np.array(self.particles.stresses[0])

        self.stress_list = []
        self.strain_list = []
        self.step_list = []
        self.volume_list = []

        self.pc_list = []

        if is_staged:
            self.prestrain = np.array(self.particles.F[0])
        else:
            if is_finite_strain:
                self.prestrain = np.identity(3)
            else:
                self.prestrain = np.zeros((3, 3))

        self.strain_prev = self.prestrain
        self.particles.F = [self.prestrain]
        # print(self.strain_prev)

    def set_sign_flip(self, sign_flip):
        """
        Set sign at which multiply stress/ strain tensor (compression/tension convention)
        """
        # TODO change function name
        # SOL 1. (best) eliminate the need for this function by making compresison always positive
        # SOL 2. Rename function...?
        self.sign_flip = sign_flip

    def store_results(self, step):
        """
        Store the results in a list.
        """

        stress = np.array(self.particles.stresses[0])
        strain = np.array(self.particles.F[0])
        volume = np.array(self.particles.volumes[0])
        self.stress_list.append(stress)
        self.strain_list.append(strain)
        self.step_list.append(step)
        self.volume_list.append(volume)

        # if self.material.__class__.__name__ == "DruckerPragerCap":
        pc = self.material.pc_gpu[0]
        self.pc_list.append(pc)

        # print(f"stress: {np.diag(stress)}")
        # print(f"strain: {np.diag(strain)}")

        # if self.verbose:
        # print(f"storing array step {step}, stess {np.diag(stress)}")

    def post_process(self):
        self.stress_list = np.array(self.stress_list)

        self.strain_list = np.array(self.strain_list)

        self.volume_list = np.array(self.volume_list)

        self.pressure_list = -(1 / 3.0) * (
            self.stress_list[:, 0, 0]
            + self.stress_list[:, 1, 1]
            + self.stress_list[:, 2, 2]
        )

        self.dev_stress_list = (
            self.stress_list
            + np.identity(3) * self.pressure_list[:, None, None]
        )

        self.q_vm_list = np.array(
            list(
                map(
                    lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                    self.dev_stress_list,
                )
            )
        )

        self.tau_list = np.array(
            list(
                map(
                    lambda s: 0.5 * np.trace(s @ s.T),
                    self.dev_stress_list,
                )
            )
        )

        self.volumetric_strain_list = -(
            self.strain_list[:, 0, 0]
            + self.strain_list[:, 1, 1]
            + self.strain_list[:, 2, 2]
        )

        self.dev_strain_list = (
            self.strain_list
            + (1.0 / 3)
            * np.identity(3)
            * self.volumetric_strain_list[:, None, None]
        )

        self.gamma_list = np.array(
            list(
                map(
                    lambda s: 0.5 * np.trace(s @ s.T),
                    self.dev_strain_list,
                )
            )
        )

        self.pc_list = np.array(self.pc_list)

        # print(self.stress_list[:, 0, 0] / 1000000.0)
        # print(self.strain_list[:, 0, 0])

    def plot_strain_grid(self, file=None):
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)

        strain_list_np = np.array(self.strain_list)

        for i in range(3):
            for j in range(3):
                axes[i, j].plot(self.step_list, strain_list_np[:, i, j])

        if file is not None:
            plt.savefig(file)

    def plot_q_p(self, normalize_stress=1.0e6, ax=None, file=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            self.pressure_list / normalize_stress,
            self.q_vm_list / normalize_stress,
            ls="-",
            lw=3,
            c="b",
        )

        units = ""
        if np.isclose(normalize_stress, 1.0e6):
            units = "(MPa)"

        ax.set_xlabel(f"$p$ {units}")
        ax.set_ylabel(f"$q$ {units}")
        if file is not None:
            plt.savefig(file)

    def plot_q_eps11(
        self,
        normalize_stress=1.0e6,
        ax=None,
        file=None,
        color="b",
        label=None,
        ls="-",
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            -self.strain_list[:, 0, 0],
            self.q_vm_list / normalize_stress,
            lw=3,
            c=color,
            label=label,
            ls=ls,
        )

        units = ""
        if np.isclose(normalize_stress, 1.0e6):
            units = "(MPa)"

        ax.set_xlabel(f"$\\varepsilon_1$ (-)")
        ax.set_ylabel(f"$q$ {units}")
        if file is not None:
            plt.savefig(file)

    def plot_sigma11_eps11(
        self, normalize_stress=1.0e6, ax=None, file=None, label=None, color="b"
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            -self.strain_list[:, 0, 0],
            -self.stress_list[:, 0, 0] / normalize_stress,
            lw=3,
            ls="--",
            label=label,
            color=color,
        )
        units = ""
        if np.isclose(normalize_stress, 1.0e6):
            units = "(MPa)"

        ax.set_ylabel(f"$\sigma_1$ {units}")
        ax.set_xlabel("$\\varepsilon_1$ (-)")

    def plot_void_ratio_ln_p(
        self,
        normalize_stress=1.0e6,
        ax=None,
        file=None,
        label=None,
        initial_void_ratio=1.0,
        color="b",
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        # Vs0 = 1 / density

        v0 = initial_void_ratio + 1.0
        eps_v_0 = 0.0
        e_list = []
        for eps_v in self.volumetric_strain_list:
            delta_eps = eps_v - eps_v_0

            vs = v0 * (1.0 - delta_eps)
            # print(vs,eps_v)

            void_ratio = vs - 1.0
            e_list.append(-(initial_void_ratio - void_ratio))
            eps_v_0 = eps_v
            v0 = vs

        ax.plot(
            np.log(-self.stress_list[:, 0, 0] / normalize_stress),
            e_list,
            lw=3,
            ls="--",
            label=label,
            color=color,
        )
        units = ""
        if np.isclose(normalize_stress, 1.0e6):
            units = "(MPa)"

        ax.set_xlabel(f"ln $\sigma_1$ {units}")
        ax.set_ylabel("$\Delta e$")

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

        # Vs0 = 1 / density
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

    def plot_q_p_anim(
        self, normalize_stress=1.0e6, ax=None, fig=None, file=None
    ):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        line1 = ax.plot(
            self.pressure_list / normalize_stress,
            self.q_vm_list / normalize_stress,
            c="b",
            ls="-",
            lw=2,
            marker="s",
            ms=2,
            markevery=5,
        )[0]

        units = ""
        if np.isclose(normalize_stress, 1.0e6):
            units = "(MPa)"

        ax.set_xlabel(f"$p$ {units}")
        ax.set_ylabel(f"$q$ {units}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        def update(frame, p_values, q_values, pc_values, material, ax):
            M = material.M
            Pt = material.Pt
            beta = material.beta

            p_yield = np.arange(0, 40)
            q_yield = [
                yield_surface_q_of_p(M, Pt, beta, pc_values[i], P)
                for i, P in enumerate(p_yield)
            ]

            ax.plot(p_yield, q_yield, c="k", lw=2, ls="--")
            # for each frame, update the data stored on each artist.
            x = p_values[:frame]
            y = q_values[:frame]
            # update the scatter plot:
            line1.set_xdata(x)
            line1.set_ydata(y)
            return (line1,)

        anim = animation.FuncAnimation(
            fig=fig,
            func=update,
            fargs=(
                self.pressure_list / normalize_stress,
                self.q_vm_list / normalize_stress,
                self.pc_list / normalize_stress,
                self.material,
                ax,
            ),
            frames=np.shape(self.pressure_list)[0],
            interval=30,
        )
        # writer = animation.PillowWriter(
        # fps=15, metadata=dict(artist="Me"), bitrate=1800
        # )
        return anim


# fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
# line1 = ax.plot(
#     strain_list[:, 1, 1],
#     q_values,
#     c="b",
#     ls="-",
#     lw=2,
#     marker="s",
#     ms=10,
#     markevery=5,
# )[0]
# fig.gca().tick_params(axis="both", which="major", labelsize=30)
# fig.gca().set_ylabel(r"$q \ [kPa]$", fontsize=40)
# fig.gca().set_xlabel(r"$\epsilon_{11}$", fontsize=40)
# fig.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
# fig.gca().yaxis.set_major_locator(plt.MaxNLocator(6))
# fig.set_tight_layout(True)


# def update(frame):
#     x = strain_list[:, 1, 1][:frame]
#     y = q_values[:frame]
#     line1.set_xdata(x)
#     line1.set_ydata(y)
#     return (line1,)  # %%


# anim = animation.FuncAnimation(
#     fig=fig,
#     func=update,
#     frames=np.shape(strain_list)[0],
#     interval=30,
# )
# writer = animation.PillowWriter(
#     fps=15, metadata=dict(artist="Me"), bitrate=1800
# )
# anim.save("plots/mcc/mcc_q_eps11.gif", writer=writer)

# %%
