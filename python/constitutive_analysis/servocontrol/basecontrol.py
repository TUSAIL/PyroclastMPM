import numpy as np
import pyroclastmpm.MPM3D as pm


class BaseControl:
    def setup_material(self):
        self.particles = pm.ParticlesContainer([[0.0, 0.0, 0.0]])

        self.particles.volumes_original = [self.volume]

        self.particles.volume = [self.volume]

        self.particles.stresses = [
            np.array(
                [
                    [-self.confine, 0.0, 0.0],
                    [0.0, -self.confine, 0.0],
                    [0.0, 0.0, -self.confine],
                ]
            )
        ]

        if self.material_type == "ModifiedCamClay":
            self.material = pm.ModifiedCamClay(**self.material_params)

        self.particles, _ = self.material.initialize(self.particles, 0)

    def __init__(
        self,
        material_type,
        material_params,
        volume,
        confine,
        is_finite_strain=False,
        is_staged=False,
        output_step=1,
        verbose=False,
        opt_options={},
    ):
        self.confine = confine
        self.volume = volume
        self.material_type = material_type
        self.material_params = material_params

        self.setup_material()

        self.is_finite_strain = is_finite_strain
        self.is_staged = is_staged
        self.mode = None
        self.output_step = output_step
        self.verbose = verbose
        self.opt_options = opt_options

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

    from .plotting import (
        plot_dvs_ln_p,
        plot_q_eps1,
        plot_q_p,
        plot_sigma1_eps1,
        plot_strain_grid,
        plot_stress_grid,
        plot_tau_gamma,
    )


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
