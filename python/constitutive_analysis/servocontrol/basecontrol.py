#%%


import numpy as np
import pyroclastmpm.MPM3D as pm



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

        if is_staged:
            self.prestrain = np.array(self.particles.F[0])
        else:
            if is_finite_strain:
                self.prestrain = np.identity(3)
            else:
                self.prestrain = np.zeros((3, 3))

        self.strain_prev = self.prestrain
        self.particles.F = [self.prestrain]
        print(self.strain_prev)

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

        stress = np.array(self.particles.stresses[0]) * self.sign_flip
        strain = np.array(self.particles.F[0]) * self.sign_flip
            
        self.stress_list.append(stress)
        self.strain_list.append(strain)
        self.step_list.append(step)

        print(f"stress: {stress}")
        print(f"strain: {strain}")
        x = input()
        if self.verbose:
            print(f"storing array step {step}")




    def post_process(self):
        self.stress_list = np.array(self.stress_list)
        self.strain_list = np.array(self.strain_list)

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
                    lambda s: 0.5* np.trace(s @ s.T),
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
            + (1./3)*np.identity(3) * self.volumetric_strain_list[:, None, None]
        )

        self.gamma_list = np.array(
            list(
                map(
                    lambda s: 0.5* np.trace(s @ s.T),
                    self.dev_strain_list,
                )
            )
        )
        print(self.stress_list[:,0,0]/1000000.0)
        print(self.strain_list[:,0,0])



    def plot_strain_grid(self, file=None):
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)

        strain_list_np = np.array(self.strain_list)

        for i in range(3):
            for j in range(3):
                axes[i, j].plot(self.step_list, strain_list_np[:, i, j])

        if file is not None:
            plt.savefig(file)

    def plot_q_p(self, file=None):
        import matplotlib.pyplot as plt

        stress_list_np = np.array(self.stress_list)

        pressure = -(1 / 3.0) * (
            stress_list_np[:, 0, 0]
            + stress_list_np[:, 1, 1]
            + stress_list_np[:, 2, 2]
        )

        dev_stress = stress_list_np + np.identity(3) * pressure[:, None, None]

        q_vomMises = np.array(
            list(
                map(
                    lambda s: np.sqrt(3 * 0.5 * np.trace(s @ s.T)),
                    dev_stress,
                )
            )
        )
        fig, ax = plt.subplots()

        ax.plot(pressure, q_vomMises)
        if file is not None:
            plt.savefig(file)

    def plot_q_eps11(self, file=None):
        import matplotlib.pyplot as plt

        stress_list_np = np.array(self.stress_list)

# %%
