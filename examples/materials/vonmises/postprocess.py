# %%

import tomllib

import matplotlib.pyplot as plt
import numpy as np
import plot_utils

plt.rcParams["text.usetex"] = True


# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)


"""
1. Uniaxial loading conditions
"""

stresses = np.load(config["uniaxial"]["output_directory"] + "stress.npy")
deformation_matrices = np.load(config["uniaxial"]["output_directory"] + "F.npy")


strain_list = []
for F in deformation_matrices:
    strain_list.append(0.5 * (F.T + F) - np.identity(3))

strain = np.array(strain_list)


plot_utils.plot_stress_strain_components(
    stresses,
    strain,
    0,
    0,
    "./plots/uniaxial/strain_stress_curve.png",
    "Uniaxial strain-stress curve (component 11)",
)

plot_utils.q_p_plot(stresses, "./plots/uniaxial/q-p.png", "Uniaxial q-p plot")


plot_utils.plot_principal(
    stresses,
    "./plots/uniaxial/principal.png",
    "Uniaxial compression - principle stresses plot",
)

plot_utils.plot_stress_subplot(
    stresses,
    range(stresses.shape[0]),
    "./plots/uniaxial/strain_stress",
    "Simple shear strain-stress curves",
)

"""
1. simple shear loading conditions
"""

stresses = np.load(config["simpleshear"]["output_directory"] + "stress.npy")
deformation_matrices = np.load(config["simpleshear"]["output_directory"] + "F.npy")


strain_list = []
for F in deformation_matrices:
    strain_list.append(0.5 * (F.T + F) - np.identity(3))
strain = np.array(strain_list)


plot_utils.plot_stress_strain_components(
    stresses,
    strain,
    0,
    1,
    "./plots/simpleshear/strain_stress_curve.png",
    "Simple shear strain-stress curve (component 12)",
)

plot_utils.q_p_plot(stresses, "./plots/simpleshear/q-p.png", "Uniaxial q-p plot")


plot_utils.plot_principal(
    stresses,
    "./plots/simpleshear/principal.png",
    "Simple shear - principle stresses plot",
    res=50,
)

plot_utils.plot_stress_subplot(
    stresses,
    range(stresses.shape[0]),
    "./plots/simpleshear/strain_stress",
    "Simple shear strain-stress curves",
)
