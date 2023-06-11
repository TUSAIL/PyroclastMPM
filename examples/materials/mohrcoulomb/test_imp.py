# %%

import tomllib

import matplotlib.pyplot as plt
import numpy as np
import plot_utils


from matplotlib import patheffects


plt.rcParams["text.usetex"] = True


# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)


"""
1. Uniaxial loading conditions
"""

stresses = np.load(config["uniaxial"]["output_directory"] + "stress.npy")
deformation_matrices = np.load(config["uniaxial"]["output_directory"] + "F.npy")

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


# %%
fig, ax = plt.subplots(figsize=(6, 6))

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
delta = 50
xrange = np.linspace(-35000, 35000, delta)
yrange = np.linspace(-35000, 35000, delta)
x, y = np.meshgrid(xrange, yrange)


eq0 = mohr_coulomb0(x, y, -3000)

plt.imshow(eq1, cmap="hot")
plt.legend()
# # ax.contour(x, y, eq0, [0])
# # eq1 = mohr_coulomb1(x, y, -3000)
# # ax.contour(x, y, eq1, [0])
# # eq2 = mohr_coulomb2(x, y, -3000)
# # ax.contour(x, y, eq2, [0])
# # eq3 = mohr_coulomb3(x, y, -3000)
# # ax.contour(x, y, eq3, [0])
# # eq4 = mohr_coulomb4(x, y, -3000)
# # ax.contour(x, y, eq4, [0])
# # eq5 = mohr_coulomb5(x, y, -3000)
# # ax.contour(x, y, eq5, [0])


# print(np.shape(eq1))
# ax.fill_between(xrange, eq0, eq1, where=yrange > eq0, color="red", alpha=0.5)

# print(x, y)
# plt.setp(ax.collections, path_effects=[patheffects.withTickedStroke(spacing=7)])

# plt.show()


# cg1 = ax.contour(x1, x2, g1, [0], colors='sandybrown')
# plt.setp(cg1.collections,
#          path_effects=[patheffects.withTickedStroke(angle=135)])

# cg2 = ax.contour(x1, x2, g2, [0], colors='orangered')
# plt.setp(cg2.collections,
#          path_effects=[patheffects.withTickedStroke(angle=60, length=2)])

# cg3 = ax.contour(x1, x2, g3, [0], colors="mediumblue")


# %%
# plt.show()

# %


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
    res=25,
)

plot_utils.plot_stress_subplot(
    stresses,
    range(stresses.shape[0]),
    "./plots/uniaxial/strain_stress",
    "Uniaxial strain-stress curves",
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
    res=25,
)

plot_utils.plot_stress_subplot(
    stresses,
    range(stresses.shape[0]),
    "./plots/simpleshear/strain_stress",
    "Simple shear strain-stress curves",
)

# %%
