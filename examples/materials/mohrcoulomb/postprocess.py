# BSD 3-Clause License
# Copyright (c) 2023, Retief Lubbe
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this
#  list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import tomli

import matplotlib.pyplot as plt
import numpy as np
import plot_utils

plt.rcParams["text.usetex"] = True


# load config file
with open("./config.toml", "rb") as f:
    config = tomli.load(f)


"""
1. Uniaxial loading conditions
"""

stresses = np.load(config["uniaxial"]["output_directory"] + "stress.npy")
deformation_matrices = np.load(
    config["uniaxial"]["output_directory"] + "F.npy"
)

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
deformation_matrices = np.load(
    config["simpleshear"]["output_directory"] + "F.npy"
)


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

plot_utils.q_p_plot(
    stresses, "./plots/simpleshear/q-p.png", "Uniaxial q-p plot"
)


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
