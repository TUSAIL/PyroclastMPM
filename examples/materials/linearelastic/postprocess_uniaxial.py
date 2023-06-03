#%%
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import glob
import os
import pandas as pd 

import tomllib

plt.rcParams['text.usetex'] = True


plt.style.use('science')


import plot_utils

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)


stresses = np.load(config['uniaxial']['output_directory'] + "stress.npy")
deformation_matrices = np.load(config['uniaxial']['output_directory'] + "F.npy")
strain = deformation_matrices - np.identity(3)


plot_utils.plot_stress_strain_components(
    stresses,
    strain,
    0,0,
    "./plots/uniaxial/strain_stress_curve.png",
    "Uniaxial strain-stress curve (component 11)"
    )

plot_utils.q_p_plot(
    stresses, 
    "./plots/uniaxial/q-p.png",
    "Uniaxial q-p plot")
# %%
