#%%
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import scienceplots


import glob
import os
import pandas as pd 

import tomllib

import plot_utils

plt.rcParams['text.usetex'] = True


# plt.style.use('science')

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)


stresses = np.load(config['simpleshear']['output_directory'] + "stress.npy")
deformation_matrices = np.load(config['simpleshear']['output_directory'] + "F.npy")

strain_list = []
for F in deformation_matrices:
    strain_list.append(0.5*(F.T + F) - np.identity(3))
    
strain = np.array(strain_list)



plot_utils.plot_stress_strain_components(
    stresses,
    strain,
    0,1,
    "./plots/simpleshear/strain_stress_curve.png",
    "Simple shear strain-stress curve (component 12)"
    )

