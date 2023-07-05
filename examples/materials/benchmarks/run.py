import os

import numpy as np
import pyroclastmpm.MPM3D as pm
import tomli
from servo import mixed_control

stress_list, strain_list, velgrad_list, mask_list = [], [], [], []


def callback(particles, material, step):
    """Callback function to store particle and material data

    Parameters
    ----------
    particles : ParticlesContainer
        a single particle containing stress, strain, and velocity gradient, etc.
    material : Material
        material model (e.g. VonMises)
    step : int
        output step
    """
    global stress_list, strain_list, velgrad_list
    stress_list.append(particles.stresses[0])
    # infinite strain tensor / or deformation gradient for finite strain control
    strain_list.append(particles.F[0])
    # strain increment / or velocity gradient for finite strain
    velgrad_list.append(particles.velocity_gradient[0])
    print(f"step: {step}")


def save_data(
    cfg,
    model_name,
    benchmark_name,
    stress_list,
    strain_list,
    velgrad_list,
    mask_list,
):
    """Helper function to save data to a file

    Parameters
    ----------
    cfg : dict
        Dictionary containing the configuration parameters
    model_name : str
        Name of the material (e.g. von_mises)
    benchmark_name : str
        Name of the benchmark (e.g. isotropic_compression)
    stress_list : list
        stress tensor data
    strain_list : list
        strain tensor / or deformation gradient (finite strain)
    velgrad_list : list
        strain increment / or velocity gradient (finite strain)
    mask_list : list
        mask to say if boundary is strain or stress control
    """
    # convert lists to numpy arrays
    # shape (N, 3, 3)
    stress_list = np.array(stress_list)
    strain_list = np.array(strain_list)
    velgrad_list = np.array(velgrad_list)
    mask_list = np.array(mask_list)

    # create output directory
    main_dir = cfg["global"]["output_dir"]
    sub_dir = model_name + "/"
    output_dir = main_dir + sub_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_pre = output_dir + benchmark_name + "_"

    # save data (e.g. output/von_mises/isotropic_compression_stress.npy)
    output_dir = output_dir + f"/{model_name}_{benchmark_name}"

    np.save(file_pre + "stress.npy", stress_list)
    np.save(file_pre + "strain.npy", strain_list)
    np.save(file_pre + "velgrad.npy", velgrad_list)
    np.save(file_pre + "mask.npy", mask_list)


def create_material(cfg, model_name):
    """

    Helper function to create a material model

    Parameters
    ----------
    cfg : dict
        Dictionary containing the configuration parameters
    model_name : str
        Name of the material (e.g. von_mises)

    Returns
    -------
    tuple
        Initialized particles and material
    """
    particles = pm.ParticlesContainer([[0.0, 0.0, 0.0]])
    material = None
    if model_name == "von_mises":
        material = pm.VonMises(
            cfg[model_name]["density"],
            cfg[model_name]["E"],
            cfg[model_name]["pois"],
            cfg[model_name]["yield_stress"],
            cfg[model_name]["H"],
        )

    if material is not None:
        particles, _ = material.initialize(particles, 0)
    return particles, material


def run_save_model(
    cfg,
    model_name,
    benchmark_name,
    target_strain=None,  # NOSONAR
    target_stress=None,
    mask=None,
    cycles=1,
    is_finite_strain=False,
):
    """Run a model against a servo mix boundary servo control and save the output data

    Can be either driven by infinite strain or finite strain

    If infinite strain, the deformation gradient array is used to store the
    total strain and the velocity gradient is used to store the strain increment.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the configuration parameters
    model_name : str
        Name of the material (e.g. von_mises)
    benchmark_name : str
        Name of the benchmark (e.g. isotropic_compression)
    target_strain : np.array, optional
        (3,3) np.array containing the target strain, by default None
    target_stress : np.array, optional
        (3,3) np.array containing the target stress, by default None
    mask : np.array, optional
        (3,3) np.array booleans if the target is strain or stress
        control, by default None
    cycles : int, optional
        Number of cycles (or targets) to run, by default 1
    is_finite_strain : bool, optional
        Flag if finite deformation driven, by default False
    """
    global stress_list, strain_list, F_list, velgrad_list, mask_list
    stress_list, strain_list, velgrad_list, mask_list = (
        [],
        [],
        [],
        [],
    )

    dt = cfg["global"]["timestep"]
    time = cfg["global"]["time"]

    pm.set_global_timestep(dt)

    particles, material = create_material(cfg, model_name)

    # loop through each set of targets
    for ci in range(cycles):
        print(f"cycle: {ci}")
        particles, material = mixed_control(
            particles,
            material,
            time,
            dt,
            mask[ci],
            target_strain[ci],
            target_stress=target_stress[ci],
            callback=callback,
            callback_step=cfg["global"]["output_steps"],
            is_finite_strain=is_finite_strain,
            cycle=ci,
            tolerance=cfg["mixed_control"]["tolerance"],
        )

    save_data(
        cfg,
        model_name,
        benchmark_name,
        stress_list,
        strain_list,
        velgrad_list,
        mask_list,
    )


# load config file
with open("./config.toml", "rb") as f:
    cfg = tomli.load(f)


##### ISOTROPIC COMPRESSION ######
target_strain = np.zeros((3, 3))
target_strain[0, 0] = -0.1
target_strain[1, 1] = -0.1
target_strain[2, 2] = -0.1

target_stress = np.zeros((3, 3))

mask = np.zeros((3, 3)).astype(bool)

run_save_model(
    cfg,
    "von_mises",
    "isotropic_compression",
    [target_strain],
    [target_stress],
    [mask],
)

###### UNIAXIAL COMPRESSION ######

target_strain = np.zeros((3, 3))
target_strain[0, 0] = -0.1
target_stress = np.zeros((3, 3))
mask = np.zeros((3, 3)).astype(bool)

run_save_model(
    cfg,
    "von_mises",
    "uniaxial_compression",
    [target_strain],
    [target_stress],
    [mask],
)

###### PURE SHEAR COMPRESSION ######

target_strain = np.zeros((3, 3))
target_strain[0, 1] = 0.1
target_strain[1, 0] = 0.1
target_stress = np.zeros((3, 3))
mask = np.zeros((3, 3)).astype(bool)

run_save_model(
    cfg,
    "von_mises",
    "pure_shear",
    [target_strain],
    [target_stress],
    [mask],
)


###### SIMPLE SHEAR COMPRESSION ######
target_strain = np.zeros((3, 3))
target_strain[0, 1] = 0.1
target_strain[1, 0] = 0.1
target_stress = np.zeros((3, 3))
mask = np.zeros((3, 3)).astype(bool)

run_save_model(
    cfg,
    "von_mises",
    "simple_shear",
    [target_strain],
    [target_stress],
    [mask],
)


###### TRIAXIAL COMPRESSION ######

target_strain = np.zeros((3, 3))
target_strain[0, 0] = -0.15

target_stress = np.zeros((3, 3))

mask = np.zeros((3, 3)).astype(bool)
mask[1, 1] = True
mask[2, 2] = True

run_save_model(
    cfg,
    "von_mises",
    "triaxial_compression",
    [target_strain],
    [target_stress],
    [mask],
)

###### CYCLIC LOADING ########


target_stress = np.zeros((3, 3))
target_strain_unload = np.zeros((3, 3))

mask = np.zeros((3, 3)).astype(bool)
mask[1, 1] = True
mask[2, 2] = True

mask_unload = np.zeros((3, 3)).astype(bool)
mask_unload[1, 1] = True
mask_unload[2, 2] = True

target_strain1_load = np.zeros((3, 3))
target_strain1_load[0, 0] = -0.1

target_strain2_load = np.zeros((3, 3))
target_strain2_load[0, 0] = -0.11


target_strain3_load = np.zeros((3, 3))
target_strain3_load[0, 0] = -0.12


target_strain4_load = np.zeros((3, 3))
target_strain4_load[0, 0] = -0.13

target_strain5_load = np.zeros((3, 3))
target_strain5_load[0, 0] = -0.14

target_strain6_load = np.zeros((3, 3))
target_strain6_load[0, 0] = -0.15

strain_control = [
    target_strain1_load,
    target_strain_unload,
    target_strain2_load,
    target_strain_unload,
    target_strain3_load,
    target_strain_unload,
    target_strain4_load,
    target_strain_unload,
    target_strain5_load,
    target_strain_unload,
    target_strain5_load,
]

run_save_model(
    cfg,
    "von_mises",
    "cyclic_loading",
    strain_control,
    [target_stress for ts in strain_control],
    [mask for ts in strain_control],
    cycles=len(strain_control),
)
