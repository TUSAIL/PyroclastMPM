import os

import module_plot
import module_servo
import numpy as np
import pyroclastmpm.MPM3D as pm
import tomli

global stress_list, strain_list, F_list, velgrad_list, mask_list
stress_list, strain_list, velgrad_list, mask_list = [], [], [], []

work_dir = os.getcwd()
print(f"Current working directory: {work_dir}")


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
    print(f"step: {step}", end="\r")


def create_material(model):
    """
    Helper function to create a material model

    Parameters
    ----------
    model : dict
        Dictionary containing the model name and
        configuration parameters

    Returns
    -------
    tuple
        Initialized particles and material
    """
    particles = pm.ParticlesContainer([[0.0, 0.0, 0.0]])
    material = None

    particles.volumes = [model["volume"]]
    particles.volumes_original = [model["volume"]]
    particles.stresses = [np.array(model["prestress"])]
    if model["type"] == "von_mises":
        material = pm.VonMises(
            model["density"],
            model["E"],
            model["pois"],
            model["yield_stress"],
            model["H"],
        )
    elif model["type"] == "mohr_coulomb":
        material = pm.MohrCoulomb(
            model["density"],
            model["E"],
            model["pois"],
            model["cohesion"],
            model["friction_angle"],
            model["dilation_angle"],
            model["H"],
        )
    elif model["type"] == "modified_cam_clay":
        material = pm.ModifiedCamClay(
            model["density"],
            model["E"],
            model["pois"],
            model["M"],
            model["lam"],
            model["kap"],
            model["Vs"],
            model["Pc0"],
            model["Pt"],
            model["beta"],
        )

    if material is not None:
        particles, _ = material.initialize(particles, 0)
    return particles, material


# 1. Load config file

CONFIG = "./modifiedcamclay_config.toml"
print(f"Loading config file {CONFIG}")

with open(CONFIG, "rb") as f:
    global_cfg = tomli.load(f)

output_data = {}

# 2. Run models against benchmarks
for model in global_cfg["models"]:
    model_name = model["name"]
    # skip models not in white list
    if model_name not in global_cfg["white_lists"]["model_names"]:
        continue
    print(f"Running model: {model_name}")

    output_data[model_name] = {}
    for benchmark in global_cfg["benchmarks"]:
        benchmark_name = benchmark["name"]
        # skip benchmarks not in white list
        if benchmark_name not in global_cfg["white_lists"]["benchmark_names"]:
            continue
        print(f"Running benchmark: {benchmark_name}")

        stress_list, strain_list, velgrad_list, mask_list = (
            [],
            [],
            [],
            [],
        )

        dt = global_cfg["global"]["timestep"]
        time = global_cfg["global"]["time"]

        pm.set_global_timestep(dt)

        particles, material = create_material(model)

        for ci, cycle in enumerate(benchmark["run"]):
            particles, material = module_servo.mixed_control(
                particles,
                material,
                time,
                dt,
                np.array(cycle["is_stress_control"]),
                np.array(cycle["target_strain"]),
                target_stress=np.array(cycle["target_stress"]),
                callback=callback,
                callback_step=global_cfg["global"]["output_steps"],
                is_finite_strain=global_cfg["mixed_control"][
                    "is_finite_strain"
                ],
                cycle=ci,
                tolerance=global_cfg["mixed_control"]["tolerance"],
            )
        output_data[model_name][benchmark_name] = {}
        output_data[model_name][benchmark_name]["stress"] = np.array(
            stress_list
        )
        output_data[model_name][benchmark_name]["strain"] = np.array(
            strain_list
        )
        output_data[model_name][benchmark_name]["velgrad"] = np.array(
            velgrad_list
        )
        output_data[model_name][benchmark_name]["mask"] = np.array(mask_list)


# 3. Plot results
output_dir = global_cfg["global"]["output_dir"]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Plotting in {output_dir}")

for plot in global_cfg["plot"]:
    plot_name = plot["name"]

    # skip plots not in white list
    if plot_name not in global_cfg["white_lists"]["plot_names"]:
        continue

    print(f"Plotting: {plot_name}")

    stress_list, strain_list, names_list = ([], [], [])

    color_id_list, marker_id_list = [], []
    for mi, model_name in enumerate(global_cfg["white_lists"]["model_names"]):
        for bi, benchmark_name in enumerate(
            global_cfg["white_lists"]["benchmark_names"]
        ):
            color_id_list.append(bi)
            marker_id_list.append(mi)
            names_list.append(f"{model_name}_{benchmark_name}")
            stress_list.append(
                output_data[model_name][benchmark_name]["stress"]
            )
            strain_list.append(
                output_data[model_name][benchmark_name]["strain"]
            )
    style_id_tuples = list(zip(color_id_list, marker_id_list))

    plot_type = plot["type"]
    plot_file = output_dir + plot_type + ".png"
    if plot_type == "q_p":
        module_plot.q_p_plot(
            stress_list,
            names_list,
            plot_file,
            f"{plot_name} - q vs p plot",
        )
    if plot_type == "stress_vs_steps":
        module_plot.plot_component_step_subplot(
            stress_list,
            names_list,
            style_id_tuples,
            plot_file,
            f"{plot_name} - stress vs steps",
            ptype="stress",
        )

    if plot_type == "strain_vs_steps":
        module_plot.plot_component_step_subplot(
            strain_list,
            names_list,
            style_id_tuples,
            plot_file,
            f"{plot_name} - strain vs steps",
            ptype="strain",
        )
    if plot_type == "stress_vs_strain":
        module_plot.plot_component_vs(
            strain_list,
            stress_list,
            names_list,
            style_id_tuples,
            file=plot_file,
            title=f"{plot_name} - stress vs strain",
        )
    if plot_type == "volume_vs_lnp":
        module_plot.volume_plot(
            strain_list,
            stress_list,
            names_list,
            style_id_tuples,
            file=plot_file,
            over="p",
        )
    if plot_type == "volume_vs_q":
        module_plot.volume_plot(
            strain_list,
            stress_list,
            names_list,
            style_id_tuples,
            file=plot_file,
            over="q",
        )
    if plot_type == "shear_volume_strain":
        module_plot.shear_volume_strain_plot(
            strain_list,
            names_list,
            plot_file,
            f"{plot_name} - q vs p plot",
        )
