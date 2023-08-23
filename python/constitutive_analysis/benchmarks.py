import numpy as np
import pyroclastmpm.MPM3D as pm

from ._servocontroller import mixed_control


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


def run_benchmark(model_cfg, benchmark_cfg, global_cfg, callback):
    # initialize global variables
    dt = global_cfg["timestep"]
    time = global_cfg["time"]
    pm.set_global_timestep(dt)

    # initialize particles and material
    particles, material = create_material(model_cfg)

    for ci, cycle in enumerate(benchmark_cfg["run"]):
        particles, material = mixed_control(
            particles,
            material,
            time,
            dt,
            np.array(cycle["is_stress_control"]),
            np.array(cycle["target_strain"]),
            target_stress=np.array(cycle["target_stress"]),
            callback=callback,
            callback_step=global_cfg["output_steps"],
            is_finite_strain=global_cfg["is_finite_strain"],
            cycle=ci,
            tolerance=global_cfg["tolerance"],
        )

    return particles, material


def run(global_cfg, callback):
    # 2. Run models against benchmarks
    for model in global_cfg["models"]:
        model_name = model["name"]
        # skip models not in white list
        if model_name not in global_cfg["white_lists"]["model_names"]:
            continue
        print(f"Running model: {model_name}")

        # output_data[model_name] = {}
        for benchmark in global_cfg["benchmarks"]:
            benchmark_name = benchmark["name"]
            # skip benchmarks not in white list
            if (
                benchmark_name
                not in global_cfg["white_lists"]["benchmark_names"]
            ):
                continue
            print(f"Running benchmark: {benchmark_name}")

            # stress_list, strain_list, velgrad_list, mask_list = (
            #     [],
            #     [],
            #     [],
            #     [],
            # )

            dt = global_cfg["global"]["timestep"]
            time = global_cfg["global"]["time"]

            pm.set_global_timestep(dt)

            particles, material = create_material(model)

            for ci, cycle in enumerate(benchmark["run"]):
                particles, material = mixed_control(
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
            # output_data[model_name][benchmark_name] = {}
            # output_data[model_name][benchmark_name]["stress"] = np.array(
            #     stress_list
            # )
            # output_data[model_name][benchmark_name]["strain"] = np.array(
            #     strain_list
            # )
            # output_data[model_name][benchmark_name]["velgrad"] = np.array(
            #     velgrad_list
            # )
            # output_data[model_name][benchmark_name]["mask"] = np.array(
            #     mask_list
            # )

    # return output_data
