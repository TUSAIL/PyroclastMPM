import numpy as np
import pyroclastmpm.MPM3D as pm

from .servo_control import mixed_control


def create_material(model):
    """
    Helper function to create a material model

    Parameters
    ----------
    model : dict
        a dictionary containing the model name and configuration parameters

    Returns
    -------
    tuple
        Initialized particles and material
    """
    particles = pm.ParticlesContainer([[0.0, 0.0, 0.0]])
    material = None

    particles.volumes = [model["volume"]]
    if "volume_original" not in model:
        particles.volumes_original = [model["volume"]]
        particles.masses = [model["density"] * model["volume"]]
    else:
        particles.volumes_original = [model["volume_original"]]
        particles.masses = [model["density"] * model["volume_original"]]
    if "prestress" in model:
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
            model["R"],
            model["Pt"],
            model["beta"],
        )
    elif model["type"] == "mu_i":  # TODO fix
        material = pm.MuIJop(
            model["density"],
            0,  # replace dummy
            model["bulk_modulus"],
            0,  # replace dummy
        )

    try:  # TODO fix workaround
        if material is not None:
            particles, _ = material.initialize(particles, 0)
    except AttributeError:
        pass
    return particles, material


def run_benchmark(model_cfg, benchmark_cfg, global_cfg, callback):
    """Run a single benchmark against a single model.

    Parameters
    ----------
    model_cfg : dict
        a dictionary containing the model configuration parameters
    benchmark_cfg : dict
        a dictionary containing the benchmark configuration parameters
    global_cfg : _type_
        a dictionary containing the global configuration parameters
    callback : function
        a callback function to be called at each step of the simulation

    Returns
    -------
    particles : ParticlesContainer
        a container of particles (material points) with updated stress and strain state
    material : Material
        a material model with updated internal state

    """
    # initialize global variables
    time = global_cfg["time"]

    if type(global_cfg["timestep"]) == float:
        dt = global_cfg["timestep"]
        pm.set_global_timestep(dt)
        particles, material = create_material(model_cfg)
        for ci, cycle in enumerate(benchmark_cfg["run"]):
            print(f"cycle {ci} timestep {dt} cycle index {ci}")
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
    elif type(global_cfg["timestep"]) == list:
        for ti, dt in enumerate(global_cfg["timestep"]):
            pm.set_global_timestep(dt)
            particles, material = create_material(model_cfg)
            num_cycles = len(benchmark_cfg["run"])
            for ci, cycle in enumerate(benchmark_cfg["run"]):
                cycle_index = ci + num_cycles * ti
                print(f"cycle {ci} timestep {dt} cycle index {cycle_index}")
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
                    cycle=cycle_index,  # linearize index
                    tolerance=global_cfg["tolerance"],
                )

    return particles, material


def run(global_cfg, callback):
    """Run a list of models against a list of benchmarks.

    Parameters
    ----------
    global_cfg : dict
        a dictionary containing the global configuration parameters
    callback : function
        a callback function to be called at each step of the simulation
    """

    # 2. Run models against benchmarks

    for model in global_cfg["models"]:
        model_name = model["name"]

        # skip models not in white list
        if model_name not in global_cfg["white_lists"]["model_names"]:
            continue

        print(f"Running model: {model_name}")

        for benchmark in global_cfg["benchmarks"]:
            benchmark_name = benchmark["name"]

            # skip benchmarks not in white list
            if (
                benchmark_name
                not in global_cfg["white_lists"]["benchmark_names"]
            ):
                continue

            print(f"Running benchmark: {benchmark_name}")

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
