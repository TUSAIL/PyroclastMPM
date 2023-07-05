# %%
import tomllib

import numpy as np
import plot


def get_data(cfg, model_name, benchmark_name):
    main_dir = cfg["global"]["output_dir"]
    sub_dir = model_name + "/"
    output_dir = main_dir + sub_dir
    file_pre = output_dir + benchmark_name + "_"

    stress = np.load(file_pre + "stress.npy")
    strain = np.load(file_pre + "strain.npy")
    velgrad = np.load(file_pre + "velgrad.npy")
    mask = np.load(file_pre + "mask.npy")
    return stress, strain, velgrad, mask


def plot_single_mp_data(
    stress, strain, velgrad, mask, model_code, benchmark_name
):
    plot_code = "components"

    plot.plot_component_subplot(
        stress,
        mask,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}_stress.png",
        f"{model} {benchmark_name} - stress vs loading steps",
        ptype="stress",
    )

    plot.plot_component_subplot(
        strain,
        mask,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}_strain.png",
        f"{model} {benchmark_name} - strain vs loading steps",
        ptype="strain",
    )

    plot.plot_component_subplot(
        velgrad,
        mask,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}_velgrad.png",
        f"{model} {benchmark_name} - stress vs loading steps",
        ptype="velgrad",
    )

    plot_code = "q_p"
    plot.q_p_plot(
        stress,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}.png",
        f"{model} {benchmark_name} - q vs p plot",
    )
    plot_code = "volume"
    plot.volume_plot(
        stress,
        strain,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}_over_p.png",
        f"{model} {benchmark_name} - det(F) vs p plot",
        over="p",
    )
    plot.volume_plot(
        stress,
        strain,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}_over_q.png",
        f"{model} {benchmark_name} - det(F) vs q plot",
        over="q",
    )

    plot_code = "stress_strain"
    plot.plot_component_vs(
        strain,
        stress,
        mask,
        f"./plots/{model_code}_{benchmark_name}_{plot_code}.png",
        f"{model} {benchmark_name} - stress vs strain",
        ptype1="strain",
        ptype2="stress",
    )


######### Von Mises #########

model = "Von Mises"
model_name = "von_mises"

benchmarks_names = [
    # "isotropic_compression",
    # "uniaxial_compression",
    # "simple_shear",
    # "pure_shear",
    "triaxial_compression",
    "cyclic_loading",
]

with open("./config.toml", "rb") as f:
    cfg = tomllib.load(f)

for i in range(len(benchmarks_names)):
    benchmark_name = benchmarks_names[i]
    stress, strain, velgrad, mask = get_data(cfg, model_name, benchmark_name)

    print(f"Plotting {model} {benchmark_name}")
    plot_single_mp_data(
        stress, strain, velgrad, mask, model_name, benchmark_name
    )
