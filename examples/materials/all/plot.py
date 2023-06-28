import matplotlib.pyplot as plt
import numpy as np


def plot_stress_subplot(list_dataset, file, title):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=False)
    fig.suptitle(title)
    colors = ["red", "blue", "green"]
    for di, dataset in enumerate(list_dataset):
        stresses = dataset["stress"]
        print(stresses.shape)
        steps = np.arange(len(stresses))

        for i in range(3):
            for j in range(3):
                axs[i, j].scatter(steps, stresses[:, i, j], color=colors[di])
                nx = i + 1  # x component of stress/strain component
                ny = j + 1  # y component of stress/strain component
                axs[i, j].set_ylabel(f"$\sigma_{{{nx}{ny}}}$ ")
                # axs[i,j].set_ylabel(f'$\varepsilon_{{{nx}{ny}}}$ ')
                # axs[i, j].grid()

    plt.tight_layout()
    plt.savefig(
        file,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
    plt.clf()
