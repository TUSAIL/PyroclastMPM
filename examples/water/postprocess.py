import numpy as np
import pyvista as pv

total_steps = 80000
output_steps = 50

# 1. Define global plot settings
pv.set_plot_theme(pv.themes.DocumentTheme())
pv.global_theme.colorbar_horizontal.position_x = 0.2


# 2. Define helper function to compute kinetic energy
def calc_velmag(particles):
    """Calculate kinetic energy of particles."""
    return np.sqrt(
        particles["Velocity"][:, 0] ** 2 + particles["Velocity"][:, 1] ** 2
    )


particles = pv.read("./output/particles0.vtp")

scalars = calc_velmag(particles)

scalar_name = "Velocity Magnitude"

particles.point_data.set_scalars(scalars, name=scalar_name)


nodes = pv.read("./output/nodes0.vtp")

# 5. Open a movie file and add initial frame
pl = pv.Plotter(notebook=False, off_screen=True)

pl.open_gif("./plots/flow.gif")

pl.add_mesh(
    particles,
    style="points",
    point_size=10,
    scalars=scalar_name,
    cmap="kbc",
    clim=[-2.5, 5],
)

pl.add_mesh(nodes.outline(), line_width=10, color="k")

pl.camera_position = "xy"

pl.camera.zoom(0.9)

pl.show(auto_close=False)

pl.write_frame()


# 6. Add remaining frames

for i in range(output_steps, total_steps, output_steps):
    try:
        particles_updated = pv.read(f"./output/particles{i}.vtp")
        scalars = calc_velmag(particles)
        particles.copy_from(particles_updated)
        particles.point_data.set_scalars(scalars, name=scalar_name)

        pl.write_frame()
    except FileNotFoundError as e:
        print(e)
        break

# # Be sure to close the plotter when finished
pl.close()
