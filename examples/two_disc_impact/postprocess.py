"""_summary_

This is a postprocessing script for the 2D body impact example.

"""


import tomli

import matplotlib.pyplot as plt
import pyvista as pv

# 1. Define global plot settings
pv.set_plot_theme(pv.themes.DocumentTheme())
pv.global_theme.colorbar_horizontal.position_x = 0.2


# 2. Define helper function to compute kinetic energy
def calc_ke(particles):
    """Calculate kinetic energy of particles."""
    return (
        0.5
        * particles["Mass"]
        * (particles["Velocity"][:, 0] ** 2 + particles["Velocity"][:, 1] ** 2)
    )


# 3. Load config file
with open("./config.toml", "rb") as f:
    config = tomli.load(f)

# 4. Load particles and nodes


particles = pv.read("./output/particles0.vtp")

scalars = calc_ke(particles)

scalar_name = "Kinetic Energy"

particles.point_data.set_scalars(scalars, name=scalar_name)

nodes = pv.read("./output/nodes0.vtp")

# 5. Open a movie file and add initial frame
pl = pv.Plotter(notebook=False, off_screen=True)

pl.open_gif(config["postprocess"]["movie_file"])

pl.add_mesh(
    particles,
    render_points_as_spheres=True,
    point_size=15,
    scalars=scalar_name,
    cmap="hot",
    pbr=False,
    lighting=True,
    interpolate_before_map=True,
    clim=[0.005, 0.010],
)

pl.add_mesh(nodes, render_points_as_spheres=True)

pl.camera_position = "xy"

pl.camera.zoom(0.9)

pl.show(auto_close=False)

pl.write_frame()

# 6. Add remaining frames
KE = [scalars.sum()]
time = [0]
total_steps = config["global"]["total_steps"]
output_steps = config["global"]["output_steps"]
for i in range(output_steps, total_steps, output_steps):
    particles_updated = pv.read(f"./output/particles{i}.vtp")
    scalars = calc_ke(particles)
    particles.copy_from(particles_updated)
    particles.point_data.set_scalars(scalars, name=scalar_name)
    time.append(i * config["global"]["dt"])
    KE.append(scalars.sum())
    pl.render()
    pl.write_frame()

# # Be sure to close the plotter when finished
pl.close()

error = abs((KE[-1] - KE[0]) / KE[0])

plt.plot(time, KE)
plt.xlabel("Time (s)")
plt.ylabel("Kinetic Energy (J)")

plt.savefig(config["postprocess"]["ke_time_plot"])
assert error < 0.1, "KE should be conserved"
