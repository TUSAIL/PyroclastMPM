import tomllib

import pyvista as pv
from pyvista import examples

cubemap = examples.download_sky_box_cube_map()

pv.set_plot_theme(pv.themes.DocumentTheme())

pv.global_theme.colorbar_horizontal.position_x = 0.2

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

particles = pv.read("./output/particles0.vtp")
particles["values"] = particles.point_data["Volume"]

particles["KE"] = (
    0.5
    * particles["Mass"]
    * (particles["Velocity"][:, 0] ** 2 + particles["Velocity"][:, 1] ** 2)
)

plane = pv.Plane(
    center=(
        (config["nodes"]["node_start"][0] + config["nodes"]["node_end"][0])
        / 2,
        (config["nodes"]["node_start"][1] + config["nodes"]["node_end"][1])
        / 2,
        0,
    ),
    direction=(0.0, 0.0, 1.0),
    i_size=config["nodes"]["node_end"][0],
    j_size=config["nodes"]["node_end"][1],
    i_resolution=int(
        config["nodes"]["node_end"][0] / config["nodes"]["node_spacing"]
    )
    + 1,
    j_resolution=int(
        config["nodes"]["node_end"][1] / config["nodes"]["node_spacing"]
    )
    + 1,
)

# Open a movie file
pl = pv.Plotter(notebook=False, off_screen=True)

pl.open_gif(config["postprocess"]["movie_file"])

pl.add_mesh(
    particles,
    render_points_as_spheres=True,
    point_size=15,
    scalars="KE",
    cmap="hot",
    pbr=False,
    lighting=True,
    interpolate_before_map=True,
    clim=[0.005, 0.008],
)


pl.camera_position = "xy"
pl.camera.zoom(0.9)

pl.show(auto_close=False)  # only necessary for an off-screen movie
pl.set_background("aliceblue", top="white")
pl.set_environment_texture(cubemap)

# Run through each frame
pl.write_frame()  # write initial data

# Update scalars on each frame
total_steps = config["global"]["total_steps"]
output_steps = config["global"]["output_steps"]
for i in range(0, total_steps, output_steps):
    particles_updated = pv.read(f"./output/particles{i}.vtp")
    particles_updated["KE"] = (
        0.5
        * particles_updated["Mass"]
        * (
            particles_updated["Velocity"][:, 0] ** 2
            + particles_updated["Velocity"][:, 1] ** 2
        )
    )
    particles.copy_from(particles_updated)

    # pl.add_text(f"Iteration: {i}", name='time-label')
    pl.render()
    pl.write_frame()


# Be sure to close the plotter when finished
pl.close()
