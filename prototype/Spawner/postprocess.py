# %%
import tomllib

import numpy as np
import pyvista as pv
from pyroclastmpm import get_bounds
from pyvista import examples

# Download skybox
cubemap = examples.download_sky_box_cube_map()

pv.set_plot_theme(pv.themes.DocumentTheme())

pv.global_theme.colorbar_horizontal.position_x = 0.2

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

particles = pv.read("./output/particles0.vtp")
# particles
# %%
domain = pv.read(config["project"]["domain_file"])

# create 2d grid


particles = pv.read("./output/particles0.vtp")

# # Open a movie file
pl = pv.Plotter(notebook=False, off_screen=True)
# pl.open_movie(config['postprocess']['movie_file'])
pl.open_gif(config["postprocess"]["movie_file"])

# # Add initial mesh
pl.add_mesh(
    domain.outline(),
    color="lightgray",
    pbr=True,
    metallic=0.8,
    roughness=0.3,
    diffuse=1,
    line_width=4,
    render_lines_as_tubes=True,
)

pl.add_mesh(
    particles,
    render_points_as_spheres=True,
    point_size=5,
    scalars="Volume",
    cmap="winter",
    pbr=False,
    ambient=0.5,  # amount of light that reaches the object
    specular=0.2,
    specular_power=0.1,
    lighting=True,
    interpolate_before_map=True,
    # clim=[0, 0.25],
)


# mesh.plot(scalars=mesh['values'], )
# pl.enable_eye_dome_lighting()
pl.camera_position = "xz"
# pl.camera.roll(90)
pl.camera.elevation += 10
pl.camera.zoom(1.0)
pl.set_background("aliceblue", top="white")
pl.set_environment_texture(cubemap)

pl.show(auto_close=False)  # only necessary for an off-screen movie

# Run through each frame
pl.write_frame()  # write initial data


filenotfound = False
# Update scalars on each frame
for i in range(0, config["global"]["total_steps"], config["global"]["output_steps"]):
    try:
        upd = pv.read(f"./output/particles{i}.vtp")
    except FileNotFoundError:
        filenotfound = True
        continue
    upd["values"] = upd.point_data["Volume"]
    particles.copy_from(upd)

    # pl.add_text(f"Iteration: {i}", name='time-label')
    pl.render()
    pl.write_frame()

if filenotfound:
    print("File not found, could be incomplete run.")
# Be sure to close the plotter when finished
pl.export_gltf("plots/output.gltf")
pl.close()


# %%
