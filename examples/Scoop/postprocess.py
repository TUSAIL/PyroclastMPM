# %%
import numpy as np
import pyvista as pv
import tomllib

from pyroclastmpm import (
    get_bounds,
)
pv.set_plot_theme(pv.themes.DocumentTheme())

pv.global_theme.colorbar_horizontal.position_x = 0.2

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

particles = pv.read('./output/particles0.vtp')
# particles
#%%
domain = pv.read(config['project']['domain_file'])
scoop = pv.read(config['project']['scoop_file'])

particles['values'] = particles.point_data['Volume']
# create 2d grid


particles = pv.read('./output/particles0.vtp')

# # Open a movie file
pl = pv.Plotter(notebook=False, off_screen=True)
pl.open_movie(config['postprocess']['movie_file'])

# # Add initial mesh
pl.add_mesh(domain.outline(),line_width=5)
pl.add_mesh(scoop,
            line_width=5,
            specular=1,
            lighting=True, color='deeppink',
            opacity=0.5)
pl.add_mesh(particles,
            render_points_as_spheres=True, 
            specular=1,
            point_size=5,
            scalars='Velocity',
            cmap='coolwarm' )
# pl.enable_eye_dome_lighting()
pl.camera_position = 'xz'
pl.camera.zoom(1.)
# pl.set_background('aliceblue')
pl.show(auto_close=False)  # only necessary for an off-screen movie

# Run through each frame
pl.write_frame()  # write initial data



# Update scalars on each frame
for i in range(0, config['global']['total_steps'],config['global']['output_steps']):
    
    upd = pv.read(f'./output/particles{i}.vtp')
    upd['values'] = upd.point_data['Volume']
    particles.copy_from(upd)

    pl.add_text(f"Iteration: {i}", name='time-label')
    pl.render()
    pl.write_frame()


# Be sure to close the plotter when finished
pl.close()


# %%
