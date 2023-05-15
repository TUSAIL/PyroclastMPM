# %%
import numpy as np
import pyvista as pv
import tomllib

from pyroclastmpm import (
    get_bounds,
)

# load config file
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

particles = pv.read('./output/particles500.vtp')

domain = pv.read(config['project']['domain_file'])
scoop = pv.read(config['project']['scoop_file'])

# load stls
domain_start, domain_end = get_bounds(config['project']['domain_file'])

# calculate cell size
cell_size = abs(np.min(domain_end - domain_start) / config['global']['cell_size_ratio'])

# cell_size /= 2

grid = pv.UniformGrid(
    origin= domain_start,
    spacing=(cell_size,cell_size,cell_size),
    dimensions=(
        int((domain_end[0]-domain_start[0])/cell_size),
        int((domain_end[1]-domain_start[1])/cell_size),
        int((domain_end[2]-domain_start[2])/cell_size))
)

grid.clear_data()

particles = pv.read('./output/particles0.vtp')

# # # Open a movie file
# pl = pv.Plotter()

pl = pv.Plotter(notebook=False, off_screen=True)
pl.open_movie(config['postprocess']['movie_file'])
particles['values'] = particles.point_data['Mass']

interp = grid.interpolate(particles, radius=cell_size, sharpness=2, strategy='mask_points')
vol_opac = [0, 0, 0.2, 0.2, 0.5, 0.5]

# Add initial mesh
pl.add_mesh(grid.outline(),line_width=5)
pl.add_mesh(scoop,
            line_width=5,
            specular=1,
            lighting=True, color='white',
            opacity=0.5)

pl.add_volume(interp,cmap='coolwarm',opacity=vol_opac)

pl.add_mesh(particles,
            render_points_as_spheres=True, 
            specular=1,
            point_size=7.5,
            scalars='Velocity',
            cmap='coolwarm' )

# pl.enable_eye_dome_lighting()
pl.camera_position = 'xz'
pl.camera.zoom(1.2)
pl.set_background('blue', top='white')
pl.show(auto_close=False)  # only necessary for an off-screen movie
# Run through each frame
pl.write_frame()  # write initial data

#%%

# Update scalars on each frame
for i in range(0, config['global']['total_steps'],config['global']['output_steps']):
    print(i)
    upd = pv.read(f'./output/particles{i}.vtp')
    particles.copy_from(upd)
    particles['values'] = upd.point_data['Mass']

    grid.clear_data()
    upd_interp = grid.interpolate(particles, radius=cell_size, sharpness=2, strategy='mask_points')

    
    interp.copy_from(upd_interp)

    pl.add_text(f"Iteration: {i}", name='time-label')
    pl.render()
    pl.write_frame()


# Be sure to close the plotter when finished
pl.close()


# %%
