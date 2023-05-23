

#%%
from pyroclastmpm import (
    LinearElastic,
    ParticlesContainer,
    VTK, CSV
)

particles = ParticlesContainer(
    positions=np.array([[0.1,0.2,0.3]]),

    output_formats=[VTK, CSV]
)

# particles = ParticlesContainer(
#     positions=positions,
#     # velocities=velocities,
#     # colors=colors,
#     output_formats=[VTK, CSV]
#     )

print("success")
# %%
