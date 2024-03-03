# trunk-ignore-all(ruff/F401)

import logging

failed_imports = []
errors = []

# TODO: how to avoid nested try/excepts?
# problem loading both MPM3D and MPM2D
# solutuon 1: make it possible to only load one of them at a time
# solution 2: make it possible to load both of them at the same time, under different submodules

from . import MPM2D
# try:
#     from . import MPM3D
# except ImportError as e:
#     failed_imports.append("MPM3D")
#     errors.append(e)
#     try:
#         from . import MPM2D
#     except ImportError as e:
#         failed_imports.append("MPM2D")
#         errors.append(e)
#         try:
#             from . import MPM1D
#         except ImportError as e:
#             failed_imports.append("MPM1D")
#             errors.append(e)
#             raise

if len(failed_imports) > 0:
    logging.warning(
        "failed to import the following modules: {}".format(failed_imports)
    )
