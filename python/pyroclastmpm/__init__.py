import logging

failed_imports = []
errors = []
try:
    from . import MPM1D
except ImportError as e:
    failed_imports.append("MPM1D")
    errors.append(e)

try:
    from . import MPM2D
except ImportError as e:
    failed_imports.append("MPM2D")
    errors.append(e)

try:
    from . import MPM3D
except ImportError as e:
    failed_imports.append("MPM3D")
    errors.append(e)

if len(failed_imports) > 0:
    logging.warning(
        "failed to import the following modules: {}".format(failed_imports)
    )
