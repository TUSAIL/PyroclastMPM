![header](https://capsule-render.vercel.app/api?type=waving&color=0:12c2e9,50:c471ed,100:f64f59&height=280&section=header&text=PyroclastMPM&fontSize=90&fontColor=FFFFFF)

<p align="center" style="margin-bottom: 0px !important;  padding-left: 100px">
  <img width="400" src="./doc/scoop_no_boundary.gif" alt="PyroclastMPM logo" align="center">
</p>

<!--
<h1 align="center" style="margin-top: 0px;">PyroclastMPM</h1> -->

<p align="center"  > A modular GPU based Material Point Method (MPM) solver. </p>

# Welcome

PyroclastMPM is an explicit GPU-based Material Point Method (MPM) solver developed by Retief Lubbe. The solver is written in CUDA and C++ with Python bindings and is designed to be fast, modular, and highly extensible.

# Motivation

The main goal of PyroclastMPM is to accurately model granular impact problems involving the state transition from solid to fluid and back ⏳️. The solver is specifically designed to be applicable in industrial settings, striking a balance between computational performance and code readability.

# Overview of Features

Coming soon!

# Showcase examples

Coming soon!

# Requirements

Checking requirements 📋

- GCC version `gcc --version` and `g++ --version` should be less than `>8 or <12`
- CMAKE version `cmake --version` should be `>=3.11`
- CUDA `nvcc --version`
- Poetry `poetry --version`
- vtk9 and libvtk9-dev

# Installation

## Method A: Building source 🔧

1. Check if all the [requirements](#requirements)
2. Clone repository and submodules (with ssh)

```
git clone --recurse-submodules -j8 git@github.com:TUSAIL/PyroclastMPM.git
cd PyroclastMPM
```

3. Install Python depedencies `poetry install`
4. Build project

```
mkdir build
cmake -B ./build/ -S .
make -j4 -C ./build
```

5. Optional - run tests `./build/tests/tests` (compiled executable), `pytest ./tests/` (Python)
6. Run example ` make -C examples/Scoop`

## Method B: Python wheel

(Coming soon 👀
Will look something like this) `pip install <wheel>`

## Method C: Docker-compose

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/TUSAIL/PyroclastMPM/tree/cpu)

# Documentation

## Usage

## Detailed list of features

## Programming frameworks used

<a >
  <img height="50" src="./doc/docker.png" />
</a>
<a >
  <img height="50" src="./doc/python.png" />
</a>
<a >
  <img height="50" src="./doc/cpp.png" />
</a>
<a >
  <img height="50" src="./doc/nvidia.png" />
</a>

## Known issues

<!-- # Recommended citation -->

# Contribute

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/TUSAIL/PyroclastMPM/issues)

## Contributing

Coming soon 👀

## Contributor list

## Want to contribute?

# See also

Coming soon 👀

## Credits

## Inspiration

## Alternatives

<!--
## Implementation
* 1D, 2D, 3D
* C++ / CUDA with
* Python bindings
* Particle volume and surface body fill

## Shape functions
* Linear
* Quadratic
* cubic splines

## Solvers
* Update stress last
* Modified Update Stress Last (in progress)
* Total Lagrangian Material Point Method (MPM) (in progress)
* APIC/MLS (in progress)

## Boundary conditions
* Rigid body level-set (UL)
* DEM-based contacts (TODO)

# Constitutive models
* Linear Elasticity
* Newtonian Fluid
* Local granular rheology  (in progress)
* Non Local granular rheology  (in progress)
* Druger prager  (in progress)

# Prerequisites

Building PyroclastMPM requires the following software installed:

* A C++11-compliant compiler (required)
* CMake `>= 3.22` (required)
* VTK 9 (required)
* CUDA 10> (required)
* Doxygen (optional, documentation building is skipped if missing)* Python `>= 3.8` for building Python bindings

# Installation

The following sequence of commands install PyroclastMPM.
## Python package


# Source file


It assumes that your current working directory is the top-level directory
of the freshly cloned repository:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>={ON, OFF}` to the `cmake` call:

* `BUILD_TESTING`: Enable building of the test suite (default: `ON`)
* `BUILD_DOCS`: Enable building the documentation (default: `ON`)
* `BUILD_PYTHON`: Enable building the Python bindings (default: `ON`)


If you wish to build and install the project as a Python project without
having access to C++ build artifacts like libraries and executables, you
can do so using `pip` from the root directory:

```
python -m pip install .
```

# Testing PyroclastMPM

When built according to the above explanation (with `-DBUILD_TESTING=ON`),
the C++ test suite of `PyroclastMPM` can be run using
`ctest` from the build directory:

```
cd build
ctest
```

The Python test suite can be run by first `pip`-installing the Python package
and then running `pytest` from the top-level directory:

```
python -m pip install .
pytest
```

# Documentation

PyroclastMPM provides a Sphinx-based documentation, that can
be browsed [online at readthedocs.org](https://pyroclastmpm.readthedocs.io).
To build it locally, first ensure the requirements are installed by running this command from the top-level source directory:

```
pip install -r doc/requirements.txt
```

Then build the sphinx documentation from the top-level build directory:

```
cmake --build . --target sphinx-doc
```

The web documentation can then be browsed by opening `doc/sphinx/index.html` in your browser. -->
