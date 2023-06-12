# BSD 3-Clause License
# Copyright (c) 2023, Retief Lubbe
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this
#  list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(pyroclastmpm VERSION 0.1.0 LANGUAGES C CXX)

# Compiler options
set(CMAKE_POLICY_DEFAULT_CMP0025 NEW)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(DIM 3 CACHE INTERNAL "")
# SET(DIM 3 CACHE INTERNAL "Dimension of the simulation (plane strain)")
# set(DIMENSION 3 CACHE INTERNAL "")

message(DIM='${DIM}')

# option(BUILD_OPENMP "Enable CUDA Acceleration" OFF) # later
option(BUILD_CUDA "Enable CUDA Acceleration" ON)
option(BUILD_PYTHON "Enable building of Python bindings" ON)
option(BUILD_DOCS "Enable building of documentation" OFF)
option(BUILD_TESTING "Enable testing" ON)
option (DOUBLE_PRECISION "Use double precision" OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DDIM=${DIM})



if (DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLES) 
endif()

if(BUILD_CUDA)
    add_definitions(-DCUDA_ENABLED) 
    project(pyroclastmpm VERSION 0.0.1 LANGUAGES C CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    project(pyroclastmpm VERSION 0.0.1 LANGUAGES C CXX)
    # Thrust
    add_subdirectory(ext/thrust thrust)
    #solves https://github.com/NVIDIA/thrust/issues/386
    add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

# EIGEN3
add_subdirectory(ext/eigen/)

# VTK
find_package(VTK REQUIRED
    COMPONENTS
    CommonCore
    CommonDataModel
    CommonTransforms
    FiltersSources
    FiltersCore
    FiltersGeneral
    FiltersPoints
    InfovisCore # DataObject
    IOCore
    IOExport
    IOGeometry
    IOXML
    IOHDF
)

include_directories(
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
PUBLIC
)

# main ompile the library
add_subdirectory(src)

# compile the tests
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()

if(BUILD_DOCS)
    add_subdirectory(doc)
endif()

if(BUILD_PYTHON)
    add_subdirectory(ext/pybind11)
    add_subdirectory(python/pyroclastmpm/pybind)
endif()

# Add an alias target for use if this project is included as a subproject in another project
add_library(pyroclastmpm::pyroclastmpm ALIAS pyroclastmpm)

# Install targets and configuration
install(
    TARGETS pyroclastmpm
    EXPORT pyroclastmpm-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT pyroclastmpm-config
    NAMESPACE pyroclastmpm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyroclastmpm
)

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# This prints a summary of found dependencies
include(FeatureSummary)

feature_summary(WHAT ALL)