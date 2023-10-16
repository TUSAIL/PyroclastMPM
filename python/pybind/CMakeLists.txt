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
# ARISING IN ANY WAY OUT OF THE USE OF THIS SpOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


set(
  PyroclastPyBind

  "_pyroclastmpm.cpp"
  "particles/particles_bind.cpp"
  "nodes/nodes_bind.cpp"


  "common/global_settings_bind.cpp"
  "common/tools_bind.cpp"

  "solver/solver_bind.cpp"
  "solver/usl_bind.cpp"

  "boundaryconditions/boundaryconditions_bind.cpp"
  "boundaryconditions/bodyforce_bind.cpp"
  "boundaryconditions/gravity_bind.cpp"
  "boundaryconditions/rigidbodylevelset_bind.cpp"
  "boundaryconditions/planardomain_bind.cpp"
  "boundaryconditions/nodedomain_bind.cpp"

  "materials/materials_bind.cpp"
  "materials/vonmises_bind.cpp"
  "materials/linearelastic_bind.cpp"
  "materials/modified_camclay_bind.cpp"
  "materials/mohr_coulomb_bind.cpp"
  "materials/newtonfluid_bind.cpp"
  "materials/localrheology_bind.cpp"
  "materials/mu_i_bind.cpp"
  "materials/modified_camclay_mu_i_bind.cpp"

  # "spatialpartition_bind.cu" // currently no need to import this in python . . .
)

include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Python COMPONENTS Interpreter Development)

find_package(pybind11 CONFIG)

pybind11_add_module(MPM${DIM}D SHARED ${PyroclastPyBind})

set_target_properties(MPM${DIM}D
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../pyroclastmpm/"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../pyroclastmpm/"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../pyroclastmpm/"
)


target_link_libraries(MPM${DIM}D PRIVATE pyroclastmpm_${DIM}D)

target_link_libraries(MPM${DIM}D PRIVATE GTest::gtest_main Eigen3::Eigen ${VTK_LIBRARIES})

if(BUILD_CUDA)
  target_link_libraries(MPM${DIM}D PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(MPM${DIM}D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    --expt-relaxed-constexpr
    >
  )
  set_target_properties(MPM${DIM}D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_source_files_properties(${PyroclastPyBind} PROPERTIES LANGUAGE CUDA)
else()
  # let g++ compile cuda files
  target_link_libraries(MPM${DIM}D PRIVATE Thrust)
  set_source_files_properties(${PyroclastPyBind} PROPERTIES LANGUAGE CXX)
endif()

vtk_module_autoinit(
  FORCE_STATIC TARGETS MPM${DIM}D
  MODULES ${VTK_LIBRARIES}
)