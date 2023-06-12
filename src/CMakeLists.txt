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

set(
  PyroclastSrc

  "boundaryconditions/bodyforce.cpp"
  "boundaryconditions/gravity.cpp"
  "boundaryconditions/rigidbodylevelset.cpp"
  "boundaryconditions/planardomain.cpp"
  "boundaryconditions/nodedomain.cpp"

  "materials/linearelastic.cpp"
  "materials/newtonfluid.cpp"
  "materials/localrheo.cpp"
  "materials/vonmises.cpp"
  "materials/mohrcoulomb.cpp"

  "solver/solver.cpp"
  "shapefunction/shapefunction.cpp"
  "solver/usl/usl.cpp"
  "particles/particles.cpp"
  "nodes/nodes.cpp"

  "common/helper.cpp"
  "common/global_settings.cpp"
  "common/output.cpp"
  "common/tools.cpp"

  "spatialpartition/spatialpartition.cpp"



)


  # # future
  # # "solver/apic/apic.cu"
  # "solver/musl/musl.cu"
  # "solver/tlmpm/tlmpm.cu"

add_library(pyroclastmpm ${PyroclastSrc})

target_link_libraries(pyroclastmpm PRIVATE Eigen3::Eigen ${VTK_LIBRARIES} )


if(BUILD_CUDA)
  target_link_libraries(pyroclastmpm PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(pyroclastmpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    --expt-relaxed-constexpr
    >
  )

  set_source_files_properties(${PyroclastSrc} PROPERTIES LANGUAGE CUDA)
  set_target_properties(pyroclastmpm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

else()
#let g++ compile cuda files
set_source_files_properties(${PyroclastSrc} PROPERTIES LANGUAGE CXX)
endif()

vtk_module_autoinit(
TARGETS pyroclastmpm
MODULES ${VTK_LIBRARIES}
)