
set(
  PyroclastSrc

  "boundaryconditions/bodyforce.cu"
  "boundaryconditions/gravity.cu"
  "boundaryconditions/rigidbodylevelset.cu"
  "boundaryconditions/planardomain.cu"
  "boundaryconditions/nodedomain.cu"

  # materials
  "materials/linearelastic.cu"
  "materials/newtonfluid.cu"
  "materials/localrheo.cu"
  "solver/solver.cu"


  "shapefunction/shapefunction.cu"
  "solver/usl/usl.cu"
  # # future
  # # "solver/apic/apic.cu"
  # "solver/musl/musl.cu"
  # "solver/tlmpm/tlmpm.cu"

  "particles/particles.cu"
  "nodes/nodes.cu"


  "common/helper.cu"
  "common/global_settings.cu"
  "common/output.cu"

  "common/tools.cu"
  "spatialpartition/spatialpartition.cu"



)

add_library(pyroclastmpm ${PyroclastSrc})

target_link_libraries(pyroclastmpm PRIVATE Eigen3::Eigen ${VTK_LIBRARIES} )

# target_link_libraries(pyroclastmpm PRIVATE Eigen3::Eigen )


if(BUILD_CUDA)
  target_link_libraries(pyroclastmpm PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(pyroclastmpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    --expt-relaxed-constexpr
    -x cu
    >
  )
  set_target_properties(pyroclastmpm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

else()
#let g++ compile cuda files
set_source_files_properties(${PyroclastSrc} PROPERTIES LANGUAGE CXX)
endif()

vtk_module_autoinit(
TARGETS pyroclastmpm
MODULES ${VTK_LIBRARIES}
)

target_include_directories(pyroclastmpm PUBLIC
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
