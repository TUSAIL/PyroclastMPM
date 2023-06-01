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