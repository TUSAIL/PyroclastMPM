# "boundaryconditions/rigidsurface/rigidsurface.cu"
# "boundaryconditions/rigidsurface/rigidsurface_kernels.cu"
set(
  PyroclastSrc

  # "boundaryconditions/bodyforce/bodyforce_kernels.cu"
  # "boundaryconditions/bodyforce/bodyforce.cu"
  # "boundaryconditions/gravity/gravity.cu"
  # "boundaryconditions/gravity/gravity_kernels.cu"
  # "boundaryconditions/rigidparticles/rigidparticles.cu"
  # "boundaryconditions/rigidparticles/rigidparticles_kernels.cu"
  # "boundaryconditions/planardomain/planardomain.cu"
  # "boundaryconditions/planardomain/planardomain_kernels.cu"
  # "boundaryconditions/nodedomain/nodedomain.cu"
  # "boundaryconditions/nodedomain/nodedomain_kernels.cu"
  # "materials/linearelastic/linearelasticmat.cu"
  # "materials/linearelastic/linearelasticmat_kernels.cu"
  # "materials/newtonfluid/newtonfluidmat.cu"
  # "materials/newtonfluid/newtonfluidmat_kernels.cu"

  # "materials/elastoplastic/elastoplasticmat_kernels.cu"
  # "materials/elastoplastic/elastoplasticmat.cu"
  # "materials/localrheo/localrheomat_kernels.cu"
  # "materials/localrheo/localrheomat.cu"
  # "materials/druckerprager/druckerpragermat_kernels.cu"
  # "materials/druckerprager/druckerpragermat.cu"
  # "solver/solver.cu"
  # "shapefunctions/shapefunction_kernels.cu"
  # "solver/usl/usl.cu"
  # "solver/usl/usl_kernels.cu"
  # "solver/musl/musl.cu"
  # "solver/musl/musl_kernels.cu"
  # "solver/tlmpm/tlmpm.cu"
  # "solver/tlmpm/tlmpm_kernels.cu"

  # "particles/particles.cu"
  # "particles/particles_kernels.cu"
  "nodes/nodes.cpp"

  # "nodes/nodes_kernels.cu"

  "common/helper.cpp"
  "common/global_settings.cpp"
  # "common/output.cu"

  # "common/tools.cu"
  # "spatialpartition/spatialpartition.cu"
  # "spatialpartition/spatialpartition_kernels.cu"

  # # future
  # # "materials/nonlocalngf/nonlocalngfmat_kernels.cu"
  # # "materials/nonlocalngf/nonlocalngfmat.cu"

  # # "solver/apic/apic.cu"
  # # "solver/apic/apic_kernels.cu"
)

add_library(pyroclastmpm ${PyroclastSrc})

# target_link_libraries(pyroclastmpm PRIVATE Eigen3::Eigen Thrust ${VTK_LIBRARIES} )

target_link_libraries(pyroclastmpm PRIVATE Eigen3::Eigen )


if(BUILD_CUDA)
  target_link_libraries(pyroclastmpm PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(pyroclastmpm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    --expt-relaxed-constexpr
    >
  )
  set_target_properties(pyroclastmpm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# vtk_module_autoinit(
# TARGETS pyroclastmpm
# MODULES ${VTK_LIBRARIES}
# )

target_include_directories(pyroclastmpm PUBLIC
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
