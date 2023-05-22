set(
  PyroclastPyBind

  "_pyroclastmpm.cpp"
  "particles_bind.cu"
  "nodes_bind.cu"
  "boundaryconditions_bind.cu"
  "materials_bind.cu"
  "solver_bind.cu"
  "global_settings_bind.cu"
  "tools_bind.cu"

  # "spatialpartition_bind.cu" // currently no need to import this in python . . .
)

include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Python COMPONENTS Interpreter Development)

find_package(pybind11 CONFIG)

pybind11_add_module(pyroclastmpm_pybind SHARED ${PyroclastPyBind})

set_target_properties(pyroclastmpm_pybind
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

target_link_libraries(pyroclastmpm_pybind PRIVATE pyroclastmpm)

target_link_libraries(pyroclastmpm_pybind PRIVATE GTest::gtest_main Eigen3::Eigen ${VTK_LIBRARIES})

if(BUILD_CUDA)
  target_link_libraries(pyroclastmpm_pybind PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(pyroclastmpm_pybind PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    --expt-relaxed-constexpr
    >
  )
  set_target_properties(pyroclastmpm_pybind PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  # let g++ compile cuda files
  set_source_files_properties(${PyroclastPyBind} PROPERTIES LANGUAGE CXX)
endif()

vtk_module_autoinit(
  FORCE_STATIC TARGETS pyroclastmpm_pybind
  MODULES ${VTK_LIBRARIES}
)

target_include_directories(pyroclastmpm_pybind PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
