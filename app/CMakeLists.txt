

add_executable(pyroclastmpm_app pyroclastmpm_app.cpp)

target_link_libraries(pyroclastmpm_app PRIVATE pyroclastmpm)

target_link_libraries(pyroclastmpm_app PRIVATE Eigen3::Eigen CUDA::cudart CUDA::cuda_driver ${VTK_LIBRARIES})


target_compile_options(pyroclastmpm_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    >
)

set_target_properties(pyroclastmpm_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

vtk_module_autoinit(
    TARGETS pyroclastmpm_app
    MODULES ${VTK_LIBRARIES}
)

