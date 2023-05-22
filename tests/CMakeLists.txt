set(
  PyroclastSrc_test
  "test_main.cu"
)
enable_testing()

add_executable(tests ${PyroclastSrc_test})

target_link_libraries(tests PRIVATE pyroclastmpm)

target_link_libraries(tests PRIVATE GTest::gtest_main Eigen3::Eigen ${VTK_LIBRARIES})

if(BUILD_CUDA)
  target_link_libraries(tests PRIVATE CUDA::cudart CUDA::cuda_driver)
  target_compile_options(tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --relocatable-device-code=true
    -O3
    --expt-relaxed-constexpr
    >
  )
  set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  # let g++ compile cuda files
  set_source_files_properties(${PyroclastSrc_test} PROPERTIES LANGUAGE CXX)
endif()

vtk_module_autoinit(
  FORCE_STATIC TARGETS tests
  MODULES ${VTK_LIBRARIES}
)

target_include_directories(tests PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

include(GoogleTest)
gtest_discover_tests(tests)
