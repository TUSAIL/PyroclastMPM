# perform cmake -DCMAKE_BUILD_TYPE=Release ../pyroclastmpm/
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(pyroclastmpm VERSION 0.1.0 LANGUAGES C CXX)

# Define the minimum C++ standard that is required
set(CMAKE_POLICY_DEFAULT_CMP0025 NEW)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
option(BUILD_CUDA "Enable CUDA Acceleration" OFF)
# option(BUILD_OPENMP "Enable CUDA Acceleration" OFF) # later
option(BUILD_PYTHON "Enable building of Python bindings" ON)
option(BUILD_DOCS "Enable building of documentation" OFF)
option(BUILD_TESTING "Enable testing" ON)
option(CMAKE_BUILD_TYPE "Make release build" Release)

if(BUILD_CUDA)
    project(pyroclastmpm VERSION 0.0.1 LANGUAGES C CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    project(pyroclastmpm VERSION 0.0.1 LANGUAGES C CXX)
endif()

# Thrust
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust HOST)

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

    # IOXdmf
)

include_directories(${VTK_INCLUDE_DIRS})

# main ompile the library
add_subdirectory(src)

# # compile the tests
# if(BUILD_TESTING)
#     include(FetchContent)
#     FetchContent_Declare(
#         googletest
#         GIT_REPOSITORY https://github.com/google/googletest.git
#         GIT_TAG release-1.12.1
#     )
#     # For Windows: Prevent overriding the parent project's compiler/linker settings
#     set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#     FetchContent_MakeAvailable(googletest)
#     add_subdirectory(tests)
# endif()

# if(BUILD_DOCS)
#     add_subdirectory(doc)
# endif()

# if(BUILD_PYTHON)
#     add_subdirectory(ext/pybind11)
#     add_subdirectory(python/pyroclastmpm)
# endif()

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
