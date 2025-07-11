cmake_minimum_required(VERSION 3.16)

project(nn)
enable_language(C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Optional: Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Place all executables in bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(EXISTS "${CMAKE_SOURCE_DIR}/local.cmake")
    include("${CMAKE_SOURCE_DIR}/local.cmake")
endif()

add_compile_options(
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Recursively add all .c files in src/
file(GLOB_RECURSE C_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.c"
)

# Add option to enable CUDA
option(USE_CUDA "Enable CUDA support")

# If USE_CUDA is ON, check for CUDA
if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found. Building with CUDA support.")
        set(CMAKE_CUDA_ARCHITECTURES OFF)
        enable_language(CUDA)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        # Recursively add all .cu files in src/
        file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/src/*.cu"
        )
        # Build CUDA code as a static library
        add_library(cuda_kernels STATIC ${CUDA_SOURCES})
        target_link_libraries(cuda_kernels PRIVATE CUDA::cudart)
    else()
        message(WARNING "CUDA not found! Building without CUDA support.")
        set(USE_CUDA OFF)
    endif()
endif()

# Define the executable from C sources
add_executable(out ${C_SOURCES})

# Link CUDA library if enabled
if(USE_CUDA)
    target_link_libraries(out PRIVATE cuda_kernels)
    add_compile_definitions(USE_CUDA)
endif()
