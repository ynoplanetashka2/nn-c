cmake_minimum_required(VERSION 3.16)

project(nn)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

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
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.c"
)

# Define the executable
add_executable(out ${SOURCES})
