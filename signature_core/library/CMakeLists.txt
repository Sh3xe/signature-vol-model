cmake_minimum_required(VERSION 3.22)
project(signature_core_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library setup (to be used by the testbench)
add_library(signature_core STATIC src/signatures.cpp)

target_include_directories(signature_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Python bindings (to be used by python)
add_subdirectory(extern/pybind11)

pybind11_add_module(signature_core_py src/bindings.cpp)

target_link_libraries(signature_core_py PRIVATE signature_core)