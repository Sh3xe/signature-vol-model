cmake_minimum_required(VERSION 3.22)
project(signature_core_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library setup 
add_library(signature_core STATIC src/signatures.cpp src/utilities.cpp src/pricing.cpp )
set_target_properties(signature_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(signature_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include
)

# Python bindings
add_subdirectory(extern/pybind11)

pybind11_add_module(signature_core_py src/bindings.cpp)
target_link_libraries(signature_core_py PRIVATE signature_core)

set_target_properties(signature_core_py PROPERTIES
    OUTPUT_NAME "signature_core_cpp"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)