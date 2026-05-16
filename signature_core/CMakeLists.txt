cmake_minimum_required(VERSION 3.22)
project(signature_core_cpp LANGUAGES CXX)

add_subdirectory(extern/pybind11)

pybind11_add_module(signature_core_cpp signatures.cpp)

target_include_directories(signature_core_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include)

install(TARGETS signature_core_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib RENAME signature_core_cpp.so)