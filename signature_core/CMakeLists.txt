cmake_minimum_required(VERSION 3.22)
project(testbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(library)

add_executable(testbench testbench.cpp)

target_link_libraries(testbench PRIVATE signature_core)