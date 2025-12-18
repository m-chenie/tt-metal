# CMakeLists.txt
cmake_minimum_required(VERSION 3.22...3.30)
project(diode_equation)

add_executable(diode_equation diode_equation_single_v2.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(diode_equation PUBLIC TT::Metalium)
