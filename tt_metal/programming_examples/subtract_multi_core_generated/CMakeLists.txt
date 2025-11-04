cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_subtract_multi_core_generated)

add_executable(metal_example_subtract_multi_core_generated)
target_sources(metal_example_subtract_multi_core_generated PRIVATE subtract_multi_core_generated.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_subtract_multi_core_generated PUBLIC TT::Metalium)
