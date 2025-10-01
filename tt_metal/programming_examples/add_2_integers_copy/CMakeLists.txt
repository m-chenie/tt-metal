cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_add_2_integers_copy)

add_executable(metal_example_add_2_integers_copy)
target_sources(metal_example_add_2_integers_copy PRIVATE add_2_integers_copy.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_add_2_integers_copy PUBLIC TT::Metalium)
