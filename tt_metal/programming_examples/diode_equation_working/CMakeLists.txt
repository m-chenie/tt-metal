cmake_minimum_required(VERSION 3.22...3.30)
project(diode_equation_single_example)

add_executable(diode_equation_single_example)
target_sources(diode_equation_single_example PRIVATE diode_equation_single_example.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(diode_equation_single_example PUBLIC TT::Metalium)

target_compile_definitions(
    diode_equation_single_example
    PRIVATE
        OVERRIDE_KERNEL_PREFIX="tt_metal/programming_examples/"
)
