cmake_minimum_required(VERSION 3.18)
project(ocml_log_ulp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-file dual-mode test: HIP when built with hipcc, CPU otherwise.
# To build with hipcc, invoke directly:
#   hipcc -O2 -o test_ocml_log_ulp test_ocml_log_ulp.hip.cpp
#
# CMake builds the CPU fallback only — sufficient for methodology
# validation without AMD hardware.

add_executable(test_ocml_log_ulp_cpu test_ocml_log_ulp.hip.cpp)
target_link_libraries(test_ocml_log_ulp_cpu PRIVATE m)
target_compile_options(test_ocml_log_ulp_cpu PRIVATE -O2 -Wall)
