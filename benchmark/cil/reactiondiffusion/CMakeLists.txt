cmake_minimum_required(VERSION 2.8)

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(BuildMonoBenchmark)

GET_FILENAME_COMPONENT(projname ${CMAKE_CURRENT_BINARY_DIR} NAME)

build_mono_benchmark_project(${projname} ${projname}.exe numcil_test_Utilities)
