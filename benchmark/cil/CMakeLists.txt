cmake_minimum_required(VERSION 2.8)
set(BENCHMARK_CIL true CACHE BOOL "BENCHMARK-CIL: CIL-BENCHMARKs.")
if(NOT BENCHMARK_CIL)
    return()
endif()

#Add all benchmarks
add_subdirectory(blackscholes)
add_subdirectory(jacobi)
add_subdirectory(nbody)
add_subdirectory(nice)
add_subdirectory(reactiondiffusion)
add_subdirectory(shallow_water)
add_subdirectory(Utilities)
