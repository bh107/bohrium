cmake_minimum_required(VERSION 2.8)

set(VE_UNI true CACHE BOOL "VE-UNI: Build the Universial code generator.")
if(NOT VE_UNI)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_ve_uni SHARED ${SRC})

target_link_libraries(bh_ve_uni bh)

install(TARGETS bh_ve_uni DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)


#
#   The rest of the this file is finding the compiler and flags to write in the config file
#

include(CheckCCompilerFlag)
include(FeatureSummary)

# Import and detect OpenMP features
find_package(OpenMP COMPONENTS bohrium)
set_package_properties(OpenMP PROPERTIES TYPE RECOMMENDED PURPOSE "Multicore processing, essential for performance of the CPU VE.")
if(OPENMP_FOUND)
    # Check for the SIMD flag
    set(OpenMP_SIMD_C_FLAGS "${OpenMP_C_FLAGS} ${OpenMP_C_FLAGS}-simd")
    check_c_compiler_flag("${OpenMP_SIMD_C_FLAGS}" OPENMP_SIMD_FLAG_FOUND)
    # If the openmp-simd flag is supported, we should use it
    if(OPENMP_SIMD_FLAG_FOUND)
        set(OpenMP_C_FLAGS "${OpenMP_SIMD_C_FLAGS}")
    endif()

    # Check for SIMD support
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    check_c_source_compiles("
    #include <omp.h>
    int main() {
      int i;
      int sum=42;
      #pragma omp parallel for simd reduction(+:sum)
      for(i=0; i<100; ++i)
        sum += i;
      return 0;
    }
    " OPENMP_SIMD_FOUND)
    unset(CMAKE_REQUIRED_FLAGS)
endif()

# Check highly RECOMMENDED flags
check_c_compiler_flag(-O3 FLAG_03_FOUND)
check_c_compiler_flag(-march=native FLAG_MARCH_NATIVE_FOUND)
check_c_compiler_flag(-Werror FLAG_WERROR_FOUND)

#
# JIT-compiler capabilities: optimization, and parallelization
#
if(APPLE)
    #
    # Fallback on APPLE, this will run but without optimizations and parallelization.
    #
    set(VE_UNI_COMPILER_INC "")
    set(VE_UNI_COMPILER_FLG "-x c -dynamiclib -arch i386 -arch x86_64")
    set(VE_UNI_COMPILER_LIB "-lm -L${VE_UNI_COMPILER_INC}/lib -lbh")
    set(VE_UNI_COMPILER_EXT "")
else()
    #
    # Mandatory flags
    set(VE_UNI_COMPILER_FLG "-x c -fPIC -shared ${C99_FLAG}")
    #
    # Optimizations
    if (FLAG_03_FOUND)                                                      # Optimization Level
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} -O3")
    endif()
    if (FLAG_MARCH_NATIVE_FOUND)                                            # Machine specific code
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} -march=native")
    endif()
    if (FLAG_WERROR_FOUND)                                                  # Optimization Level
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} -Werror")
    endif()
    #
    # Parallelization
    #
    if(OPENMP_FOUND)
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} ${OpenMP_C_FLAGS}") # OpenMP
    endif()
endif()

set(VE_UNI_COMPILER_CMD "${CMAKE_C_COMPILER}"       CACHE STRING                "VE_UNI: JIT-Compiler")
set(VE_UNI_COMPILER_INC "-I${CMAKE_INSTALL_PREFIX}/share/bohrium/include")
set(VE_UNI_COMPILER_INC "${VE_UNI_COMPILER_INC}"                  CACHE STRING  "VE_UNI: JIT-Compiler includes")
set(VE_UNI_COMPILER_LIB "-lm -L${CMAKE_INSTALL_PREFIX}/lib -lbh"  CACHE STRING  "VE_UNI: JIT-Compiler libraries")
set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG}"                  CACHE STRING  "VE_UNI: JIT-Compiler flags")
set(VE_UNI_COMPILER_OPENMP       ${OPENMP_FOUND}                  CACHE BOOL    "VE_UNI: JIT-Compiler use OpenMP")
set(VE_UNI_COMPILER_OPENMP_SIMD  ${OPENMP_SIMD_FOUND}             CACHE BOOL    "VE_UNI: JIT-Compiler use OpenMP-SIMD")

# We need to cleanup the variables for the config file
if(VE_UNI_COMPILER_OPENMP)
    set(_VE_UNI_COMPILER_OPENMP "true" CACHE INTERNAL "config version")
else()
    set(_VE_UNI_COMPILER_OPENMP "false" CACHE INTERNAL "config version")
endif()
if(VE_UNI_COMPILER_OPENMP_SIMD)
    set(_VE_UNI_COMPILER_OPENMP_SIMD "true" CACHE INTERNAL "config version")
else()
    set(_VE_UNI_COMPILER_OPENMP_SIMD "false" CACHE INTERNAL "config version")
endif()
