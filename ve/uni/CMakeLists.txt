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

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(FeatureSummary)

# External code, libraries, and macros
find_package(OpenMP COMPONENTS bohrium)
set_package_properties(OpenMP PROPERTIES TYPE RECOMMENDED PURPOSE "Multicore processing, essential for performance of the CPU VE.")

#find_package(OpenACC COMPONENTS bohrium)
#set_package_properties(OpenACC PROPERTIES TYPE RECOMMENDED PURPOSE "For experimental support for offloading to GPU in CPU VE.")

set(OpenACC false CACHE BOOL "For OpenACC support.")
if (NOT OpenACC)
    set(OPENACC_FOUND false)
endif()

# Check highly RECOMMENDED flags
check_c_compiler_flag(-O3 FLAG_03_FOUND)
check_c_compiler_flag(-march=native FLAG_MARCH_NATIVE_FOUND)
check_c_compiler_flag("--param vect-max-version-for-alias-checks=100" PARAM_VECTMAX_FOUND)
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
    if (PARAM_VECTMAX_FOUND)                                                # Upper bound on alias-checks
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} --param vect-max-version-for-alias-checks=100")
    endif()
    if (FLAG_WERROR_FOUND)                                                      # Optimization Level
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} -Werror")
    endif()
    #
    # Parallelization
    #
    # Assume that when OpenACC is available then we want to use it.
    #
    if (OPENACC_FOUND)
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} ${OpenACC_C_FLAGS}")# OpenACC
    elseif(OPENMP_FOUND)
        set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} ${OpenMP_C_FLAGS}") # OpenMP
        if (FLAG_OPENMP_SIMD_FOUND)
            set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG} -fopenmp-simd") # OpenMP - SIMD
        endif()
    endif()
endif()

set(VE_UNI_COMPILER_CMD "${CMAKE_C_COMPILER}"       CACHE STRING                            "VE_UNI: JIT-Compiler")
set(VE_UNI_COMPILER_INC "-I${CMAKE_INSTALL_PREFIX}/include")
set(VE_UNI_COMPILER_INC "${VE_UNI_COMPILER_INC} -I${CMAKE_INSTALL_PREFIX}/include/bohrium")
set(VE_UNI_COMPILER_INC "${VE_UNI_COMPILER_INC} -I${CMAKE_INSTALL_PREFIX}/share/bohrium/include")
set(VE_UNI_COMPILER_INC "${VE_UNI_COMPILER_INC}"    CACHE STRING                            "VE_UNI: JIT-Compiler includes")
set(VE_UNI_COMPILER_LIB "-lm -L${CMAKE_INSTALL_PREFIX}/lib -lbh"  CACHE STRING              "VE_UNI: JIT-Compiler libraries")
set(VE_UNI_COMPILER_FLG "${VE_UNI_COMPILER_FLG}"    CACHE STRING                            "VE_UNI: JIT-Compiler flags")
