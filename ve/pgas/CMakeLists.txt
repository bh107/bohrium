cmake_minimum_required(VERSION 2.8)

set(VE_PGAS false CACHE BOOL "VE-PGAS: Build the Pgas code generator.")
if (NOT VE_PGAS)
    return()
endif ()

add_definitions("-DBH_WITH_MPI")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_ve_pgas SHARED ${SRC})

target_link_libraries(bh_ve_pgas bh)

install(TARGETS bh_ve_pgas DESTINATION ${LIBDIR} COMPONENT bohrium)

#
# The rest of the this file is finding the compiler and flags to write in the config file
#
include(CheckCCompilerFlag)
include(FeatureSummary)

# Check highly RECOMMENDED flags
check_c_compiler_flag(-O3 FLAG_03_FOUND)
check_c_compiler_flag(-march=native FLAG_MARCH_NATIVE_FOUND)
check_c_compiler_flag(-Werror FLAG_WERROR_FOUND)

# JIT-compiler capabilities: optimization, and parallelization
if (APPLE)
    set(VE_PGAS_COMPILER_FLG "-x c -dynamiclib -arch x86_64 -Wno-expansion-to-defined")
else ()
    set(VE_PGAS_COMPILER_FLG "-x c -fPIC -shared ${C99_FLAG}")
endif ()

# Optimizations
if (FLAG_03_FOUND)
    set(VE_PGAS_COMPILER_FLG "${VE_PGAS_COMPILER_FLG} -O3")

    # With Clang, -O3 gives a warning about not being able to vectorize loops
    # TODO: Figure out if we can actually have clang vectorize the loops
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(VE_PGAS_COMPILER_FLG "${VE_PGAS_COMPILER_FLG} -Wno-pass-failed")
    endif ()
endif ()
if (FLAG_MARCH_NATIVE_FOUND)
    set(VE_PGAS_COMPILER_FLG "${VE_PGAS_COMPILER_FLG} -march=native")
endif ()
if (FLAG_WERROR_FOUND)
    set(VE_PGAS_COMPILER_FLG "${VE_PGAS_COMPILER_FLG} -Werror")
endif ()

# Let the user overwrite the compile command
set(VE_PGAS_COMPILER_CMD "mpicc" CACHE STRING "VE_PGAS: JIT-Compiler")
set(VE_PGAS_COMPILER_INC "-I${CMAKE_INSTALL_PREFIX}/share/bohrium/include" CACHE STRING "VE_PGAS: JIT-Compiler includes")
set(VE_PGAS_COMPILER_LIB "-lm -L${CMAKE_INSTALL_PREFIX}/${LIBDIR} -lbh" CACHE STRING "VE_PGAS: JIT-Compiler libraries")
set(VE_PGAS_COMPILER_FLG "${VE_PGAS_COMPILER_FLG}" CACHE STRING "VE_PGAS: JIT-Compiler flags")
