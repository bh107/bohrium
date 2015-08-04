cmake_minimum_required(VERSION 2.8)
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(ExternalProject)
include(FeatureSummary)

if (APPLE)  # Build multi-arch OSX binaries
    set(CMAKE_OSX_ARCHITECTURES "i386;x86_64" CACHE STRING "Sets multi-arch (fat) binaries for OSX builds")
endif()

# We need this before 'project(BOHRIUM)' since cmake defines CMAKE_BUILD_TYPE at project()
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are:
    None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo, and MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are:
    None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo, and MinSizeRel.")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

project(BOHRIUM)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)                                     # .local is default
    set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "The install prefix (default path is ~/.local)" FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")    # Shared modules

set(BENCHMARK_DIR "${CMAKE_SOURCE_DIR}/benchmark")                                  # Benchmarks (npbackend)
message(STATUS "Benchmark-dir: ${BENCHMARKS_DIR}")

#
# External dependencies
#

find_package(PythonInterp)                                                          # Used for autogen
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED PURPOSE "Python is required to build Bohrium")
message(STATUS "Python found: ${PYTHON_EXECUTABLE}")

find_package(Boost REQUIRED COMPONENTS serialization filesystem system thread)      # Everything
set_package_properties(Boost PROPERTIES DESCRIPTION "Boost C++ source libraries" URL "www.boost.org")
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "Boost is required to build Bohrium")
include_directories(${Boost_INCLUDE_DIRS})

find_package(Cheetah REQUIRED COMPONENTS bohrium)                                   # Used for autogen
set_package_properties(Cheetah PROPERTIES DESCRIPTION "Cheetah is a template module for Python" URL "www.cheetahtemplate.org")
set_package_properties(Cheetah PROPERTIES TYPE REQUIRED PURPOSE "Cheetah is required to build Bohrium")

#
# Thirdparty headers
#
install(DIRECTORY thirdparty/Random123-1.08/include/Random123
        DESTINATION share/bohrium/include COMPONENT bohrium)

#
# Default compile-flags for Bohrium
#
check_c_compiler_flag(-pedantic HAS_C_PEDANTIC)             # C Flags
if (HAS_C_PEDANTIC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic ")
endif()
check_c_compiler_flag(-Wall HAS_C_WALL)
if (HAS_C_WALL)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall ")
endif()
check_c_compiler_flag(-std=c99 HAS_C_C99)
if (HAS_C_C99)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ")
endif()

check_cxx_compiler_flag(-Wall HAS_CXX_PEDANTIC)             # C++ Flags
if (HAS_CXX_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic ")
endif()
check_cxx_compiler_flag(-Wall HAS_CXX_WALL)
if (HAS_CXX_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
endif()
check_cxx_compiler_flag(-std=gnu++0x HAS_CXX_GNU0X)         # Sigh...
if (HAS_CXX_GNU0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x ")
else()
    check_cxx_compiler_flag(-std=c++0x HAS_CXX_CXX11)
    if (HAS_CXX_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
    endif()
endif()

#
# Bohrium components
#
add_subdirectory(core)
add_subdirectory(vem/node)
add_subdirectory(vem/proxy)
add_subdirectory(filter/pprint)
add_subdirectory(filter/fuseprinter)
add_subdirectory(filter/pricer)
add_subdirectory(filter/dimclean)
add_subdirectory(filter/bccon)
add_subdirectory(filter/bcexp)
add_subdirectory(fuser/singleton)
add_subdirectory(fuser/topological)
add_subdirectory(fuser/gentle)
add_subdirectory(fuser/optimal)
add_subdirectory(fuser/greedy)
add_subdirectory(vem/cluster)
add_subdirectory(ve/gpu)
add_subdirectory(extmethods/fftw)
add_subdirectory(extmethods/visualizer)
add_subdirectory(bridge/cpp)
add_subdirectory(bridge/c)
add_subdirectory(bridge/npbackend)
add_subdirectory(bridge/NumCIL)
add_subdirectory(bridge/cil)
add_subdirectory(test)

#
# Building the CPU engine is done as an external-project.
# This is an attempt to be able to compile with commercial
# compilers such as pgi and pathscale.
#
set(VE_CPU true CACHE BOOL "VE-CPU: Build CPU engine.")
set(VE_CPU_PROFILING false CACHE BOOL "VE-CPU: Enabling CPU profiling.")
set(VE_CPU_DEBUGGING false CACHE BOOL "VE-CPU: Enable debugging.")
set(VE_CPU_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "VE-CPU: C compiler")
set(VE_CPU_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "VE-CPU: C++ compiler")
 
if(VE_CPU)
ExternalProject_Add(
    BH_VE_CPU
    DEPENDS bh
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ve/cpu
    CMAKE_CACHE_ARGS
        -DBH_SOURCE_DIR:PATH=${CMAKE_SOURCE_DIR}
        -DBH_BINARY_DIR:PATH=${CMAKE_BINARY_DIR}
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER:PATH=${VE_CPU_C_COMPILER}
        -DCMAKE_CXX_COMPILER:PATH=${VE_CPU_CXX_COMPILER}
        -DVE_CPU_PROFILING:STRING=${VE_CPU_PROFILING}
        -DVE_CPU_DEBUGGING:STRING=${VE_CPU_DEBUGGING}
)
endif()

#
# Bohrium configuration file, depends on global or local installation
# 
# The configuration file is written from within the VE-CPU,
# because VE-CPU needs to add/modify various settings based on
# compiler and it seems like it is not possible to retrieve those
# values from ExternalProject_Add except for CMAKE_BINARY_DIR
# and the like...
#

MESSAGE(STATUS "Default c compile flags:" ${CMAKE_C_COMPILE_FLAGS})
MESSAGE(STATUS "Default c debug flags:" ${CMAKE_C_FLAGS_DEBUG})
MESSAGE(STATUS "Default c release flags:" ${CMAKE_C_FLAGS_RELEASE})
MESSAGE(STATUS "Default c min size flags:" ${CMAKE_C_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Default c flags:" ${CMAKE_C_FLAGS})
MESSAGE(STATUS "Default cxx compile flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Default cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Default cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Default cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Default cxx flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)
