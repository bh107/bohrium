cmake_minimum_required(VERSION 2.8)

# Build multi-arch OSX binaries
set(CMAKE_OSX_ARCHITECTURES "i386;x86_64")

#We need this before 'project(BOHRIUM)' since cmake defines CMAKE_BUILD_TYPE at project()
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are:
    None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo, and MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are:
    None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo, and MinSizeRel.")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

project(BOHRIUM)
include(FeatureSummary)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "The install prefix (default path is ~/.local)" FORCE)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(USE_CXX98 false CACHE BOOL "Choose to use the old C++98 standard instead of C++11 (some Bohrium components requires C++11 and will not be skipped.")
message(STATUS "USE_CXX98: ${USE_CXX98}")

# Benchmarks, used by numpy backend
set(BENCHMARK_DIR "${CMAKE_SOURCE_DIR}/benchmark")
message(STATUS "Benchmark-dir: ${BENCHMARKS_DIR}")

#Set global compile flags TODO: support other compilers
if(USE_CXX98)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -Wall")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x -Wall")
endif()

#External dependencies
find_package(PythonInterp)
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED PURPOSE "Python is required to build Bohrium")
message(STATUS "Python found: ${PYTHON_EXECUTABLE}")

# The CPU VE (and possibly other modules) requires Boost
find_package(Boost REQUIRED COMPONENTS serialization )
set_package_properties(Boost PROPERTIES DESCRIPTION "Boost C++ source libraries" URL "www.boost.org")
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "Boost is required to build Bohrium")
include_directories(${Boost_INCLUDE_DIRS})

#The core modules
add_subdirectory(core)
add_subdirectory(vem/node)
add_subdirectory(vem/proxy)
add_subdirectory(ve/cpu)
add_subdirectory(filter/pprint)
add_subdirectory(filter/fuseprinter)
add_subdirectory(fuser/singleton)
add_subdirectory(fuser/topological)
add_subdirectory(fuser/gentle)
add_subdirectory(fuser/optimal)
add_subdirectory(fuser/greedy)
add_subdirectory(extmethods/matmul)

#The other modules
add_subdirectory(vem/cluster)
add_subdirectory(ve/gpu)
add_subdirectory(extmethods/fftw)

#The NumCIL library
add_subdirectory(bridge/NumCIL)

#Non CXX98 compatible libraries
if(NOT USE_CXX98)
    add_subdirectory(bridge/cpp)
    add_subdirectory(bridge/c)
    add_subdirectory(bridge/bhpy)
    add_subdirectory(bridge/cil)
endif()

install(DIRECTORY thirdparty/Random123-1.08/include/Random123 DESTINATION share/bohrium/include COMPONENT bohrium)

# Select the approriate compiler string
if (APPLE)
    set(DISABLE_LINUX_COMPILER "#")
else()
    set(DISABLE_OSX_COMPILER "#")
endif()

#Install the config file, which depends on global or local installation
configure_file(config.ini.in config.ini)
string(REGEX MATCH "^/home/" ROOT_INSTALL ${CMAKE_INSTALL_PREFIX})
if(${ROOT_INSTALL})
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc COMPONENT bohrium)
else()
    install(CODE "
    IF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
        CONFIGURE_FILE(${CMAKE_BINARY_DIR}/config.ini \"\$ENV{HOME}/.bohrium/config.ini\" COPY_ONLY)
    ENDIF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
    " COMPONENT bohrium)
endif()

MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)
