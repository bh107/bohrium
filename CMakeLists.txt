cmake_minimum_required(VERSION 2.8)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

if (APPLE)
    # Build multi-arch OSX binaries
    set(CMAKE_OSX_ARCHITECTURES "i386;x86_64" CACHE STRING "Sets multi-arch (fat) binaries for OSX builds")
endif()

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

# Benchmarks, used by numpy backend
set(BENCHMARK_DIR "${CMAKE_SOURCE_DIR}/benchmark")
message(STATUS "Benchmark-dir: ${BENCHMARKS_DIR}")

# C Flags
check_c_compiler_flag(-pedantic HAS_C_PEDANTIC)
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

# C++ Flags
check_cxx_compiler_flag(-Wall HAS_CXX_PEDANTIC)
if (HAS_CXX_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic ")
endif()
check_cxx_compiler_flag(-Wall HAS_CXX_WALL)
if (HAS_CXX_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
endif()

# Bohrium uses some fancy C++11 and GNU features... sigh...
check_cxx_compiler_flag(-std=gnu++0x HAS_CXX_GNU0X)
if (HAS_CXX_GNU0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x ")
else()
    check_cxx_compiler_flag(-std=c++0x HAS_CXX_CXX11)
    if (HAS_CXX_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
    endif()
endif()

#External dependencies
find_package(PythonInterp)
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED PURPOSE "Python is required to build Bohrium")
message(STATUS "Python found: ${PYTHON_EXECUTABLE}")

# Bohrium Core requires Boost
find_package(Boost REQUIRED COMPONENTS serialization filesystem system thread)
set_package_properties(Boost PROPERTIES DESCRIPTION "Boost C++ source libraries" URL "www.boost.org")
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "Boost is required to build Bohrium")
include_directories(${Boost_INCLUDE_DIRS})

# Cheetah is required to build some parts of Bohrium
find_package(Cheetah REQUIRED COMPONENTS bohrium)
set_package_properties(Cheetah PROPERTIES DESCRIPTION "Cheetah is a template module for Python" URL "www.cheetahtemplate.org")
set_package_properties(Cheetah PROPERTIES TYPE REQUIRED PURPOSE "Cheetah is required to build Bohrium")

#The core modules
add_subdirectory(core)
add_subdirectory(vem/node)
add_subdirectory(vem/proxy)
add_subdirectory(ve/cpu)
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

#The other modules
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

install(DIRECTORY thirdparty/Random123-1.08/include/Random123
        DESTINATION share/bohrium/include COMPONENT bohrium)

#Install the config file, which depends on global or local installation
configure_file(config.ini.in config.ini)
string(REGEX MATCH "^$ENV{HOME}" ROOT_INSTALL ${CMAKE_INSTALL_PREFIX})
if ("${ROOT_INSTALL}" STREQUAL "")
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc/bohrium COMPONENT bohrium)
else()
    install(CODE "
    IF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
        CONFIGURE_FILE(${CMAKE_BINARY_DIR}/config.ini \"\$ENV{HOME}/.bohrium/config.ini\" COPY_ONLY)
    ENDIF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
    " COMPONENT bohrium)
endif()

MESSAGE(STATUS "Compiler c compile flags:" ${CMAKE_C_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler c debug flags:" ${CMAKE_C_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler c release flags:" ${CMAKE_C_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler c min size flags:" ${CMAKE_C_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler c flags:" ${CMAKE_C_FLAGS})
MESSAGE(STATUS "Compiler cxx compile flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)
