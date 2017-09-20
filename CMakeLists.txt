cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_MACOSX_RPATH 1) # suppress warning about MACOSX_RPATH

# We need this before 'project(BOHRIUM)' since cmake defines CMAKE_BUILD_TYPE at project()
if(DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are:
    None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo, and MinSizeRel.")
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are:
    None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used), Debug, Release, RelWithDebInfo, and MinSizeRel.")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Everyting else should come after 'project(BOHRIUM)' since 'project()' defines many of the CMAKE_BUILD_* vars
project(BOHRIUM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")    # Shared modules
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
include(FeatureSummary)
include(LibVsLib64) # Sets LIBDIR

message(STATUS "LIBDIR: ${LIBDIR}")

# Write all compile commands to 'compile_commands.json', which can be used by YouCompleteMe
# and other auto-completion software
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

# For the Clion IDE
ADD_CUSTOM_TARGET(run_install COMMAND ${CMAKE_MAKE_PROGRAM} install)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)                                     # .local is default
    set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "The install prefix (default path is ~/.local)" FORCE)
endif()

set(BENCHMARK_DIR "${CMAKE_SOURCE_DIR}/benchmark")                                  # Benchmarks (npbackend)
message(STATUS "Benchmark-dir: ${BENCHMARKS_DIR}")

####################################
# Default compile-flags for Bohrium
check_c_compiler_flag(-pedantic HAS_C_PEDANTIC)             # C Flags
if (HAS_C_PEDANTIC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic ")
endif()
check_c_compiler_flag(-Wall HAS_C_WALL)
if (HAS_C_WALL)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall ")
endif()

########################################
# C99 for restrict, for(var name=0;...)
# gnu99 for MAP_ANONYMOUS
check_c_compiler_flag(-std=gnu99 HAS_GCC_GNU99)
if (HAS_GCC_GNU99)
    set(C99_FLAG "${C99_FLAG} -std=gnu99 ")
else()
    check_c_compiler_flag(-std=c99 HAS_GCC_C99)
    if (HAS_GCC_C99)
        set(C99_FLAG "${C99_FLAG} -std=c99 ")
    else()
        check_c_compiler_flag(-c99 HAS_PGI_C99)
        if (HAS_PGI_C99)
            set(C99_FLAG "${C99_FLAG} -c99 ")
        endif()
    endif()
endif()
if (HAS_GCC_GNU99 OR HAS_GCC_C99 OR HAS_PGI_C99)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C99_FLAG}")
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
    check_cxx_compiler_flag(-std=c++11 HAS_CXX_CXX11)
    if(HAS_CXX_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
    endif()
endif()

###################
# Warning as error
set(USE_WERROR OFF CACHE BOOL "Use the -Werror flag if supported")
if (USE_WERROR)
    check_c_compiler_flag(-Werror HAS_C_WERROR)
    if (HAS_C_WERROR)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    endif()

    check_cxx_compiler_flag(-Werror HAS_CXX_WERROR)
    if (HAS_CXX_WERROR)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()
endif()

# Let's get the BH_VERSION_MAJOR, _MINOR, and _PATCH variables
include(GetVersion)

########################
# External dependencies
find_package(PythonInterp)                                                          # Used for autogen
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED PURPOSE "Python is required to build Bohrium")
message(STATUS "Python found: ${PYTHON_EXECUTABLE}")

find_package(Boost REQUIRED COMPONENTS serialization filesystem system regex)              # Everything
set_package_properties(Boost PROPERTIES DESCRIPTION "Boost C++ source libraries" URL "www.boost.org")
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "Boost is required to build Bohrium")
include_directories(${Boost_INCLUDE_DIRS})

#####################
# Bohrium components
set(OPENMP_LIBS "")
set(OPENCL_LIBS "")

add_subdirectory(core)
add_subdirectory(vem/node)
#add_subdirectory(vem/proxy) TODO: fix and enable
add_subdirectory(filter/pprint)
add_subdirectory(filter/bccon)
add_subdirectory(filter/bcexp)
add_subdirectory(filter/noneremover)
add_subdirectory(ve/openmp)
add_subdirectory(ve/opencl)
add_subdirectory(ve/cuda)
add_subdirectory(ve/fortran)

add_subdirectory(extmethods/blas)
add_subdirectory(extmethods/clblas)
add_subdirectory(extmethods/visualizer)
add_subdirectory(extmethods/tdma)
add_subdirectory(extmethods/lapack)
add_subdirectory(extmethods/opencv)

add_subdirectory(bridge/cxx)
add_subdirectory(bridge/c)
add_subdirectory(bridge/npbackend)
add_subdirectory(test)

string(REPLACE ";" ", " OPENMP_LIBS "${OPENMP_LIBS}")
string(REPLACE ";" ", " OPENCL_LIBS "${OPENCL_LIBS}")

#############################
# Install thirdparty headers
install(DIRECTORY thirdparty/Random123-1.09/include/Random123
        DESTINATION share/bohrium/include COMPONENT bohrium)

#########################################
# Install JIT-kernels dependency headers
install(DIRECTORY include/jitk/kernel_dependencies
        DESTINATION share/bohrium/include COMPONENT bohrium)

#############################
# Bohrium configuration file
set(BIN_KERNEL_CACHE_DIR ${CMAKE_INSTALL_PREFIX}/var/bohrium/cache CACHE PATH "Path to the directory of the persistent JIT kernel cache")

configure_file(${CMAKE_SOURCE_DIR}/config.ini.in ${CMAKE_BINARY_DIR}/config.ini)    # Fill out the config-template

set(FORCE_CONFIG_PATH "OFF" CACHE STRING "Manually setting the install path of the config (ini) file")
string(REGEX MATCH "^$ENV{HOME}" ROOT_INSTALL ${CMAKE_INSTALL_PREFIX})
if ("${FORCE_CONFIG_PATH}" STREQUAL "OFF")
    if ("${ROOT_INSTALL}" STREQUAL "")
        install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc/bohrium COMPONENT bohrium)
    else()
        install(CODE "
        if(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
            configure_file(${CMAKE_BINARY_DIR}/config.ini \"\$ENV{HOME}/.bohrium/config.ini\" COPYONLY)
        endif()
        " COMPONENT bohrium)
    endif()
else()
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION ${FORCE_CONFIG_PATH} COMPONENT bohrium)
endif()

find_package(Jupyter)
if(JUPYTER_FOUND)
  if ("${ROOT_INSTALL}" STREQUAL "")
      message(STATUS "Can't install IPython magic commands globally.")
  else()
      install(CODE "
      if(NOT EXISTS \"\$ENV{HOME}/.ipython/profile_default/startup/50-bohrium.py\")
          configure_file(${CMAKE_SOURCE_DIR}/ipython-magic.py \"\$ENV{HOME}/.ipython/profile_default/startup/50-bohrium.py\" COPYONLY)
      endif()
      " COMPONENT bohrium)
  endif()
endif()

MESSAGE(STATUS "C compiler:" ${CMAKE_C_COMPILER})
MESSAGE(STATUS "C compile flags:" ${CMAKE_C_COMPILE_FLAGS})
MESSAGE(STATUS "C debug flags:" ${CMAKE_C_FLAGS_DEBUG})
MESSAGE(STATUS "C release flags:" ${CMAKE_C_FLAGS_RELEASE})
MESSAGE(STATUS "C min size flags:" ${CMAKE_C_FLAGS_MINSIZEREL})
MESSAGE(STATUS "CXX compiler:" ${CMAKE_CXX_COMPILER})
MESSAGE(STATUS "CXX compile flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "CXX debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "CXX release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "CXX min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "CXX flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)
