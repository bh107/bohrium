cmake_minimum_required(VERSION 2.6)

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

set(VERSION_MAJOR "0")
set(VERSION_MINOR "3")
set(VERSION_PATCH "1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(USE_CXX98 false CACHE BOOL "Choose to use the old C++98 standard instead of C++11 (some Bohrium components requires C++11 and will not be skipped.")
message(STATUS "USE_CXX98: ${USE_CXX98}")

#Set global compile flags TODO: support other compilers
if(USE_CXX98)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
endif()

#Some of the modules needs Python in order to build
set(PYTHON "auto" CACHE STRING "The Python interpreter to use. 'auto' is default")
if(NOT (${PYTHON} STREQUAL "auto"))
    set(PYTHON_EXECUTABLE ${PYTHON})
    message(STATUS "User-defined Python Interpreter: ${PYTHON}")
endif()
#External dependencies
find_package(PythonInterp)
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE REQUIRED PURPOSE "Some Bohrium needs Python when building.")

#The core modules
add_subdirectory(core)
add_subdirectory(vem/node)
add_subdirectory(vem/proxy)
add_subdirectory(ve/cpu)
add_subdirectory(filter/pprint)
add_subdirectory(extmethods/matmul)

#The other modules
add_subdirectory(vem/cluster)
add_subdirectory(ve/gpu)
add_subdirectory(extmethods/fftw)

if(NOT USE_CXX98)
    add_subdirectory(bridge/cpp)
    add_subdirectory(bridge/c)
    add_subdirectory(bridge/bhpy)
endif()

install(DIRECTORY thirdparty/Random123/include/Random123 DESTINATION include)

#Install the config file, which depends on global or local installation
configure_file(config.ini.in config.ini)
string(REGEX MATCH "^/home/" ROOT_INSTALL ${CMAKE_INSTALL_PREFIX})
if(${ROOT_INSTALL})
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc)
elseif(NOT EXISTS "$ENV{HOME}/.bohrium/config.ini")
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION "$ENV{HOME}/.bohrium")
endif()

MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)


#Lets try to find SYS_NAME and SYS_VERSION automatically
find_program(LSB_RELEASE_EXECUTABLE lsb_release)
if(LSB_RELEASE_EXECUTABLE)
    # ---- Get the distribution codename ----
    execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -s -c
      OUTPUT_VARIABLE TMP_LSB_CODENAME
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(TOLOWER ${TMP_LSB_CODENAME} LSB_CODENAME)
    # ---- Get the release name ----
    execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -s -r
      OUTPUT_VARIABLE TMP_LSB_RELEASE
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(TOLOWER ${TMP_LSB_RELEASE} LSB_RELEASE)
    # ---- Get the distributor id ----
    execute_process(COMMAND ${LSB_RELEASE_EXECUTABLE} -s -i
      OUTPUT_VARIABLE TMP_LSB_DISTRIBUTOR_ID
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(TOLOWER ${TMP_LSB_DISTRIBUTOR_ID} LSB_DISTRIBUTOR_ID)
    set(SYS_NAME ${LSB_DISTRIBUTOR_ID})
    set(SYS_VERSION ${LSB_RELEASED})
else()
    set(SYS_NAME Unknown)
    set(SYS_VERSION 1.0)
endif()

#Package setup
set(CPACK_GENERATOR "DEB")

set(CPACK_PACKAGE_CONTACT "Bohrium <contact@bh107.org>")
set(CPACK_PACKAGE_DESCRIPTION "Bohrium Runtime System")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python")
set(CPACK_PACKAGE_NAME "Bohrium")
set(CPACK_PACKAGE_VENDOR "Niels Bohr Institut")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "bohrium_${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}_${SYS_NAME}${SYS_VERSION}_${CMAKE_SYSTEM_PROCESSOR}")

#Debian specific
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libctemplate-dev, libfftw3-dev")

include(CPack)


