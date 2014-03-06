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

set(VERSION_MAJOR "0")
set(VERSION_MINOR "3")
set(VERSION_PATCH "1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

#Set global compile flags TODO: support other compilers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

add_subdirectory(core)
add_subdirectory(vem/node)
add_subdirectory(vem/proxy)
add_subdirectory(vem/cluster)
add_subdirectory(ve/cpu)
add_subdirectory(bridge/cpp)
add_subdirectory(bridge/c)
add_subdirectory(extmethods/matmul)
add_subdirectory(extmethods/fftw)

#add_subdirectory(bridge/numpy)

install(DIRECTORY thirdparty/Random123/include/Random123 DESTINATION include)

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
