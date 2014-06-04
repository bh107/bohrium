cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

project(BOHRIUM)
include(FeatureSummary)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

#core modules
add_subdirectory(../../core ${CMAKE_CURRENT_BINARY_DIR}/core)
add_subdirectory(../../vem/node ${CMAKE_CURRENT_BINARY_DIR}/vem/node)
add_subdirectory(../../vem/proxy ${CMAKE_CURRENT_BINARY_DIR}/vem/proxy)
add_subdirectory(../../filter/pprint ${CMAKE_CURRENT_BINARY_DIR}/filter/pprint)
add_subdirectory(../../extmethods/matmul ${CMAKE_CURRENT_BINARY_DIR}/extmethods/matmul)

#optional modules
add_subdirectory(../../ve/cpu ${CMAKE_CURRENT_BINARY_DIR}/ve/cpu)
add_subdirectory(bridge/c ${CMAKE_CURRENT_BINARY_DIR}/bridge/c)

install(DIRECTORY ../../thirdparty/Random123/include/Random123 DESTINATION include)

#Install the config file, which depends on global or local installation
configure_file(../../config.ini.in config.ini)
install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc/bohrium)

MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})

feature_summary(WHAT ALL)

#### NB: changes above this line only has effect when committed in git ####

#Make sure we only run the package script once
if(DEB_SRC_PPA_ONCE)
    return()
endif()

#Package setup
set(CPACK_GENERATOR "DEB")
execute_process(COMMAND git describe --tags --long OUTPUT_VARIABLE CPACK_PACKAGE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
#Lets remove the 'v' char in the version tag (e.g. v0.2-0-g6a2352d => 0.2-0-g6a2352d)
string(SUBSTRING ${CPACK_PACKAGE_VERSION} 1 -1 CPACK_PACKAGE_VERSION)

#set(CPACK_PACKAGE_CONTACT "Bohrium (Bohrium Runtime System) <contact@bh107.org>")
set(CPACK_PACKAGE_CONTACT "Mads R. B. Kristensen <madsbk@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION "Bohrium Runtime System")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python")
set(CPACK_PACKAGE_NAME "Bohrium")
set(CPACK_PACKAGE_VENDOR "http://www.bh107.org")

#Debian specific
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libctemplate-dev, build-essential, libboost-dev")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_DISTRIBUTION_NAME "ubuntu")
set(CPACK_DEBIAN_DISTRIBUTION_RELEASES "trusty")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../../COPYING")

set(CPACK_DEBIAN_BUILD_DEPENDS debhelper cmake python swig libctemplate-dev libboost-dev)
include(CPack)
message("CPACK_PACKAGE_VERSION: ${CPACK_PACKAGE_VERSION}")
include(DebSourcePPA)
