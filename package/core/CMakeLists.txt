cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

project(BOHRIUM)
include(FeatureSummary)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

#The core modules
add_subdirectory(../../core ${CMAKE_CURRENT_BINARY_DIR}/core)
add_subdirectory(../../vem/node ${CMAKE_CURRENT_BINARY_DIR}/vem/node)
add_subdirectory(../../vem/proxy ${CMAKE_CURRENT_BINARY_DIR}/vem/proxy)
add_subdirectory(../../filter/pprint ${CMAKE_CURRENT_BINARY_DIR}/filter/pprint)
add_subdirectory(../../extmethods/matmul ${CMAKE_CURRENT_BINARY_DIR}/extmethods/matmul)

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

set(VERSION_MAJOR "0")
set(VERSION_MINOR "3")
set(VERSION_PATCH "1")

#Package setup
set(CPACK_GENERATOR "DEB")

set(CPACK_PACKAGE_CONTACT "Bohrium (Bohrium Runtime System) <contact@bh107.org>")
set(CPACK_PACKAGE_DESCRIPTION "Bohrium Runtime System")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python")
set(CPACK_PACKAGE_NAME "Bohrium")
set(CPACK_PACKAGE_VENDOR "Niels Bohr Institut")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "bohrium_${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}_${SYS_NAME}${SYS_VERSION}_${CMAKE_SYSTEM_PROCESSOR}")

#Debian specific
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libctemplate-dev")

set(CPACK_DEBIAN_DISTRIBUTION_NAME ubuntu)
set(CPACK_DEBIAN_DISTRIBUTION_RELEASES trusty)
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/../../COPYING)

set(CPACK_DEBIAN_BUILD_DEPENDS debhelper cmake python swig libctemplate-dev)
include(CPack)
include(DebSourcePPA)
