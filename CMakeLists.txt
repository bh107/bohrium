cmake_minimum_required(VERSION 2.8)

project(BOHRIUM)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "3")
set(VERSION_PATCH "1")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

add_subdirectory(core)
add_subdirectory(vem/node)
add_subdirectory(ve/cpu)
add_subdirectory(bridge/cpp)

install(DIRECTORY thirdparty/Random123/include/Random123 DESTINATION include)





#Package setup
set(CPACK_GENERATOR "DEB")

set(CPACK_PACKAGE_CONTACT "Bohrium <contact@bh107.org>")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION "Bohrium Runtime System")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Bohrium Runtime System: Automatic Vector Parallelization in C, C++, CIL, and Python")
set(CPACK_PACKAGE_NAME "Bohrium")
set(CPACK_PACKAGE_VENDOR "Niels Bohr Institut")
set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "bohrium-${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}_x86_64")

#Debian specific
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libctemplate-dev (>= 2)")



INCLUDE(CPack)
