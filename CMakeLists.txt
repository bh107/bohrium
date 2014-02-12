cmake_minimum_required(VERSION 2.8)

project(BOHRIUM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(BH_INC_DIR ${BOHRIUM_SOURCE_DIR}/include)

add_subdirectory(${BOHRIUM_SOURCE_DIR}/core)
add_subdirectory(${BOHRIUM_SOURCE_DIR}/vem/node)
add_subdirectory(${BOHRIUM_SOURCE_DIR}/ve/cpu)
add_subdirectory(${BOHRIUM_SOURCE_DIR}/bridge/cpp)


install(DIRECTORY thirdparty/Random123/include/Random123 DESTINATION include )
