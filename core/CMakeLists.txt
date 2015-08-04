cmake_minimum_required(VERSION 2.8)                     # cmake features
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)             
include(FeatureSummary)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(BH_CORE)

find_package(Boost REQUIRED COMPONENTS serialization filesystem system thread)      # Everything
set_package_properties(Boost PROPERTIES DESCRIPTION "Boost C++ source libraries" URL "www.boost.org")
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "Boost is required to build Bohrium")
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${BOHRIUM_SOURCE_DIR}/include)
file(GLOB SRC *.cpp)

add_library(bh SHARED ${SRC} bh_opcode.cpp)

# Rules for how to generate bh_opcode.cpp and bh_opcode.h
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bh_opcode.cpp COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py VERBATIM)
set_source_files_properties(bh_opcode.cpp PROPERTIES GENERATED TRUE)

add_subdirectory(iniparser)                     # bh_component depends on iniparser
target_link_libraries(bh iniparser)
target_link_libraries(bh ${CMAKE_DL_LIBS})      # bh_component depends on dlopen etc.
target_link_libraries(bh ${Boost_LIBRARIES})    # A shit ton of stuff depends on boost

install(TARGETS bh DESTINATION lib COMPONENT bohrium)
install(DIRECTORY ${BOHRIUM_SOURCE_DIR}/include/ DESTINATION include/bohrium COMPONENT bohrium)

