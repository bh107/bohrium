cmake_minimum_required(VERSION 2.8)
set(BRIDGE_C true CACHE BOOL "BRIDGE-C: Build the C-bridge.")
if(NOT BRIDGE_C)
    return()
endif()

include_directories(../../include)
include_directories(../cpp/bxx)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.h
           ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.cpp
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_types.py ${CMAKE_CURRENT_BINARY_DIR}/out
    DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/gen_types.py
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.h
           ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.cpp
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_array_operations.py ${CMAKE_CURRENT_BINARY_DIR}/out
    DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/gen_array_operations.py
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.h
           ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.cpp
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_specials.py ${CMAKE_CURRENT_BINARY_DIR}/out
    DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/gen_specials.py
)

add_library(bhc SHARED ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.cpp
                       ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.cpp
                       ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.cpp
)

#We depend on bh.so
target_link_libraries(bhc bh)

