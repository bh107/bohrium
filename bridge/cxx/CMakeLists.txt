cmake_minimum_required(VERSION 2.8)
set(BRIDGE_BHXX true CACHE BOOL "BRIDGE-BHXX: Build the BHXX-bridge.")
if(NOT BRIDGE_BHXX)
    return()
endif()

set(SRC_H ${CMAKE_CURRENT_BINARY_DIR}/bhxx/array_operations.hpp)

add_custom_command(
    OUTPUT ${SRC_H}
    COMMAND ${PYTHON_EXECUTABLE} gen_array_operations.py ${CMAKE_CURRENT_BINARY_DIR}/bhxx
    DEPENDS gen_array_operations.py
            ../../core/codegen/opcodes.json
            ../cpp/codegen/element_types.json
            ${CMAKE_CURRENT_SOURCE_DIR}/bhxx/runtime.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/bhxx/array_operations.hpp
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(BHXX ALL DEPENDS ${SRC_H})

install(DIRECTORY bhxx DESTINATION include COMPONENT bohrium)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bhxx/array_operations.hpp DESTINATION include/bhxx COMPONENT bohrium)

add_subdirectory(examples)