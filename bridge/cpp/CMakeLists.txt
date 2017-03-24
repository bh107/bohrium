cmake_minimum_required(VERSION 2.8)
set(BRIDGE_BXX true CACHE BOOL "BRIDGE-BXX: Build the BXX-bridge.")
if(NOT BRIDGE_BXX)
    return()
endif()

#
# Autogeneration - begin
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/bxx/traits.hpp PROPERTIES GENERATED TRUE)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/bxx/operators.hpp PROPERTIES GENERATED TRUE)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/bxx/runtime.operations.hpp PROPERTIES GENERATED TRUE)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/bxx/runtime.typechecker.hpp PROPERTIES GENERATED TRUE)

# Output of codegen
set(CODEGEN_SRC "${CODEGEN_SRC};${CMAKE_CURRENT_SOURCE_DIR}/bxx/traits.hpp")
set(CODEGEN_SRC "${CODEGEN_SRC};${CMAKE_CURRENT_SOURCE_DIR}/bxx/operators.hpp")
set(CODEGEN_SRC "${CODEGEN_SRC};${CMAKE_CURRENT_SOURCE_DIR}/bxx/runtime.operations.hpp")
set(CODEGEN_SRC "${CODEGEN_SRC};${CMAKE_CURRENT_SOURCE_DIR}/bxx/runtime.typechecker.hpp")

# Files defining the codegen
set(CODEGEN_DEF "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen.py")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/element_types.json")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/operators.json")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/traits.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/traits.array.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/traits.const.tpl")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.footer.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.header.tpl")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.nops0.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.nops1.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.nops2.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.nops3.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.nops4.tpl")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.random.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.reduce.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.accumulate.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/runtime.typechecker.tpl")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.footer.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.header.tpl")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.nops2.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.nops2.bool.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.nops2.intern.tpl")

set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.nops3.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.nops3.bool.tpl")
set(CODEGEN_DEF "${CODEGEN_DEF};${CMAKE_CURRENT_SOURCE_DIR}/codegen/templates/sugar.nops3.intern.tpl")

add_custom_command(OUTPUT ${CODEGEN_SRC}
    COMMAND ${PYTHON_EXECUTABLE} ./codegen/gen.py
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/traits.hpp ${CMAKE_CURRENT_SOURCE_DIR}/bxx/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/operators.hpp ${CMAKE_CURRENT_SOURCE_DIR}/bxx/

    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/runtime.operations.hpp ${CMAKE_CURRENT_SOURCE_DIR}/bxx/
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/runtime.typechecker.hpp ${CMAKE_CURRENT_SOURCE_DIR}/bxx/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CODEGEN_DEF}
    COMMENT "BXX autogenerator"
    VERBATIM)

add_custom_target(BXX DEPENDS ${CODEGEN_SRC})
# Autogeneration - end
#

# Install all header files
install(DIRECTORY bxx DESTINATION include COMPONENT bohrium)

# Compile hello world as a test
add_subdirectory(examples)
