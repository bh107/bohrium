cmake_minimum_required(VERSION 2.8)                     # cmake features

# Rules for how to generate bh_opcode.cpp and bh_opcode.h
set(INCLUDE_DIR     ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${INCLUDE_DIR})
set(OPCODE_H    ${INCLUDE_DIR}/bh_opcode.h)
set(OPCODE_CPP  ${CMAKE_CURRENT_BINARY_DIR}/bh_opcode.cpp)
set(OPCODE_JSON ${CMAKE_CURRENT_SOURCE_DIR}/codegen/opcodes.json)
set(OPCODE_PY   ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py)
add_custom_command(OUTPUT ${OPCODE_H} ${OPCODE_CPP}
    COMMAND ${PYTHON_EXECUTABLE} ${OPCODE_PY} ${OPCODE_JSON} ${OPCODE_H} ${OPCODE_CPP}
    DEPENDS ${OPCODE_JSON} ${OPCODE_PY})

include_directories(${CMAKE_SOURCE_DIR}/include ${INCLUDE_DIR})

file(GLOB SRC *.cpp jitk/*.cpp jitk/engines/*.cpp)
add_library(bh SHARED ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/bh_opcode.cpp)

target_link_libraries(bh ${CMAKE_DL_LIBS})      # bh_component depends on dlopen etc.
target_link_libraries(bh ${Boost_LIBRARIES})    # A shit ton of stuff depends on boost
target_link_libraries(bh ${LIBSIGSEGV_LIBRARY}) # bh_mem_signal depends on LibSigSegv

set(CORE_LINK_FLAGS "" CACHE STRING "Link flags to use when creating _bh.so (e.g. -static-libgcc -static-libstdc++)")
target_link_libraries(bh ${CORE_LINK_FLAGS})

install(TARGETS bh DESTINATION ${LIBDIR} COMPONENT bohrium)
install(DIRECTORY ${BOHRIUM_SOURCE_DIR}/include/ DESTINATION include/bohrium COMPONENT bohrium)
install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION include/bohrium COMPONENT bohrium)
