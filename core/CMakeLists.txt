cmake_minimum_required(VERSION 2.8)                     # cmake features

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC *.cpp)

add_library(bh SHARED ${SRC} bh_opcode.cpp)

# Rules for how to generate bh_opcode.cpp and bh_opcode.h
set(OPCODE_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/../include/bh_opcode.h
                ${CMAKE_CURRENT_SOURCE_DIR}/bh_opcode.cpp)
set(OPCODE_JSON ${CMAKE_CURRENT_SOURCE_DIR}/codegen/opcodes.json)
set(OPCODE_PY   ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py)
add_custom_command(OUTPUT ${OPCODE_SRC}
    COMMAND ${OPCODE_PY} ${OPCODE_JSON} ${OPCODE_SRC}
    DEPENDS ${OPCODE_JSON} ${OPCODE_PY})

add_subdirectory(iniparser)                     # bh_component depends on iniparser
target_link_libraries(bh iniparser)
target_link_libraries(bh ${CMAKE_DL_LIBS})      # bh_component depends on dlopen etc.
target_link_libraries(bh ${Boost_LIBRARIES})    # A shit ton of stuff depends on boost

install(TARGETS bh DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)
install(DIRECTORY ${BOHRIUM_SOURCE_DIR}/include/ DESTINATION include/bohrium COMPONENT bohrium)
