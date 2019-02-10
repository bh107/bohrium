cmake_minimum_required(VERSION 2.8)
set(BRIDGE_C ${BRIDGE_BHXX} CACHE BOOL "BRIDGE-C: Build the C-bridge.")
if(NOT BRIDGE_C)
    return()
endif()
if(NOT BRIDGE_BHXX)
    message(FATAL_ERROR "BRIDGE_BHXX is required for BRIDGE_C, so please set BRIDGE_BHXX to ON or BRIDGE_C to OFF")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/bridge/cxx/include)
include_directories(${CMAKE_BINARY_DIR}/bridge/cxx/include)

set(SRC_H ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.h
          ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.h
          ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials_enum_typed.h
          ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.h
          ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations_enum_typed.h)

set(SRC_CPP ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials_enum_typed.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations_enum_typed.cpp)


#Let's generate the source files of the C bridge
#Note that we concatenate all header files in to one 'bhc.h'
add_custom_command(
    OUTPUT ${SRC_H} ${SRC_CPP} ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h
    COMMAND ${PYTHON_EXECUTABLE} gen_types.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} gen_array_operations.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} gen_array_operations_enum_typed.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} gen_specials.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} gen_specials_enum_typed.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND cat ${SRC_H} > ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h
    COMMAND cat ${CMAKE_SOURCE_DIR}/include/bohrium/bh_mem_signal.h >> ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h
    DEPENDS gen_types.py
            gen_array_operations.py
            gen_array_operations_enum_typed.py
            gen_specials.py
            gen_specials_enum_typed.py
            ../../core/codegen/opcodes.json
            ../../core/codegen/types.json
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(bhc SHARED ${SRC_CPP})

# We depend on bhxx.so
target_link_libraries(bhc bhxx)

# Finally, let's install the compiled library and the combined header file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h DESTINATION include/bohrium COMPONENT bohrium)
install(TARGETS bhc DESTINATION ${LIBDIR} COMPONENT bohrium)
