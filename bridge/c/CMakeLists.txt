cmake_minimum_required(VERSION 2.8)
set(BRIDGE_C true CACHE BOOL "BRIDGE-C: Build the C-bridge.")
if(NOT BRIDGE_C)
    return()
endif()

include_directories(../../include)
include_directories(../cpp/bxx)

set(SRC_H ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.h
          ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.h
          ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.h)

set(SRC_CPP ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.cpp)


#Let's generate the source files of the C bridge
#Note that we concatenate all header files in to one 'bhc.h'
add_custom_command(
    OUTPUT ${SRC_H} ${SRC_CPP} ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h
    COMMAND ${PYTHON_EXECUTABLE} gen_types.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} gen_array_operations.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} gen_specials.py ${CMAKE_CURRENT_BINARY_DIR}/out
    COMMAND ${PYTHON_EXECUTABLE} cat_files.py ${SRC_H} --output ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h
    DEPENDS gen_types.py
            gen_array_operations.py
            gen_specials.py
            ../../core/codegen/opcodes.json
            ../cpp/codegen/element_types.json
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(bhc SHARED ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_types.cpp
                       ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_array_operations.cpp
                       ${CMAKE_CURRENT_BINARY_DIR}/out/bhc_specials.cpp
)

#We depend on bh.so
target_link_libraries(bhc bh)

#And we depend on bxx being built
add_dependencies(bhc BXX)


# Finally, let's install the compiled library and the combined header file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/out/bhc.h DESTINATION include/bohrium COMPONENT bohrium)
install(TARGETS bhc DESTINATION lib COMPONENT bohrium)
