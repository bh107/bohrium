cmake_minimum_required(VERSION 2.8)
set(BRIDGE_BHXX true CACHE BOOL "BRIDGE-BHXX: Build the BHXX-bridge.")
if(NOT BRIDGE_BHXX)
    return()
endif()

set(SRC_H ${CMAKE_CURRENT_BINARY_DIR}/include/bhxx/array_operations.hpp)
set(SRC_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/array_operations.cpp)

add_custom_command(
    OUTPUT ${SRC_H} ${SRC_CPP}
    COMMAND ${PYTHON_EXECUTABLE} gen_array_operations.py ${CMAKE_CURRENT_BINARY_DIR}/include/bhxx ${CMAKE_CURRENT_BINARY_DIR}/src
    DEPENDS gen_array_operations.py
            ../../core/codegen/opcodes.json
            ../../core/codegen/types.json
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
#add_custom_target(BHXX_CODEGEN DEPENDS ${SRC_H})

# Include the headers of Bohrium
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

# Include the headers of bhxx
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Define the bhxx library
file(GLOB SRC src/*.cpp)
add_library(bhxx SHARED ${SRC} ${SRC_CPP})

# We depend on bh.so
target_link_libraries(bhxx bh)

# And we depend on the code generated files
#add_dependencies(bhxx BHXX_CODEGEN)

# Let's install header files (both static and generated headers)
install(DIRECTORY include/bhxx DESTINATION include COMPONENT bohrium)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/bhxx DESTINATION include COMPONENT bohrium)

# Let's install the library
install(TARGETS bhxx DESTINATION ${LIBDIR} COMPONENT bohrium)

add_subdirectory(examples)
