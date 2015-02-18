cmake_minimum_required(VERSION 2.8)
set(BRIDGE_C true CACHE BOOL "BRIDGE-C: Build the C-bridge.")
if(NOT BRIDGE_C)
    return()
endif()

include_directories(../../include)
include_directories(../cpp/bxx)
include_directories(codegen/output/)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c.h
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c_type_definitions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c_implementation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c_implementation_basics.cpp)

    set(PY_SCRIPT ${PYTHON_EXECUTABLE} CACHE STRING "The Python interpreter to use for the build script and NOT the Python/NumPy Bridge")

if(PYTHONINTERP_FOUND)

    file(GLOB DEPS codegen/templates/*.ctpl codegen/*.json codegen/*.py)
    add_custom_command(
      OUTPUT ${SRC}
      COMMAND ${PY_SCRIPT} codegen/gen.py
      DEPENDS ${DEPS}  #TODO: gen.py writes a old timestamp thus this will always run
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_library(bhc SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bhc bh)

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bh_c.h DESTINATION include/bohrium COMPONENT bohrium)
    install(TARGETS bhc DESTINATION lib COMPONENT bohrium)

else()
    message(STATUS "The C-Bridge cannot be build!")
endif()
