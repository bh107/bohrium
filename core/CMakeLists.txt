cmake_minimum_required(VERSION 2.6)

include_directories(../include)
file(GLOB SRC *.cpp bhir/*.cpp)

#The timing require C++11
if(USE_CXX98)
    file(GLOB EXCLUDE_SRC bh_timing.cpp)
    list(REMOVE_ITEM SRC ${EXCLUDE_SRC})
endif()

message(STATUS ${SRC})
add_library(bh SHARED ${SRC})

add_custom_command(TARGET bh PRE_BUILD COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_opcodes.py)

#We depend on iniparser
add_subdirectory(iniparser)
target_link_libraries(bh iniparser)

#We depend on dlopen etc.
target_link_libraries(bh ${CMAKE_DL_LIBS})

install(TARGETS bh DESTINATION lib)

