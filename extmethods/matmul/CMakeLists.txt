cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_matmul SHARED ${SRC})

find_package(OpenMP COMPONENTS bohrium)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#We depend on bh.so
target_link_libraries(bh_matmul bh)

install(TARGETS bh_matmul DESTINATION lib COMPONENT bohrium)
