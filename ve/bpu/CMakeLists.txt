cmake_minimum_required(VERSION 2.8)
set(VE_BPU true CACHE BOOL "VE-BPU: Bohrium Processing Unit - FPGA based VE.")
if(NOT VE_BPU)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp *.c)

add_library(bh_ve_bpu SHARED ${SRC})

target_link_libraries(bh_ve_bpu bh)
install(TARGETS bh_ve_bpu DESTINATION lib COMPONENT bohrium-bpu)
#install(DIRECTORY include DESTINATION share/bohrium COMPONENT bohrium-bpu)
