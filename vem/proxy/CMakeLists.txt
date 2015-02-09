cmake_minimum_required(VERSION 2.8)
set(VEM_PROXY true CACHE BOOL "VEM-PROXY: Build the proxy VEM.")
if(NOT VEM_PROXY)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_vem_proxy SHARED ${SRC})

add_executable(bh_proxy_client bh_proxy_client.cpp)

#We depend on bh.so
target_link_libraries(bh_vem_proxy bh)
target_link_libraries(bh_proxy_client bh_vem_proxy)

install(TARGETS bh_vem_proxy DESTINATION lib COMPONENT bohrium)
install(TARGETS bh_proxy_client DESTINATION bin COMPONENT bohrium)

