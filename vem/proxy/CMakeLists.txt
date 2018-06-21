cmake_minimum_required(VERSION 2.8)
set(VEM_PROXY false CACHE BOOL "VEM-PROXY: Build the proxy VEM.")
if(NOT VEM_PROXY)
    return()
endif()

find_package(ZLIB)
set_package_properties(ZLIB PROPERTIES DESCRIPTION "zlib general purpose compression library" URL "www.zlib.net")
set_package_properties(ZLIB PROPERTIES TYPE REQUIRED PURPOSE "zlib is required to build the Proxy-VEM")

if(NOT ZLIB_FOUND)
    message(FATAL_ERROR " zlib not found! Set VEM_PROXY=OFF or install zlib.")
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR " OpenCV not found! Set VEM_PROXY=OFF or install OpenCV 3.")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

include_directories(${ZLIB_INCLUDE_DIRS})

file(GLOB SRC *.cpp)

add_library(bh_vem_proxy SHARED ${SRC})

add_executable(bh_proxy_backend backend.cpp)

#We depend on bh.so
target_link_libraries(bh_vem_proxy bh ${ZLIB_LIBRARIES})
target_link_libraries(bh_proxy_backend bh_vem_proxy bh ${ZLIB_LIBRARIES})

install(TARGETS bh_vem_proxy DESTINATION ${LIBDIR} COMPONENT bohrium)
install(TARGETS bh_proxy_backend DESTINATION bin COMPONENT bohrium)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(bh_vem_proxy ${OpenCV_LIBS})
target_link_libraries(bh_proxy_backend ${OpenCV_LIBS})

