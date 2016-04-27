cmake_minimum_required(VERSION 2.8)
set(FUSER_SINGLETON true CACHE BOOL "FUSER-SINGLETON: Build SINGLETON fuser.")
if(NOT FUSER_SINGLETON)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_fuser_singleton SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_fuser_singleton bh)

install(TARGETS bh_fuser_singleton DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

file(MAKE_DIRECTORY fuse_cache)
install(DIRECTORY fuse_cache DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
