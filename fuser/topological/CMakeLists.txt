cmake_minimum_required(VERSION 2.8)
set(FUSER_TOPOLOGICAL true CACHE BOOL "FUSER-TOPOLOGICAL: Build TOPOLOGICAL fuser.")
if(NOT FUSER_TOPOLOGICAL)
    return()
endif()

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_fuser_topological SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_fuser_topological bh)

install(TARGETS bh_fuser_topological DESTINATION lib COMPONENT bohrium)

file(MAKE_DIRECTORY fuse_cache)
install(DIRECTORY fuse_cache DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
