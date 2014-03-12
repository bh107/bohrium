cmake_minimum_required(VERSION 2.6)
include_directories(../../include)

file(GLOB SRC bh_ve_cpu.cpp)

add_library(bh_ve_cpu SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_ve_cpu bh)

install(TARGETS bh_ve_cpu DESTINATION lib)

#External dependencies
find_package(CTemplate REQUIRED)
include_directories(${CTEMPLATE_INCLUDE_DIR})
set(LIBS ${LIBS} ${CTEMPLATE_LIBRARIES})
target_link_libraries(bh_ve_cpu ${LIBS})

#Install all kernal templates
install(DIRECTORY templates DESTINATION share/bh FILES_MATCHING PATTERN "*.tpl")

#We need some empty dirs writable by all
file(MAKE_DIRECTORY kernels)
file(MAKE_DIRECTORY objects)
install(DIRECTORY kernels DESTINATION var/bh DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE)
install(DIRECTORY objects DESTINATION var/bh DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE)

