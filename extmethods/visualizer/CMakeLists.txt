cmake_minimum_required(VERSION 2.8)
set(EXT_VISUALIZER true CACHE BOOL "EXT-VISUALIZER: Build VISUALIZER extension method.")
if(NOT EXT_VISUALIZER)
    return()
endif()

# freeglut3 freeglut3-dev libxmu-dev libxi-dev

#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

include_directories(../../include)

file(GLOB SRC *.cpp)

add_library(bh_visualizer SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_visualizer bh ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

install(TARGETS bh_visualizer DESTINATION lib COMPONENT bohrium)
