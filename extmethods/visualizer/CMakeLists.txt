cmake_minimum_required(VERSION 2.8)
set(EXT_VISUALIZER true CACHE BOOL "EXT-VISUALIZER: Build VISUALIZER extension method.")
if(NOT EXT_VISUALIZER)
    return()
endif()

# Ubuntu packages: freeglut3 freeglut3-dev libxmu-dev libxi-dev

find_package(GLUT REQUIRED)
set_package_properties(GLUT PROPERTIES DESCRIPTION "The OpenGL Utility Toolkit" URL "www.opengl.org/resources/libraries/glut/")
set_package_properties(GLUT PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Visualizer if FreeGLUT is found.")

if(GLUT_FOUND)
    if(EXISTS "${GLUT_INCLUDE_DIR}/GL/freeglut.h")
        include_directories(${GLUT_INCLUDE_DIR})
        link_directories(${GLUT_LIBRARIES})
        add_definitions(${GLUT_DEFINITIONS})
    else()
        message(ERROR " GLUT found, but FreeGLUT is required!")
        unset(GLUT_FOUND)
        unset(GLUT_INCLUDE_DIR)
        unset(GLUT_LIBRARIES)
        unset(GLUT_DEFINITIONS)
    endif()
else()
    message(FATAL_ERROR "GLUT not found! Set EXT_VISUALIZER=OFF or install FreeGLUT.")
endif()

find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OPENGL not found! Set EXT_VISUALIZER=OFF or install OpenGL.")
endif()

if(GLUT_Xi_LIBRARY MATCHES "NOTFOUND")
    message(FATAL_ERROR "libxi not found! Set EXT_VISUALIZER=OFF or install libxi.")
endif()

if(GLUT_Xmu_LIBRARY MATCHES "NOTFOUND")
    message(FATAL_ERROR "libxmu not found! Set EXT_VISUALIZER=OFF or install libxmu.")
endif()

# Dependencies are met

if (GLUT_FOUND AND OPENGL_FOUND)
    include_directories(${CMAKE_SOURCE_DIR}/include)
    include_directories(${CMAKE_BINARY_DIR}/include)

    include_directories(${GLUT_INCLUDE_DIRS})
    link_directories(${GLUT_LIBRARY_DIRS})
    add_definitions(${GLUT_DEFINITIONS})

    include_directories(${OpenGL_INCLUDE_DIRS})
    link_directories(${OpenGL_LIBRARY_DIRS})
    add_definitions(${OpenGL_DEFINITIONS})

    file(GLOB SRC *.cpp)

    add_library(bh_visualizer SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bh_visualizer bh ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

    install(TARGETS bh_visualizer DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium-visualizer)
endif()
