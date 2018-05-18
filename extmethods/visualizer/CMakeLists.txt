cmake_minimum_required(VERSION 2.8)
set(EXT_VISUALIZER true CACHE BOOL "EXT-VISUALIZER: Build VISUALIZER extension method.")
if(NOT EXT_VISUALIZER)
    return()
endif()

find_package(GLUT)
set_package_properties(GLUT PROPERTIES DESCRIPTION "The OpenGL Utility Toolkit" URL "www.opengl.org/resources/libraries/glut/")
set_package_properties(GLUT PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Visualizer if FreeGLUT is found.")

if(GLUT_FOUND)
    if(EXISTS "${GLUT_INCLUDE_DIR}/GL/freeglut.h")
        include_directories(${GLUT_INCLUDE_DIR})
        link_directories(${GLUT_LIBRARIES})
        add_definitions(${GLUT_DEFINITIONS})
    else()
        set(VIZ_VIZ_ERR_MSG "${VIZ_VIZ_ERR_MSG}  * GLUT found, but FreeGLUT is required.\n")
    endif()
else()
    set(VIZ_ERR_MSG "${VIZ_ERR_MSG}  * FreeGLUT not found, on ubuntu install `freeglut3-dev`\n")
endif()

find_package(OpenGL)
if(NOT OPENGL_FOUND)
    set(VIZ_ERR_MSG "${VIZ_ERR_MSG}  * OpenGL not found, on ubuntu install `libgl1-mesa-dev`.\n")
endif()

if(GLUT_Xi_LIBRARY MATCHES "NOTFOUND")
    set(VIZ_ERR_MSG "${VIZ_ERR_MSG}  * libxi not found, on ubuntu install `libxi-dev`.\n")
endif()

if(GLUT_Xmu_LIBRARY MATCHES "NOTFOUND")
    set(VIZ_ERR_MSG "${VIZ_ERR_MSG}  * libxmu not found, on ubuntu install `libxmu-dev`.\n")
endif()

# Check if found all requirements
if(DEFINED VIZ_ERR_MSG)
    message(FATAL_ERROR "\nBohrium-Visualizer cannot be install, set `-DEXT_VISUALIZER=OFF` or install requirements:\n${VIZ_ERR_MSG}\n")
endif()

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

install(TARGETS bh_visualizer DESTINATION ${LIBDIR} COMPONENT bohrium-visualizer)

# Add visualizer to OpenMP and OpenCL libs
set(BH_OPENMP_LIBS ${BH_OPENMP_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_visualizer${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
set(BH_OPENCL_LIBS ${BH_OPENCL_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_visualizer${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
set(BH_CUDA_LIBS ${BH_CUDA_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_visualizer${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
set(BH_PROXY_LIBS ${BH_PROXY_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_visualizer${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
