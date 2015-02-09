cmake_minimum_required(VERSION 2.8)
set(VE_GPU true CACHE BOOL "VE-GPU: Build GPU engine.")
if(NOT VE_GPU)
    return()
endif()

#External dependencies
find_package(OpenCL)
set_package_properties(OpenCL PROPERTIES DESCRIPTION "Open Computing Language" URL "www.khronos.org/opencl/")
set_package_properties(OpenCL PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the GPU-VE.")

find_package(OpenGL)
set_package_properties(OpenGL PROPERTIES DESCRIPTION "Open Graphics Library" URL "www.opengl.org")
set_package_properties(OpenGL PROPERTIES TYPE RECOMMENDED PURPOSE "Required by OpenCL/GPU-VE.")

if(OPENCL_FOUND AND OPENGL_FOUND)
#    add_definitions("-DBH_TIMING_SUM")
#    add_definitions("-DDEBUG")

    include_directories(../../include)

IF (APPLE)
    include_directories(${OPENCL_INCLUDE_DIRS}/Headers)
    add_definitions("-framework OpenCL")
ELSE (APPLE)
    include_directories(${OPENCL_INCLUDE_DIRS})
ENDIF (APPLE)

    file(GLOB SRC *.cpp *.c)

    add_library(bh_ve_gpu SHARED ${SRC})

    target_link_libraries(bh_ve_gpu bh ${OPENCL_LIBRARIES})
    install(TARGETS bh_ve_gpu DESTINATION lib COMPONENT bohrium-gpu)
    install(DIRECTORY include DESTINATION share/bohrium COMPONENT bohrium-gpu)
endif()
