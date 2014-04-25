cmake_minimum_required(VERSION 2.6)

#External dependencies
find_package(OpenCL)
set_package_properties(OpenCL PROPERTIES DESCRIPTION "Open Computing Language" URL "www.khronos.org/opencl/")
set_package_properties(OpenCL PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the GPU-VE.")

if(OpenCL)
    include_directories(${OPENCL_INCLUDE_DIRS})
    include_directories(../../include)

    file(GLOB SRC *.cpp)

    add_library(bh_ve_gpu SHARED ${SRC})

    target_link_libraries(bh_ve_gpu bh ${OPENCL_LIBRARIES})
endif()
