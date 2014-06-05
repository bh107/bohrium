cmake_minimum_required(VERSION 2.6)

#External dependencies
find_package(OpenCL)
set_package_properties(OpenCL PROPERTIES DESCRIPTION "Open Computing Language" URL "www.khronos.org/opencl/")
set_package_properties(OpenCL PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the GPU-VE.")

if(OPENCL_FOUND)
    include_directories(${OPENCL_INCLUDE_DIRS})
    include_directories(../../include)
#    add_definitions("-DBH_TIMING_SUM")
#    add_definitions("-DSTATIC_KERNEL")
#    add_definitions("-DDEBUG")
    file(GLOB SRC *.cpp *.c)

    add_library(bh_ve_gpu SHARED ${SRC})

    target_link_libraries(bh_ve_gpu bh ${OPENCL_LIBRARIES})
    install(TARGETS bh_ve_gpu DESTINATION lib)
    install(DIRECTORY include DESTINATION share/bohrium)
    install(DIRECTORY ocl_source DESTINATION share/bohrium)

endif()
