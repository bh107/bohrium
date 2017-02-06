cmake_minimum_required(VERSION 2.8)
set(EXT_CLBLAS true CACHE BOOL "EXT-CLBLAS: Build CLBLAS extension method.")
if(NOT EXT_CLBLAS)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

#External dependencies
find_package(CLBLAS)
set_package_properties(CLBLAS PROPERTIES DESCRIPTION "Basic Linear Algebra Subprograms implemented in OpenCL" URL "clmathlibraries.github.io/clBLAS/")
set_package_properties(CLBLAS PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the CLBLAS extended method")

if(CLBLAS_FOUND)
    file(GLOB SRC main.cpp)

    add_library(bh_clblas SHARED ${SRC})

    # We depend on bh.so
    target_link_libraries(bh_clblas bh)

    if(APPLE)
      target_link_libraries(bh_clblas "-lclblas -framework OpenCL")
    endif()

    include_directories(${CLBLAS_INCLUDES})
    set(LIBS ${LIBS} ${CLBLAS_LIBRARIES})
    target_link_libraries(bh_clblas ${LIBS})

    install(TARGETS bh_clblas DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

    set(OPENCL_LIBS "${CMAKE_INSTALL_PREFIX}/lib/libbh_clblas${CMAKE_SHARED_LIBRARY_SUFFIX}" ${OPENCL_LIBS} PARENT_SCOPE)
else()
    if (CLBLAS_INCLUDES_PRE)
        set_package_properties(CLBLAS PROPERTIES TYPE RECOMMENDED PURPOSE "CLBLAS was found, but not enabled. \n   Manually enable CLBLAS with CLBLAS_INCLUDES and CLBLAS_LIBRARIES\n   if you have a clang multi-arch CLBLAS binary.")
    endif()
endif()
