cmake_minimum_required(VERSION 2.8)

set(EXT_CLBLAS true CACHE BOOL "EXT-CLBLAS: Build CLBLAS extension method.")
if(NOT EXT_CLBLAS)
    return()
endif()

#External dependencies
find_package(CLBLAS)
set_package_properties(CLBLAS PROPERTIES DESCRIPTION "Basic Linear Algebra Subprograms implemented in OpenCL" URL "clmathlibraries.github.io/clBLAS/")
set_package_properties(CLBLAS PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the CLBLAS extended method")

if(CLBLAS_FOUND)
    set(EXTMETHOD_DIR ${CMAKE_BINARY_DIR}/extmethods/clblas)
    file(MAKE_DIRECTORY ${EXTMETHOD_DIR})
    set(CLBLAS_LVL3 ${EXTMETHOD_DIR}/clblas.cpp)
    set(CLBLAS_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/templates/)
    set(GEN_EXTMETHOD ${CMAKE_CURRENT_SOURCE_DIR}/../../core/codegen/gen_extmethod.py)
    add_custom_command(
      OUTPUT ${CLBLAS_LVL3}
      COMMAND ${GEN_EXTMETHOD} ${CLBLAS_TEMPLATE_DIR} ${CLBLAS_LVL3}
      DEPENDS ${GEN_EXTMETHOD}
    )

    include_directories(${CMAKE_SOURCE_DIR}/include)
    include_directories(${CMAKE_BINARY_DIR}/include)
    add_library(
      bh_clblas SHARED
      ${CMAKE_CURRENT_BINARY_DIR}/clblas.cpp
    )

    # We depend on bh.so
    target_link_libraries(bh_clblas bh)

    if(APPLE)
      target_link_libraries(bh_clblas ${OPENCL_LIBRARIES} clBLAS)
    endif()

    include_directories(${CLBLAS_INCLUDES})
    target_link_libraries(bh_clblas ${CLBLAS_LIBRARIES})

    install(TARGETS bh_clblas DESTINATION ${LIBDIR} COMPONENT bohrium)

    set(BH_OPENCL_LIBS ${BH_OPENCL_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_clblas${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
else()
    if (CLBLAS_INCLUDES_PRE)
        set_package_properties(CLBLAS PROPERTIES TYPE RECOMMENDED PURPOSE "CLBLAS was found, but not enabled. \n   Manually enable CLBLAS with CLBLAS_INCLUDES and CLBLAS_LIBRARIES.")
    endif()
endif()
