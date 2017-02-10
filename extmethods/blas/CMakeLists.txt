cmake_minimum_required(VERSION 2.8)

set(EXT_BLAS true CACHE BOOL "EXT-BLAS: Build CBLAS extension method.")
if(NOT EXT_BLAS)
    return()
endif()

#External dependencies
find_package(CBLAS)
set_package_properties(CBLAS PROPERTIES DESCRIPTION "Basic Linear Algebra Subprograms" URL "www.netlib.org/blas")
set_package_properties(CBLAS PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the CBLAS extended method")

if(CBLAS_FOUND)
    set(EXTMETHOD_DIR ${CMAKE_BINARY_DIR}/extmethods/blas)
    file(MAKE_DIRECTORY ${EXTMETHOD_DIR})
    set(BLAS_LVL3 ${EXTMETHOD_DIR}/level3.cpp)
    set(BLAS_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/templates/)
    set(BLAS_PY   ${CMAKE_CURRENT_SOURCE_DIR}/../../core/codegen/gen_blas.py)
    add_custom_command(
      OUTPUT ${BLAS_LVL3}
      COMMAND ${BLAS_PY} ${BLAS_TEMPLATE_DIR} ${BLAS_LVL3}
      DEPENDS ${BLAS_PY}
    )

    include_directories(${CMAKE_SOURCE_DIR}/include)
    include_directories(${CMAKE_BINARY_DIR}/include)
    add_library(
      bh_blas SHARED
      ${CMAKE_CURRENT_BINARY_DIR}/level3.cpp
    )

    # We depend on bh.so
    target_link_libraries(bh_blas bh)

    include_directories(${CBLAS_INCLUDES})
    target_link_libraries(bh_blas ${CBLAS_LIBRARIES})

    install(TARGETS bh_blas DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

    set(OPENMP_LIBS ${OPENMP_LIBS} "${CMAKE_INSTALL_PREFIX}/lib/libbh_blas${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
else()
    if (CBLAS_INCLUDES_PRE)
        set_package_properties(BLAS PROPERTIES TYPE RECOMMENDED PURPOSE "CBLAS was found, but not enabled. \n   Manually enable BLAS with CBLAS_INCLUDES and CBLAS_LIBRARIES\n   if you have a clang multi-arch BLAS binary.")
    endif()
endif()
