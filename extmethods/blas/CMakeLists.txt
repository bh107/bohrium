cmake_minimum_required(VERSION 2.8)
set(EXT_BLAS true CACHE BOOL "EXT-BLAS: Build BLAS extension method.")
if(NOT EXT_BLAS)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

#External dependencies
find_package(BLAS)
set_package_properties(BLAS PROPERTIES DESCRIPTION "Basic Linear Algebra Subprograms" URL "www.netlib.org/blas")
set_package_properties(BLAS PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the BLAS extended method")

if(BLAS_FOUND)
    file(GLOB SRC main.cpp)

    add_library(bh_blas SHARED ${SRC})

    # We depend on bh.so
    target_link_libraries(bh_blas bh)

    include_directories(${BLAS_INCLUDES})
    set(LIBS ${LIBS} ${BLAS_LIBRARIES})
    target_link_libraries(bh_blas ${LIBS})

    install(TARGETS bh_blas DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium)

    set(OPENMP_LIBS ${OPENMP_LIBS} "${CMAKE_INSTALL_PREFIX}/lib/libbh_blas${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
else()
    if (BLAS_INCLUDES_PRE)
        set_package_properties(BLAS PROPERTIES TYPE RECOMMENDED PURPOSE "BLAS was found, but not enabled. \n   Manually enable BLAS with BLAS_INCLUDES and BLAS_LIBRARIES\n   if you have a clang multi-arch BLAS binary.")
    endif()
endif()
