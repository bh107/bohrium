cmake_minimum_required(VERSION 2.8)

set(EXT_LAPACK true CACHE BOOL "EXT-LAPACK: Build LAPACK extension method.")
if(NOT EXT_LAPACK)
    return()
endif()

#External dependencies
find_package(LAPACKE)
find_package(CBLAS)
set_package_properties(LAPACKE PROPERTIES DESCRIPTION "Linear Algebra PACKage" URL "www.netlib.org/lapack/lapacke.html")
set_package_properties(LAPACKE PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the LAPACK extended method")

find_path(CLAPACK_INCLUDE_DIR NAMES clapack.h PATHS /usr/include/atlas)
if(CLAPACK_INCLUDE_DIR)
  include_directories(${CLAPACK_INCLUDE_DIR})
endif()

if(LAPACKE_FOUND AND CBLAS_FOUND)
    set(EXTMETHOD_DIR ${CMAKE_BINARY_DIR}/extmethods/lapack)
    file(MAKE_DIRECTORY ${EXTMETHOD_DIR})

    include_directories(${CMAKE_SOURCE_DIR}/include)
    include_directories(${CMAKE_BINARY_DIR}/include)
    include_directories(${LAPACKE_INCLUDE_DIR})
    include_directories(${CBLAS_INCLUDES})

    set(GEN_EXTMETHOD ${CMAKE_CURRENT_SOURCE_DIR}/../../core/codegen/gen_extmethod.py)

    set(LAPACK_DRIVER_LIBS "")
    foreach(DRIVER le lsp sesvp gesvp ele elsp egsep egnep)
        set(LAPACK_DRIVER_CPP ${EXTMETHOD_DIR}/${DRIVER}.cpp)
        set(LAPACK_DRIVER_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/drivers/${DRIVER}/)

        add_custom_command(
          OUTPUT ${LAPACK_DRIVER_CPP}
          COMMAND ${PYTHON_EXECUTABLE} ${GEN_EXTMETHOD} ${EXT_LAPACK_CMD} ${LAPACK_DRIVER_TEMPLATE_DIR} ${LAPACK_DRIVER_CPP}
          DEPENDS ${CLANG_FORMAT}
        )

        add_library(
          bh_lapack_${DRIVER} SHARED
          ${CMAKE_CURRENT_BINARY_DIR}/${DRIVER}.cpp
        )

        # We depend on bh.so
        target_link_libraries(bh_lapack_${DRIVER} bh)

        if(APPLE)
          target_link_libraries(bh_lapack_${DRIVER} "-framework Accelerate")
        else()
          target_link_libraries(bh_lapack_${DRIVER} ${LAPACKE_LIBRARIES})
          target_link_libraries(bh_lapack_${DRIVER} ${CBLAS_LIBRARIES})
        endif()

        install(TARGETS bh_lapack_${DRIVER} DESTINATION ${LIBDIR} COMPONENT bohrium)

        set(LAPACK_DRIVER_LIBS ${LAPACK_DRIVER_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_lapack_${DRIVER}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endforeach(DRIVER)

    set(BH_OPENMP_LIBS ${BH_OPENMP_LIBS} ${LAPACK_DRIVER_LIBS} PARENT_SCOPE)
endif()
