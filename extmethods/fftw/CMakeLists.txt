cmake_minimum_required(VERSION 2.8)
set(EXT_FFTW true CACHE BOOL "EXT-FFTW: Build FFTW extension method.")
if(NOT EXT_FFTW)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

#External dependencies
find_package(FFTW)
set_package_properties(FFTW PROPERTIES DESCRIPTION "Fastest Fourier Transform in the West" URL "www.fftw.org")
set_package_properties(FFTW PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the FFTW extended method")

if(FFTW_FOUND)
    file(GLOB SRC main.cpp)

    add_library(bh_fftw SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bh_fftw bh)

    include_directories(${FFTW_INCLUDES})
    set(LIBS ${LIBS} ${FFTW_LIBRARIES})
    target_link_libraries(bh_fftw ${LIBS})

    install(TARGETS bh_fftw DESTINATION ${LIBDIR} COMPONENT bohrium)

    set(BH_OPENMP_LIBS ${BH_OPENMP_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_fftw${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
else()
    if (FFTW_INCLUDES_PRE)
        set_package_properties(FFTW PROPERTIES TYPE RECOMMENDED PURPOSE "FFTW was found, but not enabled. \n   Manually enable FFTW with FFTW_INCLUDES and FFTW_LIBRARIES\n   if you have a clang multi-arch FFTW binary.")
    endif()
endif()
