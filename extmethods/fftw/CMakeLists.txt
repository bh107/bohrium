cmake_minimum_required(VERSION 2.8)
include_directories(../../include)


#External dependencies
find_package(FFTW)
set_package_properties(FFTW PROPERTIES DESCRIPTION "Fastest Fourier Transform in the West" URL "www.fftw.org")
set_package_properties(FFTW PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the FFTW extended method")

if(FFTW_FOUND)
    file(GLOB SRC *.cpp)

    add_library(bh_fftw SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bh_fftw bh)

    include_directories(${FFTW_INCLUDES})
    set(LIBS ${LIBS} ${FFTW_LIBRARIES})
    target_link_libraries(bh_fftw ${LIBS})

    install(TARGETS bh_fftw DESTINATION lib COMPONENT bohrium)
else()
    if (FFTW_INCLUDES_PRE)
        set_package_properties(FFTW PROPERTIES TYPE RECOMMENDED PURPOSE "FFTW was found, but not enabled. \n   Manually enable FFTW with FFTW_INCLUDES and FFTW_LIBRARIES\n   if you have a clang multi-arch FFTW binary.")
    endif()
endif()
