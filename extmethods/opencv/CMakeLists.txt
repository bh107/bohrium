cmake_minimum_required(VERSION 2.8)

set(EXT_OPENCV true CACHE BOOL "EXT-OPENCV: Build OPENCV extension method.")
if(NOT EXT_OPENCV)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

# External dependencies
find_package(OpenCV 3)
set_package_properties(OpenCV PROPERTIES DESCRIPTION "Open Source Computer Vision" URL "opencv.org")
set_package_properties(OpenCV PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the OpenCV extended method")

if(OpenCV_FOUND)
    file(GLOB SRC main.cpp)
    add_library(bh_opencv SHARED ${SRC})

    target_link_libraries(bh_opencv bh)

    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(bh_opencv ${OpenCV_LIBS})

    install(TARGETS bh_opencv DESTINATION ${LIBDIR} COMPONENT bohrium)

    set(OPENMP_LIBS ${OPENMP_LIBS} "${CMAKE_INSTALL_PREFIX}/lib/libbh_opencv${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
else()
    if (OpenCV_INCLUDES_PRE)
        set_package_properties(OPENCV PROPERTIES TYPE RECOMMENDED PURPOSE "OpenCV was found, but not enabled. \n   Manually enable OpenCV with OpenCV_INCLUDE_DIRS and OpenCV_LIBS.")
    endif()
endif()
