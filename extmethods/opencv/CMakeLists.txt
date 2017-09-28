cmake_minimum_required(VERSION 2.8)

set(EXT_OPENCV true CACHE BOOL "EXT-OPENCV: Build OPENCV extension method.")
if(NOT EXT_OPENCV)
    return()
endif()

# External dependencies
find_package(OpenCV 3 QUIET)
set_package_properties(OpenCV PROPERTIES DESCRIPTION "Open Source Computer Vision" URL "opencv.org")
set_package_properties(OpenCV PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the OpenCV extended method")

if(OpenCV_FOUND)
    set(EXTMETHOD_DIR ${CMAKE_BINARY_DIR}/extmethods/opencv)
    file(MAKE_DIRECTORY ${EXTMETHOD_DIR})

    include_directories(${CMAKE_SOURCE_DIR}/include)
    include_directories(${CMAKE_BINARY_DIR}/include)
    include_directories(${OpenCV_INCLUDE_DIRS})

    add_library(bh_opencv_filtering SHARED filtering.cpp)

    target_link_libraries(bh_opencv_filtering bh)
    target_link_libraries(bh_opencv_filtering ${OpenCV_LIBS})

    install(TARGETS bh_opencv_filtering DESTINATION ${LIBDIR} COMPONENT bohrium)

    set(OpenCV_TEMPLATE_LIBS ${OpenCV_TEMPLATE_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_opencv_filtering${CMAKE_SHARED_LIBRARY_SUFFIX}")

    set(GEN_EXTMETHOD ${CMAKE_CURRENT_SOURCE_DIR}/../../core/codegen/gen_extmethod.py)
    foreach(FUNC threshold)
      set(OpenCV_TEMPLATE_CPP ${EXTMETHOD_DIR}/${FUNC}.cpp)
      set(OpenCV_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/templates/${FUNC})

      add_custom_command(
        OUTPUT ${OpenCV_TEMPLATE_CPP}
        COMMAND ${GEN_EXTMETHOD} ${OpenCV_TEMPLATE_DIR} ${OpenCV_TEMPLATE_CPP}
      )

      add_library(
        bh_opencv_${FUNC} SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/${FUNC}.cpp
      )

      target_link_libraries(bh_opencv_${FUNC} bh)
      target_link_libraries(bh_opencv_${FUNC} ${OpenCV_LIBS})
      install(TARGETS bh_opencv_${FUNC} DESTINATION ${LIBDIR} COMPONENT bohrium)
      set(OpenCV_TEMPLATE_LIBS ${OpenCV_TEMPLATE_LIBS} "${CMAKE_INSTALL_PREFIX}/${LIBDIR}/libbh_opencv_${FUNC}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endforeach(FUNC)

    set(BH_OPENMP_LIBS ${BH_OPENMP_LIBS} ${OpenCV_TEMPLATE_LIBS} PARENT_SCOPE)
else()
    if (OpenCV_INCLUDES_PRE)
        set_package_properties(OPENCV PROPERTIES TYPE RECOMMENDED PURPOSE "OpenCV was found, but not enabled. \n   Manually enable OpenCV with OpenCV_INCLUDE_DIRS and OpenCV_LIBS.")
    endif()
endif()
