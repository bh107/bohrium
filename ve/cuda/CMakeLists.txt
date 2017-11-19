cmake_minimum_required(VERSION 2.8)

set(VE_CUDA true CACHE BOOL "VE-CUDA: Build the CUDA code generator.")
if(NOT VE_CUDA)
    return()
endif()

find_package(CUDA)
set_package_properties(CUDA PROPERTIES DESCRIPTION "Open Computing Language" URL "www.nvidia.com/object/cuda_home_new.html")
set_package_properties(CUDA PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the CUDA backend.")

if(NOT CUDA_FOUND)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Notice, CUDA_LIBRARIES only contains cudart but since we use the CUDA Driver API,
# so we also need to link against cuda by itself.
# We try to find it by searching relative to the libraries we already found.
list(GET CUDA_LIBRARIES 0 CUDA_FIRST_LIBRARY)
get_filename_component(CUDA_LIBDIR "${CUDA_FIRST_LIBRARY}" DIRECTORY)
find_library(CUDA_DRIVER_LIBRARY
    cuda
    HINTS ${CUDA_LIBDIR} ${CUDA_LIBDIR}/stubs
    DOC "Cuda runtime driver library"
)
if(NOT CUDA_DRIVER_LIBRARY)
    message(STATUS "Cuda runtime driver library not found")
    set(CUDA_FOUND false)
    return()
endif()

file(GLOB SRC *.cpp)
add_library(bh_ve_cuda SHARED ${SRC})

target_link_libraries(bh_ve_cuda bh ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
install(TARGETS bh_ve_cuda DESTINATION ${LIBDIR} COMPONENT bohrium-cuda)
