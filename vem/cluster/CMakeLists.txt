cmake_minimum_required(VERSION 2.8)
set(VEM_CLUSTER false CACHE BOOL "VEM-CLUSTER: Build the CLUSTER VEM.")
if(NOT VEM_CLUSTER)
    return()
endif()

#External dependencies
find_package(MPI)
set_package_properties(MPI PROPERTIES DESCRIPTION "Message Passing Interface" URL "www.mpi-forum.org")
set_package_properties(MPI PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Cluster-VEM.")

if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR " MPI not found! Set VEM_CLUSTER=OFF or install MPI.")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
file(GLOB SRC *.cpp)

include_directories(${MPI_CXX_INCLUDE_PATH})

add_library(bh_vem_cluster SHARED ${SRC})

add_executable(bh_vem_cluster_slave bh_vem_cluster_slave.cpp)

target_link_libraries(bh_vem_cluster ${MPI_CXX_LIBRARIES})
target_link_libraries(bh_vem_cluster bh)
target_link_libraries(bh_vem_cluster_slave ${MPI_CXX_LIBRARIES})
target_link_libraries(bh_vem_cluster_slave bh_vem_cluster)

install(TARGETS bh_vem_cluster DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT bohrium-cluster)
install(TARGETS bh_vem_cluster_slave DESTINATION bin COMPONENT bohrium-cluster)
