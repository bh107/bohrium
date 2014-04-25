cmake_minimum_required(VERSION 2.6)
include_directories(../../include)

#External dependencies
find_package(MPI)
set_package_properties(MPI PROPERTIES DESCRIPTION "Message Passing Interface" URL "www.mpi-forum.org")
set_package_properties(MPI PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Cluster-VEM.")

if(MPI_C_FOUND)
    include(CMakeForceCompiler)
    CMAKE_FORCE_C_COMPILER(mpicc ${CMAKE_C_COMPILER_ID})
    CMAKE_FORCE_CXX_COMPILER(mpicxx ${CMAKE_CXX_COMPILER_ID})

    file(GLOB SRC *.cpp)

    add_library(bh_vem_cluster SHARED ${SRC})

    add_executable(bh_vem_cluster_slave bh_vem_cluster_slave.cpp)

    #We depend on bh.so
    target_link_libraries(bh_vem_cluster bh)
    target_link_libraries(bh_vem_cluster_slave bh_vem_cluster)

    install(TARGETS bh_vem_cluster DESTINATION lib)
    install(TARGETS bh_vem_cluster_slave DESTINATION bin)
endif()
