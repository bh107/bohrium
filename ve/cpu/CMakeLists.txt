cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

set(VE_CPU true CACHE BOOL "VE-CPU: Build CPU engine.")
if(NOT VE_CPU)
    return()
endif()

#
#   Compiler flags
#

set(VE_CPU_PROFILING false CACHE BOOL "VE-CPU: Enabling CPU profiling.")
if (VE_CPU_PROFILING)
    message(STATUS "VE-CPU: Enabling CPU profiling.")
    add_definitions( -DCPU_PROFILING )
endif()

set(VE_CPU_DEBUGGING false CACHE BOOL "VE-CPU: Enable debugging.")
if (VE_CPU_DEBUGGING)
    message(STATUS "VE-CPU: Enabling debugging.")
    add_definitions( -DDEBUGGING )
endif()

# External dependencies
find_package(CTemplate)
set_package_properties(CTemplate PROPERTIES DESCRIPTION "Template language for C++" URL "code.google.com/p/ctemplate/")
set_package_properties(CTemplate PROPERTIES TYPE REQUIRED PURPOSE "Used by CPU-VE for code generation.")

if(CTEMPLATE_FOUND)
    file(GLOB SRC codegen/*.cpp *.cpp thirdparty/*.cpp)

    add_library(bh_ve_cpu SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bh_ve_cpu bh)

    install(TARGETS bh_ve_cpu DESTINATION lib COMPONENT bohrium)

    include_directories(${CTEMPLATE_INCLUDE_DIR})
    include_directories(${Boost_INCLUDE_DIRS})
    include_directories("./codegen")
    include_directories("./")
    set(LIBS ${LIBS} ${CTEMPLATE_LIBRARIES})
    target_link_libraries(bh_ve_cpu ${LIBS})

    #Install all kernal templates
    install(DIRECTORY templates COMPONENT bohrium DESTINATION share/bohrium FILES_MATCHING PATTERN "*.tpl")

    install(FILES tac.h DESTINATION include/bohrium/ COMPONENT bohrium)

    #We need some empty dirs writable by all
    file(MAKE_DIRECTORY kernels)
    file(MAKE_DIRECTORY objects)
    file(MAKE_DIRECTORY scripts)

    install(DIRECTORY kernels DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
    install(DIRECTORY objects DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)

    if (APPLE)
        install(
            FILES scripts/osx_compile.sh
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION var/bohrium/scripts
        )
    endif()
    #install(DIRECTORY scripts DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(bh_ve_cpu ${Boost_LIBRARIES})
endif()

#
#   Multi-core stuff
#
find_package(OpenMP COMPONENTS bohrium)
#set_package_properties(OpenMP PROPERTIES DESCRIPTION "OpenMP parallel library and compiler directives" URL "www.openmp.org")
#set_package_properties(OpenMP PROPERTIES TYPE RECOMMENDED PURPOSE "VE-CPU use it for multi-core parallelization.")

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    find_package(HWLOC)
    set_package_properties(HWLOC PROPERTIES DESCRIPTION "hardware locality (hwloc)" URL "http://www.open-mpi.org/projects/hwloc/")
    set_package_properties(HWLOC PROPERTIES TYPE REQUIRED PURPOSE "VE-CPU use it to control thread affinity.")

    if(HWLOC_FOUND)
        include_directories(${HWLOC_INCLUDE_DIR})
        target_link_libraries(bh_ve_cpu ${HWLOC_LIBRARY})
    endif()

else()
    if (APPLE)
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        message("!!                                                    !!")
        message("!!                 !!!! NOTICE !!!!                   !!")
        message("!!                                                    !!")
        message("!!         Compiler does not support OpenMP           !!")
        message("!!                                                    !!")
        message("!!   You will only get single threaded performance!   !!")
        message("!!                                                    !!")
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else()
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        message("!!                                                    !!")
        message("!!      WARNING  /// ERROR /// ALERT /// PANIC        !!")
        message("!!                                                    !!")
        message("!!         Compiler does not support OpenMP           !!")
        message("!!                                                    !!")
        message("!!     You should STOP, ABORT, CANCEL this build      !!")
        message("!!                                                    !!")
        message("!!                     and then                       !!")
        message("!!                                                    !!")
        message("!!   ---={[   INSTALL A DESCENT COMPILER    ]}=---    !!")
        message("!!                                                    !!")
        message("!!   Change your operating system if you have to...   !!")
        message("!!                                                    !!")
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exec_program(sleep ARGS 60)
        exec_program(timeout ARGS 60)
    endif()
endif()
