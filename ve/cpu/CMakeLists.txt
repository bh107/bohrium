cmake_minimum_required(VERSION 2.8)
include(CheckCCompilerFlag)
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

file(GLOB SRC codegen/*.cpp *.cpp thirdparty/*.cpp)

add_library(bh_ve_cpu SHARED ${SRC})

#We depend on bh.so
target_link_libraries(bh_ve_cpu bh)

install(TARGETS bh_ve_cpu DESTINATION lib COMPONENT bohrium)

include_directories(${Boost_INCLUDE_DIRS})
include_directories("./codegen")
include_directories("./")
set(LIBS ${LIBS})
target_link_libraries(bh_ve_cpu ${LIBS})

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(bh_ve_cpu ${Boost_LIBRARIES})
endif()

#Install all kernal templates
install(DIRECTORY templates COMPONENT bohrium DESTINATION share/bohrium FILES_MATCHING PATTERN "*.tpl")

install(FILES tac.h DESTINATION include/bohrium/ COMPONENT bohrium)

#We need some empty dirs writable by all
file(MAKE_DIRECTORY kernels)
file(MAKE_DIRECTORY objects)
file(MAKE_DIRECTORY scripts)

install(DIRECTORY kernels DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
install(DIRECTORY objects DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)

#
#   Multi-core stuff
#
find_package(OpenMP COMPONENTS bohrium)

#
#   Compiler-flags for JIT-compilation
#
if (APPLE)
    install(
        FILES scripts/osx_compile.sh
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION var/bohrium/scripts
    )
    set(VE_CPU_COMPILER_CMD "${CMAKE_INSTALL_PREFIX}/var/bohrium/scripts/osx_compile.sh -I${CMAKE_INSTALL_PREFIX}/include -I${CMAKE_INSTALL_PREFIX}/include/bohrium -I${CMAKE_INSTALL_PREFIX}/share/bohrium/include " PARENT_SCOPE)
    set(VE_CPU_COMPILER_INC "" PARENT_SCOPE)
    set(VE_CPU_COMPILER_LIB "" PARENT_SCOPE)
    set(VE_CPU_COMPILER_EXT "" PARENT_SCOPE)
    set(VE_CPU_COMPILER_FLG "" PARENT_SCOPE)

else()

    set(VE_CPU_COMPILER_CMD "${CMAKE_C_COMPILER}" PARENT_SCOPE)
    set(VE_CPU_COMPILER_INC "-I${CMAKE_INSTALL_PREFIX}/include -I${CMAKE_INSTALL_PREFIX}/include/bohrium -I${CMAKE_INSTALL_PREFIX}/share/bohrium/include" PARENT_SCOPE)
    set(VE_CPU_COMPILER_LIB "-lm" PARENT_SCOPE)
    set(VE_CPU_COMPILER_EXT "-fPIC -shared -x c -std=c99" PARENT_SCOPE)

    #   VE_CPU_COMPILER_FLG is set below, checking for a bunch of stuff 
    check_c_compiler_flag(-O3 HAS_O3)
    if (HAS_O3)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -O3")
    endif()
    check_c_compiler_flag(-fstrict-aliasing HAS_STRICT_ALIASING)
    if (HAS_STRICT_ALIASING)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -fstrict-aliasing")
    endif()
    check_c_compiler_flag(-fopenmp-simd HAS_OPENMP_SIMD)
    if (HAS_OPENMP_SIMD)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -fopenmp-simd")
    endif()
    check_c_compiler_flag(-march=native HAS_MARCH_NATIVE)
    if (HAS_MARCH_NATIVE)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -march=native")
    endif()

    if(OPENMP_FOUND)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} ${OpenMP_C_FLAGS}")
    endif()
    set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG}" PARENT_SCOPE)

endif()
#
#   The above are used by config.ini.in
#

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    find_package(HWLOC)
    set_package_properties(HWLOC PROPERTIES DESCRIPTION "hardware locality (hwloc)" URL "http://www.open-mpi.org/projects/hwloc/")
    set_package_properties(HWLOC PROPERTIES TYPE RECOMMENDED PURPOSE "VE-CPU use it to control thread affinity.")

    if(HWLOC_FOUND)
        include_directories(${HWLOC_INCLUDE_DIR})
        target_link_libraries(bh_ve_cpu ${HWLOC_LIBRARY})
        add_definitions(-DVE_CPU_BIND)
    endif()
else()
    if(APPLE)
        if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
            message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            message("!!                                                    !!")
            message("!!                 !!!! NOTICE !!!!                   !!")
            message("!!                                                    !!")
            message("!!         Compiler does not support OpenMP           !!")
            message("!!                                                    !!")
            message("!!   You will only get single threaded performance!   !!")
            message("!!                                                    !!")
            message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        endif()
    else()
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        message("!!                                                    !!")
        message("!!       WARNING /// ERROR /// ALERT /// PANIC        !!")
        message("!!                                                    !!")
        message("!!       !! Compiler does not support OpenMP !!       !!")
        message("!!                                                    !!")
        message("!!     You should STOP, ABORT, CANCEL this build      !!")
        message("!!                                                    !!")
        message("!!                     and then                       !!")
        message("!!                                                    !!")
        message("!!    ---={[   INSTALL A DECENT COMPILER    ]}=---    !!")
        message("!!                                                    !!")
        message("!!   Change your operating system if you have to...   !!")
        message("!!                                                    !!")
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exec_program(sleep ARGS 60)
        exec_program(timeout ARGS 60)
    endif()
endif()
