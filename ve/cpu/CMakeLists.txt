cmake_minimum_required(VERSION 2.8)                     # cmake features
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)             
include(ExternalProject)
include(FeatureSummary)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(BH_VE_CPU)

#
# Compile-time macro definitions
#
if (VE_CPU_PROFILING)
    message(STATUS "VE-CPU: Enabling CPU profiling.")
    add_definitions( -DVE_CPU_PROFILING )
endif()

if (VE_CPU_DEBUGGING)
    message(STATUS "VE-CPU: Enabling debugging.")
    add_definitions( -DVE_CPU_DEBUGGING )
endif()

check_c_compiler_flag(-pedantic HAS_C_PEDANTIC)         # C Flags
if (HAS_C_PEDANTIC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic ")
endif()
check_c_compiler_flag(-Wall HAS_C_WALL)
if (HAS_C_WALL)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall ")
endif()
check_c_compiler_flag(-std=c99 HAS_C_C99)
if (HAS_C_C99)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ")
endif()

check_cxx_compiler_flag(-Wall HAS_CXX_PEDANTIC)         # C++ Flags
if (HAS_CXX_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic ")
endif()
check_cxx_compiler_flag(-Wall HAS_CXX_WALL)
if (HAS_CXX_WALL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
endif()

check_cxx_compiler_flag(-std=c++0x HAS_CXX_CXX11)
if (HAS_CXX_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
endif()

file(GLOB SRC codegen/*.cpp *.cpp thirdparty/*.cpp)

link_directories(${BH_CORE_BINARY_DIR}) # BH_VE_CPU depends on BH_CORE
add_library(bh_ve_cpu SHARED ${SRC})
target_link_libraries(bh_ve_cpu bh)

install(TARGETS bh_ve_cpu DESTINATION lib COMPONENT bohrium)

include_directories(${BOHRIUM_SOURCE_DIR}/include)
include_directories("./codegen")
include_directories("./")

set(LIBS ${LIBS})
target_link_libraries(bh_ve_cpu ${LIBS})

# Install all kernal templates
install(DIRECTORY templates COMPONENT bohrium DESTINATION share/bohrium FILES_MATCHING PATTERN "*.tpl")

install(FILES tac.h DESTINATION include/bohrium/ COMPONENT bohrium)

# We need some empty dirs writable by all
file(MAKE_DIRECTORY kernels)
file(MAKE_DIRECTORY objects)
file(MAKE_DIRECTORY scripts)

install(DIRECTORY kernels DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
install(DIRECTORY objects DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)

#
#   Multi-core stuff
#
find_package(OpenMP COMPONENTS bohrium)
if(OPENMP_FOUND)
    # Look for HWLOC
    find_package(HWLOC)
    set_package_properties(HWLOC PROPERTIES DESCRIPTION "hardware locality (hwloc)" URL "http://www.open-mpi.org/projects/hwloc/")
    set_package_properties(HWLOC PROPERTIES TYPE RECOMMENDED PURPOSE "VE-CPU use it to control thread affinity.")

    # Tell CPU-engine that it can use HWLOC for thread-control
    if(HWLOC_FOUND)
        include_directories(${HWLOC_INCLUDE_DIR})
        target_link_libraries(bh_ve_cpu ${HWLOC_LIBRARY})
        add_definitions(-DVE_CPU_BIND)
    endif()

    # Check OMP-ATOMIC
    check_c_source_compiles( "
    #include <omp.h>
    int main(void){
        int acc = 0;
        int i;
        for(i=0; i<10; i++) {
            #pragma omp atomic
            acc = acc + 1;
        }
        return 0;
    }" OPENMP_ATOMIC_FOUND)

    # Tell CPU-engine that it can use the ATOMIC directive
    if(OPENMP_ATOMIC_FOUND)
        add_definitions(-DVE_CPU_OMP_ATOMIC)
    else()
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        message("!!                                                    !!")
        message("!!     Compiler does not support OpenMP ATOMIC        !!")
        message("!!                                                    !!")
        message("!!     Falling back to OpenMP Critical,               !!")
        message("!!     this will impact performance negatively.       !!")
        message("!!                                                    !!")
        message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    endif()

    # Assign OpenMP compile-flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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

#
#   Compiler-flags for JIT-compilation
find_package(JitFlags)

# Write the Bohrium configuration file
configure_file(${BOHRIUM_SOURCE_DIR}/config.ini.in config.ini)
string(REGEX MATCH "^$ENV{HOME}" ROOT_INSTALL ${CMAKE_INSTALL_PREFIX})
if ("${ROOT_INSTALL}" STREQUAL "")
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc/bohrium COMPONENT bohrium)
else()
    install(CODE "
    IF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
        CONFIGURE_FILE(${CMAKE_BINARY_DIR}/config.ini \"\$ENV{HOME}/.bohrium/config.ini\" COPYONLY)
    ENDIF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
    " COMPONENT bohrium)
endif()

