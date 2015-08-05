cmake_minimum_required(VERSION 2.8)                     # cmake features
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)             
include(ExternalProject)
include(FeatureSummary)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(BH_VE_CPU)

include_directories(${BH_SOURCE_DIR}/include)           # Engine depends on BH headers
link_directories(${BH_BINARY_DIR}/core)                 # Engine depends on BH_CORE (libbh / bh)

include_directories(${CMAKE_SOURCE_DIR}/codegen)        # Engine depends on its own headers
include_directories(${CMAKE_SOURCE_DIR})
file(GLOB SRC codegen/*.cpp *.cpp thirdparty/*.cpp)     # Engine source
add_library(bh_ve_cpu SHARED ${SRC})                    # The main target for the Engine

target_link_libraries(bh_ve_cpu bh)                     # Link with BH_CORE

#
# Compile-time macro definitions
#
if (VE_CPU_PROFILING)                                   # Engine profiling MACRO
    message(STATUS "VE-CPU: Enabling CPU profiling.")
    add_definitions( -DVE_CPU_PROFILING )
endif()

if (VE_CPU_DEBUGGING)                                   # Engine debugging MACRO
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

check_cxx_compiler_flag(-std=c++11 HAS_CXX_CXX11)       # Sigh...
check_cxx_compiler_flag(-std=gnu++0x HAS_CXX_GNU0X)         
if (HAS_CXX_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
elseif(HAS_CXX_GNU0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x ")
endif()

find_package(OpenMP COMPONENTS bohrium)
set_package_properties(OpenMP PROPERTIES TYPE RECOMMENDED PURPOSE "Multicore processing, quite essential for performance")

if(OPENMP_FOUND)                                                    # Multi-core MACRO, flag, compiler-support and libraries for ENGINE

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")         # Enable compilation with OpenMp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  # Check OpenMP ATOMIC
    check_c_source_compiles( "
    #include <omp.h>
    int main(void){
        int acc = 0;
        int i;
        #pragma omp parallel for
        for(i=0; i<10; i++) {
            #pragma omp atomic
            acc = acc + 1;
        }
        return 0;
    }" OPENMP_ATOMIC_FOUND)

    if (OPENMP_ATOMIC_FOUND)                                        # Enable codegen "#omp atomic" for synchronization
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

    find_package(HWLOC)                                             # Check for HWLOC
    set_package_properties(HWLOC PROPERTIES DESCRIPTION "hardware locality (hwloc)" URL "http://www.open-mpi.org/projects/hwloc/")
    set_package_properties(HWLOC PROPERTIES TYPE RECOMMENDED PURPOSE "VE-CPU use it to control thread affinity.")
    
    if (HWLOC_FOUND)                                                # Enable HWLOC for threadcontrol
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

install(
    TARGETS bh_ve_cpu 
    DESTINATION lib COMPONENT bohrium)      # INSTALL the Engine itself
install(
    DIRECTORY templates
    COMPONENT bohrium
    DESTINATION share/bohrium
    FILES_MATCHING PATTERN "*.tpl")         # INSTALL codegen templates
install(
    FILES tac.h
    DESTINATION include/bohrium/
    COMPONENT bohrium)                      # INSTALL binary representation (three-address-code) header

file(MAKE_DIRECTORY kernels)                # INSTALL directory structure and set permissions
file(MAKE_DIRECTORY objects)
file(MAKE_DIRECTORY scripts)

install(DIRECTORY kernels DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
install(DIRECTORY objects DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)

#
# JIT-compiler capabilities: optimization, and parallelization
#
if (APPLE)
    install(
        FILES scripts/osx_compile.sh
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION var/bohrium/scripts
    )
    set(VE_CPU_COMPILER_CMD "${CMAKE_INSTALL_PREFIX}/var/bohrium/scripts/osx_compile.sh -I${CMAKE_INSTALL_PREFIX}/include -I${CMAKE_INSTALL_PREFIX}/include/bohrium -I${CMAKE_INSTALL_PREFIX}/share/bohrium/include ")
    set(VE_CPU_COMPILER_INC "")
    set(VE_CPU_COMPILER_INC "")
    set(VE_CPU_COMPILER_INC "")
    set(VE_CPU_COMPILER_LIB "")
    set(VE_CPU_COMPILER_EXT "")
else()

    # Mandatory flags
    set(VE_CPU_COMPILER_CMD "${CMAKE_C_COMPILER}")
    set(VE_CPU_COMPILER_INC " ${VE_CPU_COMPILER_INC} -I${CMAKE_INSTALL_PREFIX}/include ")
    set(VE_CPU_COMPILER_INC " ${VE_CPU_COMPILER_INC} -I${CMAKE_INSTALL_PREFIX}/include/bohrium ")
    set(VE_CPU_COMPILER_INC " ${VE_CPU_COMPILER_INC} -I${CMAKE_INSTALL_PREFIX}/share/bohrium/include ")
    set(VE_CPU_COMPILER_LIB "-lm")
    set(VE_CPU_COMPILER_EXT "-fPIC -shared -std=c99")
    
    check_c_compiler_flag(-O3 HAS_O3)                                       # Highly RECOMMENDED flags
    if (HAS_O3)                                                             # Optimization
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -O3")
    endif()
    check_c_compiler_flag(-march=native HAS_MARCH_NATIVE)                   
    if (HAS_MARCH_NATIVE)                                                   # Machine specific code
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -march=native")
    endif()
    check_c_compiler_flag("--param vect-max-version-for-alias-checks=100" HAS_PARAM_VECTMAX)
    if (HAS_PARAM_VECTMAX)                                                  # Upper bound on alias-checks
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} --param vect-max-version-for-alias-checks=100")
    endif()

    if (OPENMP_FOUND)        
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} ${OpenMP_C_FLAGS}") # OpenMP
        
        check_c_compiler_flag(-fopenmp-simd HAS_OPENMP_SIMD)
        if (HAS_OPENMP_SIMD)
            set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -fopenmp-simd") # OpenMP - SIMD
        endif()
    endif()
endif()

configure_file(${BH_SOURCE_DIR}/config.ini.in config.ini)               # Fill out the config-template

string(REGEX MATCH "^$ENV{HOME}" ROOT_INSTALL ${CMAKE_INSTALL_PREFIX})  # Install it
if ("${ROOT_INSTALL}" STREQUAL "")
    install(FILES ${CMAKE_BINARY_DIR}/config.ini DESTINATION etc/bohrium COMPONENT bohrium)
else()
    install(CODE "
    IF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
        CONFIGURE_FILE(${CMAKE_BINARY_DIR}/config.ini \"\$ENV{HOME}/.bohrium/config.ini\" COPYONLY)
    ENDIF(NOT EXISTS \"\$ENV{HOME}/.bohrium/config.ini\")
    " COMPONENT bohrium)
endif()

feature_summary(WHAT ALL)
