# - Find JIT capabilities
#
# VE_CPU_COMPILER_CMD
# VE_CPU_COMPILER_INC
# VE_CPU_COMPILER_LIB
# VE_CPU_COMPILER_EXT
# VE_CPU_COMPILER_FLG
include(CheckCCompilerFlag)             

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

    # Optional flags 
    check_c_compiler_flag(-O3 HAS_O3)
    if (HAS_O3)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -O3")
    endif()
    check_c_compiler_flag(-march=native HAS_MARCH_NATIVE)
    if (HAS_MARCH_NATIVE)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -march=native")
    endif()
    check_c_compiler_flag("--param vect-max-version-for-alias-checks=100" HAS_PARAM_VECTMAX)
    if (HAS_PARAM_VECTMAX)
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} --param vect-max-version-for-alias-checks=100")
    endif()

    find_package(OpenMP)    # Parallism
    if (OPENMP_FOUND)
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
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
        
        # Check OMP-SIMD
        check_c_compiler_flag(-fopenmp-simd HAS_OPENMP_SIMD)
        if (HAS_OPENMP_SIMD)
            set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} -fopenmp-simd")
        endif()

        # Set OMP compiler-flag
        set(VE_CPU_COMPILER_FLG "${VE_CPU_COMPILER_FLG} ${OpenMP_C_FLAGS}")
    endif()
endif()

