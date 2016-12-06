#.rst:
# FindIntelLEO
# ----------
#
# Finds IntelLEO support
#
# This module can be used to detect IntelLEO support in a compiler.  If
# the compiler supports IntelLEO then the following variables are set:
#
# ::
#
#    IntelLEO_FOUND - true if support for Intel LEO is detected
#
include(FindPackageHandleStandardArgs)
include(CheckCCompilerFlag)

set(_INTEL_LEO)

# check c compiler
if(CMAKE_C_COMPILER_LOADED)

    # Check that pragma is not ignored
    check_c_source_compiles( "
    #include <omp.h>
    int main(void) {
        int foo = 0;
        #pragma offload target(mic) in(does_not_exist)
        {
            foo += 1;
        }
        return foo;
    }" LEO_COMPILE_CHECK)

    if (${LEO_COMPILE_CHECK})
        set(_INTEL_LEO "0")
    else()
        # Check that pragma compiles
        check_c_source_compiles( "
        #include <stdlib.h>
        #include <omp.h>
        int main(void) {
            int nelem = 10;
            double* foo = (double*)malloc(nelem*sizeof(double));
            #pragma offload target(mic) in(foo:length(nelem) alloc_if(1) free_if(1))
            {
                foo += 1;
            }
            free(foo);
            return 0;
        }" LEO_COMPILE_CHECK_USAGE)

        if (${LEO_COMPILE_CHECK_USAGE})
            set(_INTEL_LEO "1")
        else()
            set(_INTEL_LEO "0")
        endif()
    endif()

else()
    message(SEND_ERROR "FindIntelLEO requires C language to be enabled")
endif()

find_package_handle_standard_args(IntelLEO "Language Extensions for Offload (LEO)" _INTEL_LEO)
