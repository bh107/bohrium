#.rst:
# FindIntelLEO
# ----------
#
# Finds IntelLEO support
#
# This module can be used to detect IntelLEO support in a compiler.  If
# the compiler supports IntelLEO, the flags required to compile with
# IntelLEO support are returned in variables for the different languages.
# The variables may be empty if the compiler does not need a special
# flag to support IntelLEO.
#
# The following variables are set:
#
# ::
#
#    IntelLEO_C_FLAGS - flags to add to the C compiler for IntelLEO support
#    INTEL_LEO_FOUND - true if openmp is detected
#
# Supported compilers can be found at
# http://www.openacc.org/content/tools
include(FindPackageHandleStandardArgs)
include(CheckCCompilerFlag)

set(_INTEL_LEO_REQUIRED_VARS)
set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET ${IntelLEO_FIND_QUIETLY})

# sample openmp source code to test
set(IntelLEO_C_TEST_SOURCE
"
#include <openacc.h>
int main() {
#ifdef _INTEL_LEO
  return 0;
#else
  breaks_on_purpose
#endif
}
")

# check c compiler
if(CMAKE_C_COMPILER_LOADED)

  foreach(FLAG IN LISTS IntelLEO_C_FLAG_CANDIDATES)
    set(SAFE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${FLAG}")
    unset(IntelLEO_FLAG_DETECTED CACHE)
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Try IntelLEO C flag = [${FLAG}]")
    endif()
    check_c_source_compiles("${IntelLEO_C_TEST_SOURCE}" IntelLEO_FLAG_DETECTED)
    set(CMAKE_REQUIRED_FLAGS "${SAFE_CMAKE_REQUIRED_FLAGS}")
    if(IntelLEO_FLAG_DETECTED)
      set(IntelLEO_C_FLAGS_INTERNAL "${FLAG}")
      break()
    endif()
  endforeach()

  set(IntelLEO_C_FLAGS "${IntelLEO_C_FLAGS_INTERNAL}"
    CACHE STRING "C compiler flags for IntelLEO parallization")

  list(APPEND _INTEL_LEO_REQUIRED_VARS IntelLEO_C_FLAGS)
  unset(IntelLEO_C_FLAG_CANDIDATES)
endif()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})

if(_INTEL_LEO_REQUIRED_VARS)
  find_package_handle_standard_args(IntelLEO
                                    REQUIRED_VARS ${_INTEL_LEO_REQUIRED_VARS})

  mark_as_advanced(${_INTEL_LEO_REQUIRED_VARS})

  unset(_INTEL_LEO_REQUIRED_VARS)
else()
  message(SEND_ERROR "FindIntelLEO requires C language to be enabled")
endif()
