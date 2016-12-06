#.rst:
# FindOpenACC
# ----------
#
# Finds OpenACC support
#
# This module can be used to detect OpenACC support in a compiler.  If
# the compiler supports OpenACC, the flags required to compile with
# OpenACC support are returned in variables for the different languages.
# The variables may be empty if the compiler does not need a special
# flag to support OpenACC.
#
# The following variables are set:
#
# ::
#
#    OpenACC_C_FLAGS - flags to add to the C compiler for OpenACC support
#    OPENACC_FOUND - true if openmp is detected
#
# Supported compilers can be found at
# http://www.openacc.org/content/tools
include(FindPackageHandleStandardArgs)
include(CheckCCompilerFlag)

set(_OPENACC_REQUIRED_VARS)
set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET ${OpenACC_FIND_QUIETLY})

function(_OPENACC_FLAG_CANDIDATES LANG)
  set(OpenACC_FLAG_CANDIDATES
    #GNU
    "-fopenacc"
    #Portland Group, PathScale
    "-acc -ta=tesla:nordc"
  )

  set(ACC_FLAG_GNU "-fopenacc")
  set(ACC_FLAG_PathScale "-acc ")
  set(ACC_FLAG_PGI "-acc -ta=tesla:nordc")

  # Move the flag that matches the compiler to the head of the list,
  # this is faster and doesn't clutter the output that much. If that
  # flag doesn't work we will still try all.
  if(ACC_FLAG_${CMAKE_${LANG}_COMPILER_ID})
    list(REMOVE_ITEM OpenACC_FLAG_CANDIDATES "${ACC_FLAG_${CMAKE_${LANG}_COMPILER_ID}}")
    list(INSERT OpenACC_FLAG_CANDIDATES 0 "${ACC_FLAG_${CMAKE_${LANG}_COMPILER_ID}}")
  endif()

  set(OpenACC_${LANG}_FLAG_CANDIDATES "${OpenACC_FLAG_CANDIDATES}" PARENT_SCOPE)
endfunction()

# sample openmp source code to test
set(OpenACC_C_TEST_SOURCE
"
#include <openacc.h>
int main() {
#ifdef _OPENACC
  return 0;
#else
  breaks_on_purpose
#endif
}
")

# check c compiler
if(CMAKE_C_COMPILER_LOADED)
  # if these are set then do not try to find them again,
  # by avoiding any try_compiles for the flags
  if(OpenACC_C_FLAGS)
    unset(OpenACC_C_FLAG_CANDIDATES)
  else()
    _OPENACC_FLAG_CANDIDATES("C")
  endif()

  foreach(FLAG IN LISTS OpenACC_C_FLAG_CANDIDATES)
    set(SAFE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${FLAG}")
    unset(OpenACC_FLAG_DETECTED CACHE)
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Try OpenACC C flag = [${FLAG}]")
    endif()
    check_c_source_compiles("${OpenACC_C_TEST_SOURCE}" OpenACC_FLAG_DETECTED)
    set(CMAKE_REQUIRED_FLAGS "${SAFE_CMAKE_REQUIRED_FLAGS}")
    if(OpenACC_FLAG_DETECTED)
      set(OpenACC_C_FLAGS_INTERNAL "${FLAG}")
      break()
    endif()
  endforeach()

  set(OpenACC_C_FLAGS "${OpenACC_C_FLAGS_INTERNAL}"
    CACHE STRING "C compiler flags for OpenACC parallization")

  list(APPEND _OPENACC_REQUIRED_VARS OpenACC_C_FLAGS)
  unset(OpenACC_C_FLAG_CANDIDATES)
endif()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})

if(_OPENACC_REQUIRED_VARS)
  find_package_handle_standard_args(OpenACC
                                    REQUIRED_VARS ${_OPENACC_REQUIRED_VARS})

  mark_as_advanced(${_OPENACC_REQUIRED_VARS})

  unset(_OPENACC_REQUIRED_VARS)
else()
  message(SEND_ERROR "FindOpenACC requires C language to be enabled")
endif()
