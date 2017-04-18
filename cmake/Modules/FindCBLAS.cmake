# - Find CBLAS library
#
# This module finds an installed fortran library that implements the CBLAS 
# linear-algebra interface (see http://www.netlib.org/blas/), with CBLAS
# interface.
#
# This module sets the following variables:
#  CBLAS_FOUND - set to true if a library implementing the CBLAS interface
#    is found
#  CBLAS_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  CBLAS_LIBRARIES - uncached list of libraries (using full path name) to 
#    link against to use CBLAS
#  CBLAS_INCLUDE_DIR - path to includes
#  CBLAS_INCLUDE_FILE - the file to be included to use CBLAS
#

include(CheckFunctionExists)
include(CheckIncludeFile)
include(FindPackageHandleStandardArgs)

macro(CHECK_ALL_LIBRARIES LIBRARIES _prefix _name _flags _list _include _search_include)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the 
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.
  
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.

  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  message(STATUS "Checking for [${__list}]")
  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  set(_paths)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})

    # did we find all the libraries in the _list until now?
    # (we stop at the first unfound one)
    if(_libraries_work)      
      if(APPLE) 
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 ENV 
          DYLD_LIBRARY_PATH 
          )
      else(APPLE)
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 ENV 
          LD_LIBRARY_PATH 
          )
      endif(APPLE)
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      if(${_prefix}_${_library}_LIBRARY)
        get_filename_component(_path ${${_prefix}_${_library}_LIBRARY} PATH)
        list(APPEND _paths ${_path}/../include ${_path}/../../include)
      endif(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
    endif(_libraries_work)
  endforeach(_library ${_list})

  # Test include
  set(_bug_search_include ${_search_include}) #CMAKE BUG!!! SHOULD NOT BE THAT
  if(_bug_search_include)
    find_path(${_prefix}${_combined_name}_INCLUDE ${_include} ${_paths})
    mark_as_advanced(${_prefix}${_combined_name}_INCLUDE)
    if(${_prefix}${_combined_name}_INCLUDE)
      message(STATUS "Checking for [${__list}] -- includes found")
      set(${_prefix}_INCLUDE_DIR ${${_prefix}${_combined_name}_INCLUDE})
      set(${_prefix}_INCLUDE_FILE ${_include})
    else(${_prefix}${_combined_name}_INCLUDE)
      message(STATUS "Checking for [${__list}] -- includes not found")
      set(_libraries_work FALSE)
    endif(${_prefix}${_combined_name}_INCLUDE)
  else(_bug_search_include)
    set(${_prefix}_INCLUDE_DIR)
    set(${_prefix}_INCLUDE_FILE ${_include})
  endif(_bug_search_include)

  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
    CHECK_FUNCTION_EXISTS(${_name} ${_prefix}${_combined_name}_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    mark_as_advanced(${_prefix}${_combined_name}_WORKS)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})

    if(_libraries_work)
      message(STATUS "Checking for [${__list}] -- libraries found")
    endif(_libraries_work)

  endif(_libraries_work)
  

  if(NOT _libraries_work)
    set(${LIBRARIES} FALSE)
  endif(NOT _libraries_work)

ENDMACRO(CHECK_ALL_LIBRARIES)

set(CBLAS_LINKER_FLAGS)
set(CBLAS_LIBRARIES)

# CBLAS in intel mkl library? (shared)
if(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_sgemm
    ""
    "mkl;guide;pthread"
    "mkl_cblas.h"
    TRUE
    )
endif(NOT CBLAS_LIBRARIES)

#CBLAS in intel mkl library? (static, 32bit)
if(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_sgemm
    ""
    "mkl_ia32;guide;pthread"
    "mkl_cblas.h"
    TRUE
    )
endif(NOT CBLAS_LIBRARIES)

#CBLAS in intel mkl library? (static, em64t 64bit)
if(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_sgemm
    ""
    "mkl_em64t;guide;pthread"
    "mkl_cblas.h"
    ON
    )
endif(NOT CBLAS_LIBRARIES)

if(NOT CBLAS_LIBRARIES)
  # CBLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "cblas;f77blas;atlas"
    "cblas.h"
    TRUE
    )
endif(NOT CBLAS_LIBRARIES)

if(NOT CBLAS_LIBRARIES)
  # CBLAS in OpenBLAS library?
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "openblas"
    "cblas.h"
    TRUE
    )
endif(NOT CBLAS_LIBRARIES)

# Apple CBLAS library?
if(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "Accelerate"
    "Accelerate/Accelerate.h"
    FALSE
    )
endif(NOT CBLAS_LIBRARIES)

if(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "vecLib"
    "vecLib/vecLib.h"
    FALSE
    )
endif(NOT CBLAS_LIBRARIES)

find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDES)

mark_as_advanced(CBLAS_LIBRARIES CBLAS_INCLUDES)
