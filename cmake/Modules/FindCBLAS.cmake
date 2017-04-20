# - Find cBLAS
# Find the cBLAS includes and library
#
#  CBLAS_INCLUDES    - where to find cblas.h/Accelerate.h
#  CBLAS_LIBRARIES   - List of libraries when using CBLAS.
#  CBLAS_FOUND       - True if CBLAS found.

include(FindPackageHandleStandardArgs)
include(CheckLibraryExists)

function(FIND_AND_CHECK_CBLAS _libname _includename)
  find_path(_CBLAS_INCLUDES ${_includename})
  find_library(_CBLAS_LIBRARIES NAMES ${_libname})

  if(ARGV2) # check for optional argument
    set(_symbol ${ARGV2})
  endif(ARGV2)

  if(_CBLAS_LIBRARIES AND _CBLAS_INCLUDES)
    if(_symbol)
      # check if given symbol is present
      get_filename_component(_LIB_PATH ${_CBLAS_LIBRARIES} DIRECTORY)
      check_library_exists(${_libname} ${_symbol} ${_LIB_PATH} _HAVE_CBLAS_SYMBOL)
      unset(_LIB_PATH CACHE)
    else()
      set(_HAVE_CBLAS_SYMBOL True)
    endif()

    if(_HAVE_CBLAS_SYMBOL)
      set(CBLAS_LIBRARIES ${_CBLAS_LIBRARIES} CACHE FILEPATH "Path to CBLAS library")
      set(CBLAS_INCLUDES ${_CBLAS_INCLUDES} CACHE PATH "Path to CBLAS include directory")
    endif(_HAVE_CBLAS_SYMBOL)
  endif(_CBLAS_LIBRARIES AND _CBLAS_INCLUDES)

  # reset variables
  unset(_HAVE_CBLAS_SYMBOL CACHE)
  unset(_CBLAS_INCLUDES CACHE)
  unset(_CBLAS_LIBRARIES CACHE)
endfunction(FIND_AND_CHECK_CBLAS)


# check Apple CBLAS
if(APPLE)
  if(NOT CBLAS_LIBRARIES)
    FIND_AND_CHECK_CBLAS("Accelerate" "accelerate.h")
  endif(NOT CBLAS_LIBRARIES)
endif(APPLE)

# OpenBLAS
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("openblas" "cblas.h" "cblas_sgemm")
endif(NOT CBLAS_LIBRARIES)

# generic cblas / ATLAS
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("cblas" "cblas.h" "cblas_sgemm")
endif(NOT CBLAS_LIBRARIES)

# generic blas
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("blas" "cblas.h" "cblas_sgemm")
endif(NOT CBLAS_LIBRARIES)


if (CBLAS_LIBRARIES)
  set(CBLAS_FOUND TRUE CACHE INTERNAL "")
endif(CBLAS_LIBRARIES)


find_package_handle_standard_args (CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDES)
mark_as_advanced (CBLAS_LIBRARIES CBLAS_INCLUDES)
