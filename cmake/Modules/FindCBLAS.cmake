# - Find cBLAS
# Find the cBLAS includes and library
#
#  CBLAS_INCLUDES    - where to find cblas.h/Accelerate.h
#  CBLAS_LIBRARIES   - List of libraries when using CBLAS.
#  CBLAS_FOUND       - True if CBLAS found.

include(FindPackageHandleStandardArgs)
include(CheckLibraryExists)

function(FIND_AND_CHECK_CBLAS _libname _includename _symbol)
  find_path(_CBLAS_INCLUDES ${_includename})
  find_library(_CBLAS_LIBRARIES NAMES ${_libname})

  # check if cblas symbol is present
  if(_CBLAS_LIBRARIES AND _CBLAS_INCLUDES)
    get_filename_component(_LIB_PATH ${_CBLAS_LIBRARIES} DIRECTORY)
    check_library_exists(${_libname} ${_symbol} ${_LIB_PATH} _HAVE_CBLAS_SYMBOL)
    if(_HAVE_CBLAS_SYMBOL)
      set(CBLAS_LIBRARIES ${_CBLAS_LIBRARIES} CACHE FILEPATH "Path to CBLAS library")
      set(CBLAS_INCLUDES ${_CBLAS_INCLUDES} CACHE PATH "Path to CBLAS include directory")
    endif(_HAVE_CBLAS_SYMBOL)
  endif(_CBLAS_LIBRARIES AND _CBLAS_INCLUDES)

  # reset variables
  unset(_CBLAS_INCLUDES CACHE)
  unset(_CBLAS_LIBRARIES CACHE)
endfunction(FIND_AND_CHECK_CBLAS)


# check Apple CBLAS
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("Accelerate" "Accelerate.h" cblas_sgemm)
endif(NOT CBLAS_LIBRARIES)

# OpenBLAS
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("openblas" "cblas.h" cblas_sgemm)
endif(NOT CBLAS_LIBRARIES)

# generic cblas / ATLAS
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("cblas" "cblas.h" cblas_sgemm)
endif(NOT CBLAS_LIBRARIES)

# generic blas
if(NOT CBLAS_LIBRARIES)
  FIND_AND_CHECK_CBLAS("blas" "cblas.h" cblas_sgemm)
endif(NOT CBLAS_LIBRARIES)


if (CBLAS_LIBRARIES)
  set(CBLAS_FOUND TRUE CACHE INTERNAL "")
endif(CBLAS_LIBRARIES)


find_package_handle_standard_args (CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDES)
mark_as_advanced (CBLAS_LIBRARIES CBLAS_INCLUDES)
