# - Find BLAS
# Find the BLAS includes and library
#
#  BLAS_INCLUDES    - where to find cblas.h/Accelerate.h/clblas.h
#  BLAS_LIBRARIES   - List of libraries when using BLAS.
#  BLAS_FOUND       - True if BLAS found.

include (FindPackageHandleStandardArgs)

if (APPLE)
  find_path (BLAS_INCLUDES_PRE Accelerate.h)
  find_library (BLAS_LIBRARIES_PRE NAMES Accelerate)
else ()
  find_path (BLAS_INCLUDES_PRE cblas.h)
  find_library (BLAS_LIBRARIES_PRE NAMES cblas)
endif ()

# handle the QUIETLY and REQUIRED arguments and set BLAS_FOUND to TRUE if
# all listed variables are TRUE

set (BLAS_LIBRARIES ${BLAS_LIBRARIES_PRE})
if (BLAS_INCLUDES_PRE)
  set (BLAS_INCLUDES ${BLAS_INCLUDES_PRE})
  set (BLAS_FOUND TRUE CACHE INTERNAL "")
endif ()

find_package_handle_standard_args (BLAS DEFAULT_MSG BLAS_LIBRARIES BLAS_INCLUDES)

mark_as_advanced (BLAS_LIBRARIES BLAS_INCLUDES)
