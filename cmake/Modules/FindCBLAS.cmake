# - Find cBLAS
# Find the cBLAS includes and library
#
#  CBLAS_INCLUDES    - where to find cblas.h/Accelerate.h
#  CBLAS_LIBRARIES   - List of libraries when using CBLAS.
#  CBLAS_FOUND       - True if CBLAS found.

include (FindPackageHandleStandardArgs)

if (APPLE)
  find_path (CBLAS_INCLUDES_PRE Accelerate.h)
  find_library (CBLAS_LIBRARIES_PRE NAMES Accelerate)
else ()
  find_path (CBLAS_INCLUDES_PRE cblas.h)
  find_library (CBLAS_LIBRARIES_PRE NAMES blas)
endif ()

# handle the QUIETLY and REQUIRED arguments and set CBLAS_FOUND to TRUE if
# all listed variables are TRUE

if (CBLAS_LIBRARIES_PRE)
  set (CBLAS_LIBRARIES ${CBLAS_LIBRARIES_PRE})
endif ()
if (CBLAS_INCLUDES_PRE)
  set (CBLAS_INCLUDES ${CBLAS_INCLUDES_PRE})
  set (CBLAS_FOUND TRUE CACHE INTERNAL "")
endif ()

find_package_handle_standard_args (CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDES)

mark_as_advanced (CBLAS_LIBRARIES CBLAS_INCLUDES)
