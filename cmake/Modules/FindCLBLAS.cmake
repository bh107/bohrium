# - Find CLBLAS
# Find the CLBLAS includes and library
#
#  CLBLAS_INCLUDES    - where to find cblas.h/Accelerate.h/clblas.h
#  CLBLAS_LIBRARIES   - List of libraries when using CLBLAS.
#  CLBLAS_FOUND       - True if CLBLAS found.

include (FindPackageHandleStandardArgs)

find_path (CLBLAS_INCLUDES_PRE clBLAS.h)
find_library (CLBLAS_LIBRARIES_PRE NAMES clBLAS)

# handle the QUIETLY and REQUIRED arguments and set CLBLAS_FOUND to TRUE if
# all listed variables are TRUE

set (CLBLAS_LIBRARIES ${CLBLAS_LIBRARIES_PRE})
if (CLBLAS_INCLUDES_PRE)
  set (CLBLAS_INCLUDES ${CLBLAS_INCLUDES_PRE})
  set (CLBLAS_FOUND TRUE CACHE INTERNAL "")
endif ()

find_package_handle_standard_args (CLBLAS DEFAULT_MSG CLBLAS_LIBRARIES CLBLAS_INCLUDES)

mark_as_advanced (CLBLAS_LIBRARIES CLBLAS_INCLUDES)
