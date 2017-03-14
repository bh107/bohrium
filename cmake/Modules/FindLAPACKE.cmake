# Find and link to LAPACKE.
#
# Variables defined::
#
#   LAPACKE_FOUND
#   LAPACKE_LIBRARIES
#   LAPACKE_INCLUDE_DIR

include(FindPackageHandleStandardArgs)

set(LAPACKE_FOUND FALSE)
set(LAPACKE_LIBRARIES "NOTFOUND")
set(LAPACKE_INCLUDE_DIR "NOTFOUND")

if(APPLE)
  find_library (LAPACKE_LIBRARIES NAMES Accelerate)
  find_path (LAPACKE_INCLUDE_DIR Accelerate.h)
else()
  find_library(LAPACKE_LIBRARIES NAMES lapacke)
  find_path(LAPACKE_INCLUDE_DIR lapacke.h)
endif()

find_package_handle_standard_args(Lapacke DEFAULT_MSG LAPACKE_LIBRARIES LAPACKE_INCLUDE_DIR)
