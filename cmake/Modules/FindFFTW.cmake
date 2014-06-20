# - Find FFTW
# Find the native FFTW includes and library
#
#  FFTW_INCLUDES    - where to find fftw3.h
#  FFTW_LIBRARIES   - List of libraries when using FFTW.
#  FFTW_FOUND       - True if FFTW found.

include (FindPackageHandleStandardArgs)

if (FFTW_INCLUDES)
  # Already in cache, be silent
  set (FFTW_FIND_QUIETLY TRUE)
endif (FFTW_INCLUDES)

find_path (FFTW_INCLUDES_PRE fftw3.h)
find_library (FFTW_LIBRARIES_PRE NAMES fftw3)

# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
if (APPLE AND FFTW_INCLUDES_PRE)
    message(STATUS "FFTW is found, but does not compile with Bohrium on OSX")
else ()
    set (FFTW_INCLUDES ${FFTW_INCLUDES_PRE})
    set (FFTW_LIBRARIES ${FFTW_LIBRARIES_PRE})
endif ()

find_package_handle_standard_args (FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDES)

mark_as_advanced (FFTW_LIBRARIES_PRE FFTW_INCLUDES_PRE)
