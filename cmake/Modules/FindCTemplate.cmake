# - Find CTEMPLATE
# Find the native CTEMPLATE includes and library
#
#   CTEMPLATE_FOUND       - True if CTEMPLATE found.
#   CTEMPLATE_INCLUDE_DIR - where to find CTEMPLATE.h, etc.
#   CTEMPLATE_LIBRARIES   - List of libraries when using CTEMPLATE.
#
# Source <https://github.com/UCL/hemelb/blob/master/dependencies/Modules/Find.cmake>

#We use our included library on MacOS
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    set(CTEMPLATE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/ctemplate/include/)
    set(CTEMPLATE_LIBRARIES ${PROJECT_SOURCE_DIR}/thirdparty/ctemplate/osx-lib/libctemplate.2.dylib)

ELSE()
    IF( CTEMPLATE_INCLUDE_DIR )
        # Already in cache, be silent
        SET( CTEMPLATE_FIND_QUIETLY TRUE )
    ENDIF( CTEMPLATE_INCLUDE_DIR )

    FIND_PATH( CTEMPLATE_INCLUDE_DIR "ctemplate/template.h")

    FIND_LIBRARY( CTEMPLATE_LIBRARIES
                  NAMES "ctemplate"
                  PATH_SUFFIXES "ctemplate" )
ENDIF()

# handle the QUIETLY and REQUIRED arguments and set CTEMPLATE_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( "FindPackageHandleStandardArgs" )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( "CTemplate" DEFAULT_MSG CTEMPLATE_INCLUDE_DIR CTEMPLATE_LIBRARIES )

MARK_AS_ADVANCED( CTEMPLATE_INCLUDE_DIR CTEMPLATE_LIBRARIES )
