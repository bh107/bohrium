# - Find the Python Module Cheetah
#
#   CTEMPLATE_FOUND       - True if cheetah is found.
#

find_program(CHEETAH_COMMAND cheetah)

# handle the QUIETLY and REQUIRED arguments and set CTEMPLATE_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE( "FindPackageHandleStandardArgs" )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( "Cheetah" DEFAULT_MSG CHEETAH_COMMAND )

MARK_AS_ADVANCED( CHEETAH_COMMAND )
