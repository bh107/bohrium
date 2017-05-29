# This file defines BH_VERSION_STRING, BH_VERSION_MAJOR, BH_VERSION_MINOR, and BH_VERSION_PATCH

find_package(Git)
set_package_properties(Git PROPERTIES DESCRIPTION "Distributed Version Control" URL "www.git-scm.com")
set_package_properties(Git PROPERTIES TYPE OPTIONAL PURPOSE "For extracting the currect version of the Bohrium source")

# Let's try to extract the version using git
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --match v[0-9]* --abbrev=0 --always
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE _VERSION_STRING
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if("${_VERSION_STRING}" MATCHES "v[0-9]+")
        string(REPLACE "v" "" BH_VERSION_STRING ${_VERSION_STRING}) # removing the "v"
    endif()
endif()

if (DEFINED BH_VERSION_STRING)
    # On success, write to the version to file for later use
    file(WRITE ${CMAKE_SOURCE_DIR}/VERSION ${BH_VERSION_STRING})
else()
    # The extraction failed, read the VERSION file
    file(READ ${CMAKE_SOURCE_DIR}/VERSION BH_VERSION_STRING)
endif()

# Then we extract the MAJOR, MINOR, and PATCH version
if("${BH_VERSION_STRING}" MATCHES "([0-9]+)\\.([0-9]+)")
    set(BH_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(BH_VERSION_MINOR ${CMAKE_MATCH_2})
    # The PATCH version might not exist
    if("${BH_VERSION_STRING}" MATCHES "[0-9]+\\.[0-9]+\\.([0-9]+)")
        set(BH_VERSION_PATCH ${CMAKE_MATCH_1})
    else()
        set(BH_VERSION_PATCH 0)
    endif()
else()
    message(FATAL_ERROR "Extracting MAJOR, MINOR, and PATCH version failed." )
endif()

message(STATUS "BH_VERSION_STRING: ${BH_VERSION_STRING}")
message(STATUS "BH_VERSION_MAJOR: ${BH_VERSION_MAJOR}")
message(STATUS "BH_VERSION_MINOR: ${BH_VERSION_MINOR}")
message(STATUS "BH_VERSION_PATCH: ${BH_VERSION_PATCH}")
