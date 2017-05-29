# This file defines BH_VERSION_STRING, BH_VERSION_MAJOR, BH_VERSION_MINOR, and BH_VERSION_PATCH

find_package(Git)
set_package_properties(Git PROPERTIES DESCRIPTION "Distributed Version Control" URL "www.git-scm.com")
set_package_properties(Git PROPERTIES TYPE OPTIONAL PURPOSE "For extracting the currect version of the Bohrium source")

# Let's try to extract the version using git
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --match v[0-9]* --abbrev=0 --always
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE _VERSION_STRING
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "v" "" BH_VERSION_STRING ${_VERSION_STRING}) # removing the "v"
endif()

# If the extraction failed, read the VERSION file
if (BH_VERSION_STRING STREQUAL "")
    file(READ ${CMAKE_SOURCE_DIR}/VERSION BH_VERSION_STRING)
else()
    # Else, write to the version to file for later use
    file(WRITE ${CMAKE_SOURCE_DIR}/VERSION ${BH_VERSION_STRING})
endif()
message(STATUS "BH_VERSION_STRING: \"${BH_VERSION_STRING}\"")


# Then we extract the MAJOR, MINOR, and PATCH version
message("BH_VERSION_STRING: ${BH_VERSION_STRING}")
if("${BH_VERSION_STRING}" MATCHES "v([0-9]+)\\.([0-9]+)")
    set(${BH_VERSION_MAJOR} ${CMAKE_MATCH_1})
    set(${BH_VERSION_MINOR} ${CMAKE_MATCH_2})
    # The PATCH version might not exist
    if("${BH_VERSION_STRING}" MATCHES "v[0-9]+\\.[0-9]+\\.([0-9]+)")
        set(${BH_VERSION_PATCH} ${CMAKE_MATCH_1})
    else()
        set(${BH_VERSION_PATCH} 0)
    endif()
endif()

