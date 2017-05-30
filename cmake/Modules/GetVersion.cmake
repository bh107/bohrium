# This file defines BH_VERSION_STRING, BH_VERSION_MAJOR, BH_VERSION_MINOR, and BH_VERSION_PATCH
# And generate a version C header

# Read version information from the VERSION file
file(READ ${CMAKE_SOURCE_DIR}/VERSION BH_VERSION_STRING)

# Extract the MAJOR, MINOR, and PATCH version
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
    message(FATAL_ERROR "Extracting MAJOR, MINOR, and PATCH version failed. Please check the VERSION file." )
endif()

string(STRIP ${BH_VERSION_STRING} BH_VERSION_STRING)

message(STATUS "BH_VERSION_STRING: ${BH_VERSION_STRING}")
message(STATUS "BH_VERSION_MAJOR: ${BH_VERSION_MAJOR}")
message(STATUS "BH_VERSION_MINOR: ${BH_VERSION_MINOR}")
message(STATUS "BH_VERSION_PATCH: ${BH_VERSION_PATCH}")


# Write the version C header
configure_file(${CMAKE_SOURCE_DIR}/bh_version.h.in ${CMAKE_BINARY_DIR}/include/bh_version.h)