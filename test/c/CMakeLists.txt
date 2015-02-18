cmake_minimum_required(VERSION 2.8)
set(TEST_C true CACHE BOOL "TEST-C: C-Tests.")
if(NOT TEST_C)
    return()
endif()

#Add all tests
add_subdirectory(helloworld)
