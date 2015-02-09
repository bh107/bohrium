cmake_minimum_required(VERSION 2.8)
set(BRIDGE_BXX true CACHE BOOL "BRIDGE-BXX: Build the BXX-bridge.")
if(NOT BRIDGE_BXX)
    return()
endif()

#Install all header files
install(DIRECTORY bxx DESTINATION include COMPONENT bohrium)

#Compile hellow world as a test
add_subdirectory(examples)
