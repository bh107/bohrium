#cmake header file for all components

if(NOT DEFINED BH_INC_DIR)
    message(FATAL_ERROR "Please call cmake from the root of Bohrium")
endif()

#The global Bohrium include dir
include_directories(${BH_INC_DIR})




