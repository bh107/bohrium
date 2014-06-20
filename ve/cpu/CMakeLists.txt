cmake_minimum_required(VERSION 2.8)
include_directories(../../include)

#
#   Compiler flags
#

# CPU Profiling
set(CPU_PROFILING false CACHE BOOL "VE-CPU: Enabling CPU profiling.")
if (CPU_PROFILING)
    message(STATUS "VE-CPU: Enabling CPU profiling.")
    add_definitions( -DCPU_PROFILING )
endif()

# Debugging
set(CPU_DEBUGGING false CACHE BOOL "VE-CPU: Enable debugging.")
if (CPU_DEBUGGING)
    message(STATUS "VE-CPU: Enabling debugging.")
    add_definitions( -DDEBUGGING )
endif()

#External dependencies
find_package(CTemplate)
set_package_properties(CTemplate PROPERTIES DESCRIPTION "Template language for C++" URL "code.google.com/p/ctemplate/")
set_package_properties(CTemplate PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the CPU-VE.")

if(CTEMPLATE_FOUND)
    file(GLOB SRC *.cpp thirdparty/*.cpp)

    add_library(bh_ve_cpu SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bh_ve_cpu bh)

    install(TARGETS bh_ve_cpu DESTINATION lib COMPONENT bohrium)

    include_directories(${CTEMPLATE_INCLUDE_DIR})
    set(LIBS ${LIBS} ${CTEMPLATE_LIBRARIES})
    target_link_libraries(bh_ve_cpu ${LIBS})

    #Install all kernal templates
    install(DIRECTORY templates COMPONENT bohrium DESTINATION share/bohrium FILES_MATCHING PATTERN "*.tpl")

    install(FILES tac.h DESTINATION include/bohrium/ COMPONENT bohrium)

    #We need some empty dirs writable by all
    file(MAKE_DIRECTORY kernels)
    file(MAKE_DIRECTORY objects)
    file(MAKE_DIRECTORY scripts)
    install(DIRECTORY kernels DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
    install(DIRECTORY objects DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
    install(DIRECTORY scripts DESTINATION var/bohrium DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE COMPONENT bohrium)
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    target_link_libraries(progname ${Boost_LIBRARIES})                      
endif()

