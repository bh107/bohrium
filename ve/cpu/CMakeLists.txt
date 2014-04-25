cmake_minimum_required(VERSION 2.6)
include_directories(../../include)

# Compile-flags
#add_definitions( -DPROFILING )

#External dependencies
find_package(CTemplate)
set_package_properties(CTemplate PROPERTIES DESCRIPTION "Template language for C++" URL "code.google.com/p/ctemplate/")
set_package_properties(CTemplate PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the CPU-VE.")

if(CTEMPLATE_FOUND)
    file(GLOB SRC *.cpp thirdparty/*.cpp)

    add_library(bh_ve_cpu SHARED ${SRC})

    #We depend on bh.so
    target_link_libraries(bh_ve_cpu bh)

    install(TARGETS bh_ve_cpu DESTINATION lib)

    include_directories(${CTEMPLATE_INCLUDE_DIR})
    set(LIBS ${LIBS} ${CTEMPLATE_LIBRARIES})
    target_link_libraries(bh_ve_cpu ${LIBS})

    #Install all kernal templates
    install(DIRECTORY templates DESTINATION share/bh FILES_MATCHING PATTERN "*.tpl")

    install(FILES tac.h DESTINATION include)

    #We need some empty dirs writable by all
    file(MAKE_DIRECTORY kernels)
    file(MAKE_DIRECTORY objects)
    file(MAKE_DIRECTORY scripts)
    install(DIRECTORY kernels DESTINATION var/bh DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY objects DESTINATION var/bh DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY scripts DESTINATION var/bh DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_WRITE GROUP_READ GROUP_EXECUTE WORLD_WRITE WORLD_READ WORLD_EXECUTE)
endif()
