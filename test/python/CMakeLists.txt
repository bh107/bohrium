cmake_minimum_required(VERSION 2.8)

if(PYTHONINTERP_FOUND)

file(GLOB_RECURSE SRCS *.py)

install(FILES ${SRCS} DESTINATION share/bohrium/test/python COMPONENT bohrium)
install(DIRECTORY tests DESTINATION share/bohrium/test/python COMPONENT bohrium)

endif()
