cmake_minimum_required(VERSION 2.8)

if(PYTHONINTERP_FOUND)

file(GLOB_RECURSE SRCS *.py)

install(FILES ${SRCS} DESTINATION share/bohrium/benchmark/python COMPONENT bohrium)
install(DIRECTORY datasets DESTINATION share/bohrium/benchmark/python COMPONENT bohrium)

endif()