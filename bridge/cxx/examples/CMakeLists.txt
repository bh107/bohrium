cmake_minimum_required(VERSION 2.8)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

include_directories("../include")          # ... and header files of the C++ Bridge
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../include)

add_executable(bhxx_add "bhxx_add.cpp" )
target_link_libraries(bhxx_add bhxx)
install(TARGETS bhxx_add DESTINATION share/bohrium/test/cxx COMPONENT bohrium)

add_executable(bhxx_add_reduce "bhxx_add_reduce.cpp" )
target_link_libraries(bhxx_add_reduce bhxx)
install(TARGETS bhxx_add_reduce DESTINATION share/bohrium/test/cxx COMPONENT bohrium)

add_executable(bhxx_indexing "bhxx_indexing.cpp" )
target_link_libraries(bhxx_indexing bhxx)
install(TARGETS bhxx_indexing DESTINATION share/bohrium/test/cxx COMPONENT bohrium)
