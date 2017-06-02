cmake_minimum_required(VERSION 2.8)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_executable(bhxx_add "bhxx_add.cpp" )   # bhxx_add
target_link_libraries(bhxx_add bh)         # Depends on libbh.so ...
include_directories("..")                  # ... and header files of the C++ Bridge
include_directories("../../cpp/")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../)
install(TARGETS bhxx_add DESTINATION share/bohrium/test/cxx COMPONENT bohrium)
