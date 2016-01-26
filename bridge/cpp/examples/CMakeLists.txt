cmake_minimum_required(VERSION 2.8)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_executable(bxx_add "bxx_add.cc")    # bxx_add
target_link_libraries(bxx_add bh)       # Depends on libbh.so ...
include_directories(../)                # ... and header files of the C++ Bridge
install(TARGETS bxx_add DESTINATION share/bohrium/test/cpp COMPONENT bohrium)
add_dependencies(bxx_add BXX)

add_executable(bxx_sum "bxx_sum.cc")    # bxx_sum
target_link_libraries(bxx_sum bh)       # Depends on libbh.so ...
include_directories(../)                # ... and header files of the C++ Bridge
install(TARGETS bxx_sum DESTINATION share/bohrium/test/cpp COMPONENT bohrium)
add_dependencies(bxx_sum BXX)

add_executable(bxx_reduce "bxx_reduce.cc")  # bxx_reduce
target_link_libraries(bxx_reduce bh)        # Depends on libbh.so ...
include_directories(../)                    # ... and header files of the C++ Bridge
install(TARGETS bxx_reduce DESTINATION share/bohrium/test/cpp COMPONENT bohrium)
add_dependencies(bxx_reduce BXX)

add_executable(bxx_amax "bxx_amax.cc")      # bxx_amax
target_link_libraries(bxx_amax bh)          # Depends on libbh.so ...
include_directories(../)                    # ... and header files of the C++ Bridge
install(TARGETS bxx_amax DESTINATION share/bohrium/test/cpp COMPONENT bohrium)
add_dependencies(bxx_amax BXX)
