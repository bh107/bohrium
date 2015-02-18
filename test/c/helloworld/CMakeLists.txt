cmake_minimum_required(VERSION 2.8)
include_directories(../../../include)

file(GLOB SRC helloworld.c)

add_executable(bh_hello_world_c ${SRC})

#We depend on bh.so and bhc.so
target_link_libraries(bh_hello_world_c bh)
target_link_libraries(bh_hello_world_c bhc)

#We need to include the header file of the C Bridge
include_directories(../../../bridge/c/codegen/output)

install(TARGETS bh_hello_world_c DESTINATION share/bohrium/test/c COMPONENT bohrium)
install(FILES helloworld.c DESTINATION share/bohrium/test/c COMPONENT bohrium)
