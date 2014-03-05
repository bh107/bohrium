cmake_minimum_required(VERSION 2.6)

#Install all header files
install(DIRECTORY bh DESTINATION include)

#Compile hellow world as a test
add_subdirectory(examples)
