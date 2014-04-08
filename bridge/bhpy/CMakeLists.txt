cmake_minimum_required(VERSION 2.6)

install(CODE "execute_process(COMMAND python setup.py install --prefix=${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY bridge/bhpy)")

message(status "hej")
