cmake_minimum_required(VERSION 2.6)

#External dependencies
find_package(PythonInterp)
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with NumPy).")

find_package(NumPy)
set_package_properties(NumPy PROPERTIES DESCRIPTION "Numerical" URL "www.numpy.org")
set_package_properties(NumPy PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python).")

set(PYTHON ${PYTHON_EXECUTABLE} CACHE STRING "The Python interpreter to use. Default is '${PYTHON_EXECUTABLE}'.")

message(STATUS "Python Interpreter: ${PYTHON}")

if(PYTHONINTERP_FOUND AND NUMPY_FOUND)
    add_custom_target(
        bhpy ALL
        COMMAND ${PYTHON} setup.py install --prefix=${CMAKE_CURRENT_BINARY_DIR} libpath=${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib DESTINATION .)
endif()

