cmake_minimum_required(VERSION 2.6)

set(PYTHON "auto" CACHE STRING "The Python interpreter to use. 'auto' is default")

if(NOT (${PYTHON} STREQUAL "auto"))
    set(PYTHON_EXECUTABLE ${PYTHON})
endif()

#External dependencies
find_package(PythonInterp)
set_package_properties(PythonInterp PROPERTIES DESCRIPTION "Python Programming Language" URL "www.python.org")
set_package_properties(PythonInterp PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with NumPy).")

find_package(NumPy)
set_package_properties(NumPy PROPERTIES DESCRIPTION "Numerical" URL "www.numpy.org")
set_package_properties(NumPy PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python).")

if(PYTHONINTERP_FOUND AND NUMPY_FOUND)
    add_custom_target(
        bhpy ALL
        COMMAND ${PYTHON} setup.py install --prefix=${CMAKE_CURRENT_BINARY_DIR} libpath=${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib DESTINATION .)
endif()

