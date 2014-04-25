cmake_minimum_required(VERSION 2.6)

find_package(NumPy)
set_package_properties(NumPy PROPERTIES DESCRIPTION "Numerical Python" URL "www.numpy.org")
set_package_properties(NumPy PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python and SWIG).")

find_package(SWIG)
set_package_properties(SWIG PROPERTIES DESCRIPTION "Simplified Wrapper and Interface Generator" URL "www.swig.org")
set_package_properties(SWIG PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python and NumPy).")

if(PYTHONINTERP_FOUND AND NUMPY_FOUND AND SWIG_FOUND)
    add_custom_target(
        bhpy ALL
        COMMAND ${PYTHON_EXECUTABLE} setup.py install --prefix=${CMAKE_CURRENT_BINARY_DIR} libpath=${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib DESTINATION .)
else()
    message(STATUS "The Python/NumPy bridge cannot be build")
endif()

