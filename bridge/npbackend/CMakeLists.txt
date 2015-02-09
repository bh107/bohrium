cmake_minimum_required(VERSION 2.8)
set(BRIDGE_NPBACKEND true CACHE BOOL "BRIDGE-NPBACKEND: Build the NPBACKEND-bridge.")
if(NOT BRIDGE_NPBACKEND)
    return()
endif()

find_package(NumPy)
set_package_properties(NumPy PROPERTIES DESCRIPTION "Numerical Python" URL "www.numpy.org")
set_package_properties(NumPy PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python and SWIG).")

find_package(SWIG)
set_package_properties(SWIG PROPERTIES DESCRIPTION "Simplified Wrapper and Interface Generator" URL "www.swig.org")
set_package_properties(SWIG PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python and NumPy).")

find_package(Cython)
set_package_properties(Cython PROPERTIES DESCRIPTION "C-Extensions for Python" URL "cython.org")
set_package_properties(Cython PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Python/NumPy bridge (together with Python and SWIG).")

if(PYTHONINTERP_FOUND AND NUMPY_FOUND AND SWIG_FOUND AND CYTHON_FOUND AND NUMPY_INCLUDE_DIRS)

    # Generate distutils package and install it
    add_custom_target(
        bhpy ALL
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py build_ext --include-dirs=${NUMPY_INCLUDE_DIRS} buildpath=${CMAKE_CURRENT_BINARY_DIR} && ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install --prefix=${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib DESTINATION . COMPONENT bohrium)
else()
    if (NOT NUMPY_INCLUDE_DIRS)
        message(STATUS "The Python/NumPy bridge cannot be built, because the numpy headers are missing")
    else()
        message(STATUS "The Python/NumPy bridge cannot be built")
    endif()
endif()

