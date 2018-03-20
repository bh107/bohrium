# - Try to find cython
#
# defines
#
# CYTHON_FOUND - system has cython
# CYTHON_VERSION - cython version


# Finding cython involves calling the Python interpreter
if(CYTHON_FIND_REQUIRED)
    find_package(PythonInterp REQUIRED)
else()
    find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
    set(CYTHON_FOUND FALSE)
    return()
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import cython; print(cython.__version__);"
        RESULT_VARIABLE _CYTHON_VERSION_RESULT
        OUTPUT_VARIABLE CYTHON_VERSION
        ERROR_VARIABLE _CYTHON_ERROR_VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _CYTHON_VERSION_RESULT MATCHES 0)
    if(CYTHON_FIND_REQUIRED)
        message(FATAL_ERROR
                "CYTHON import failure:\n${_CYTHON_ERROR_VALUE}")
    endif()
    set(CYTHON_FOUND FALSE)
    return()
else()
    set(CYTHON_FOUND TRUE)
endif()

find_package_message(CYTHON "Found Cython: version \"${CYTHON_VERSION}\"" "${CYTHON_VERSION}")
