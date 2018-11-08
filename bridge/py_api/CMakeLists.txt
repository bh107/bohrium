cmake_minimum_required(VERSION 2.8)
set(BRIDGE_PY_API ${BRIDGE_C} CACHE BOOL "BRIDGE-PY_API: Build the PY_API-bridge.")
if (NOT BRIDGE_PY_API)
    return()
endif ()
if (NOT BRIDGE_C)
    message(FATAL_ERROR "BRIDGE_C is required for BRIDGE_PY_API, so please set BRIDGE_C to ON or BRIDGE_PY_API to OFF")
endif ()

if (NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "The Python API bridge cannot be built, PYTHONINTERP_FOUND not found! \
    Set PYTHONINTERP or set BRIDGE_PY_API to OFF to disable Python build.")
endif ()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip --version
        RESULT_VARIABLE VR_RES
        OUTPUT_VARIABLE VR_OUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT "${VR_RES}" STREQUAL "0")
    message(FATAL_ERROR "The Python API bridge cannot be built, the Python package \"pip\" must be installed. \
    Install it or set BRIDGE_PY_API to OFF to disable Python build.")
else()
    message(STATUS "Python package \"pip\" found: ${VR_OUT}.")
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -m virtualenv --version
        RESULT_VARIABLE VR_RES
        OUTPUT_VARIABLE VR_OUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT "${VR_RES}" STREQUAL "0")
    message(FATAL_ERROR "The Python API bridge cannot be built, the Python package \"virtualenv\" must be installed. \
    Install it or set BRIDGE_PY_API to OFF to disable Python build.")
else()
    message(STATUS "Python package \"virtualenv\" found (version ${VR_OUT}).")
endif()

# The user can set `PY_WHEEL` to export Python wheel packages instead of install them
set(PY_WHEEL "OFF" CACHE PATH "Path to the Python wheel output directory.")

# For each interpreter in the user specified list `PY_EXE_LIST`, we build a package.
set(PY_EXE_LIST "${PYTHON_EXECUTABLE}" CACHE STRING "List of Python interpreters to build with.")
message(STATUS "PY_EXE_LIST: \"${PY_EXE_LIST}\"")
foreach(PY_EXE ${PY_EXE_LIST})

    # Define `VR_PATH` based on the `PY_EXE` path in order to avoid name clash
    string(MAKE_C_IDENTIFIER ${PY_EXE} PY_EXE_IDENT)
    set(VR_PATH ${CMAKE_CURRENT_BINARY_DIR}/vr_${PY_EXE_IDENT})

    # We run `build.bash` in order to generate the wheel package.
    file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.py ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
    add_custom_command(OUTPUT ${VR_PATH}/wheel_py_api/package_name.txt
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build.bash ${VR_PATH} ${PY_EXE} ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/build.bash)

    # The `py_api` target is always called, but it will only call the `add_custom_command()` above when it's
    # depdendencies (the files in ${SRC}) changes.
    add_custom_target(py_api_${PY_EXE_IDENT} ALL
            DEPENDS bhc ${VR_PATH}/wheel_py_api/package_name.txt)

    if(PY_WHEEL)
        # Exporting the wheel package to `PY_WHEEL`.
        install(CODE "file(READ ${VR_PATH}/wheel_py_api/package_name.txt WHEEL_NAME)
                      message(STATUS \"Writing \${WHEEL_NAME} to ${PY_WHEEL}\")
                      file(MAKE_DIRECTORY ${PY_WHEEL})
                      configure_file(\${WHEEL_NAME} ${PY_WHEEL} COPYONLY)")
    else()
        # Install using `pip install --user -I` command where `PYTHONUSERBASE` is set to CMAKE_INSTALL_PREFIX
        install(CODE "file(READ ${VR_PATH}/wheel_py_api/package_name.txt WHEEL_NAME)
                      message(STATUS \"Exporting \${WHEEL_NAME} in \${CMAKE_INSTALL_PREFIX}\")
                      set(ENV{PYTHONUSERBASE} ${CMAKE_INSTALL_PREFIX})
                      execute_process(COMMAND ${PY_EXE} -m pip install --user -I \${WHEEL_NAME}
                                      RESULT_VARIABLE RES)
                      if(NOT \"\${RES}\" STREQUAL \"0\")
                        message(FATAL_ERROR \"Fatal error when installing \${WHEEL_NAME}: \${RES}.\")
                      endif()")
    endif()
endforeach()
