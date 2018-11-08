cmake_minimum_required(VERSION 2.8)
set(BRIDGE_PY_API ${BRIDGE_C} CACHE BOOL "BRIDGE-PY_API: Build the PY_API-bridge.")
if (NOT BRIDGE_PY_API)
    return()
endif ()
if (NOT BRIDGE_C)
    message(FATAL_ERROR "BRIDGE_C is required for BRIDGE_PY_API, so please set BRIDGE_C to ON or BRIDGE_PY_API to OFF")
endif ()

if (PYTHONINTERP_FOUND)

    # We run `build.bash` in order to generate the wheel package.
    set(VR_PATH ${CMAKE_CURRENT_BINARY_DIR}/vr${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.py ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
    add_custom_command(OUTPUT ${VR_PATH}/wheel_py_api/package_name.txt
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build.bash ${VR_PATH} ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/build.bash)

    # The `py_api` target is always called, but it will only call the `add_custom_command()` above when it's
    # depdendencies (the files in ${SRC}) changes.
    add_custom_target(py_api ALL DEPENDS ${VR_PATH}/wheel_py_api/package_name.txt)

    install(CODE "file(READ ${VR_PATH}/wheel_py_api/package_name.txt WHEEL_NAME)
                  message(STATUS \"Installing \${WHEEL_NAME}\")
                  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install --user -I \${WHEEL_NAME})")

else ()
    message(STATUS "The Python API bridge cannot be built, PYTHONINTERP_FOUND not found!")
endif ()

