cmake_minimum_required(VERSION 2.8)
set(BRIDGE_NPBACKEND ${BRIDGE_C} CACHE BOOL "BRIDGE-NPBACKEND: Build the NPBACKEND-bridge.")
if(NOT BRIDGE_NPBACKEND)
    return()
endif()

# Notice, all dependecy checks on Python, pip, virtualenv etc is done by the `py_api` module.
# The dependency on cython and NumPy is automatically installed in the virtualenv

# For each interpreter in the user specified list `PY_EXE_LIST`, we build a package.
foreach(PY_EXE ${PY_EXE_LIST})

    # Define `VR_PATH` which should point to the virtualenv over at the `py_api` module
    string(MAKE_C_IDENTIFIER ${PY_EXE} PY_EXE_IDENT)
    set(VR_PATH ${CMAKE_BINARY_DIR}/bridge/py_api/vr_${PY_EXE_IDENT})

    # We run `build.bash` in order to generate the wheel package.
    file(GLOB_RECURSE SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/*.py
            ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx
            ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
    add_custom_command(OUTPUT ${VR_PATH}/wheel_npbackend/package_name.txt
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build.bash ${VR_PATH} ${PY_EXE} ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/build.bash)

    # The `npbackend` target is always called, but it will only call the `add_custom_command()` above when it's
    # depdendencies (the files in ${SRC}) changes.
    add_custom_target(npbackend_${PY_EXE_IDENT} ALL
            DEPENDS py_api_${PY_EXE_IDENT} ${VR_PATH}/wheel_npbackend/package_name.txt)

    if(PY_WHEEL)
        # Exporting the wheel package to `PY_WHEEL`.
        install(CODE "file(READ ${VR_PATH}/wheel_npbackend/package_name.txt WHEEL_NAME)
                      message(STATUS \"Exporting \${WHEEL_NAME} to ${PY_WHEEL}\")
                      file(MAKE_DIRECTORY ${PY_WHEEL})
                      configure_file(\${WHEEL_NAME} ${PY_WHEEL} COPYONLY)")
    else()
        # Install using `pip install --user -I` command where `PYTHONUSERBASE` is set to CMAKE_INSTALL_PREFIX
        install(CODE "file(READ ${VR_PATH}/wheel_npbackend/package_name.txt WHEEL_NAME)
                      message(STATUS \"Installing \${WHEEL_NAME} in \${CMAKE_INSTALL_PREFIX}\")
                      set(ENV{PYTHONUSERBASE} ${CMAKE_INSTALL_PREFIX})
                      execute_process(COMMAND ${PY_EXE} -m pip install --user -I \${WHEEL_NAME}
                                      RESULT_VARIABLE RES)
                      if(NOT \"\${RES}\" STREQUAL \"0\")
                        message(FATAL_ERROR \"Fatal error when installing \${WHEEL_NAME}: \${RES}.\")
                      endif()")
    endif()
endforeach()
