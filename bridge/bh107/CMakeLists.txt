cmake_minimum_required(VERSION 2.8)
set(BRIDGE_BH107 ${BRIDGE_C} CACHE BOOL "BRIDGE_BH107: Build the BH107-bridge.")
if(NOT BRIDGE_BH107)
    return()
endif()

# Notice, all dependency checks on Python, pip, virtualenv etc is done by the `py_api` module.
# The dependency on cython and NumPy is automatically installed in the virtualenv

# For each interpreter in the user specified list `PY_EXE_LIST`, we build a package.
foreach(PY_EXE ${PY_EXE_LIST})

    # Define `VR_PATH` which should point to the virtualenv over at the `py_api` module
    if(${CMAKE_VERSION} VERSION_LESS "3.0.0")
        string(REPLACE "/" "_" PY_EXE_IDENT ${PY_EXE})
    else()
        string(MAKE_C_IDENTIFIER ${PY_EXE} PY_EXE_IDENT)
    endif()
    set(VR_PATH ${CMAKE_BINARY_DIR}/bridge/py_api/vr_${PY_EXE_IDENT})

    if(NOT PY_WHEEL)
        # Install using `pip install --user -I` command where `PYTHONUSERBASE` is
        # set to CMAKE_INSTALL_PREFIX
        install(CODE "message(STATUS \"Installing bh107 in \${CMAKE_INSTALL_PREFIX}\")
                      set(ENV{PYTHONUSERBASE} ${CMAKE_INSTALL_PREFIX})
                      execute_process(COMMAND ${PY_EXE} -m pip install --no-deps --user -I .
                                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                                      RESULT_VARIABLE RES)
                      if(NOT \"\${RES}\" STREQUAL \"0\")
                        message(FATAL_ERROR \"Fatal error when installing bh107: \${RES}.\")
                      endif()")
    endif()
    add_custom_target(bh107_${PY_EXE_IDENT} ALL DEPENDS py_api_${PY_EXE_IDENT})
endforeach()
