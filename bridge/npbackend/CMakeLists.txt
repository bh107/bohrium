cmake_minimum_required(VERSION 2.8)
set(BRIDGE_NPBACKEND ${BRIDGE_C} CACHE BOOL "BRIDGE-NPBACKEND: Build the NPBACKEND-bridge.")
if(NOT BRIDGE_NPBACKEND)
    return()
endif()
if(NOT BRIDGE_C)
    message(FATAL_ERROR "BRIDGE_C is required for BRIDGE_NPBACKEND, so please set BRIDGE_C to ON or BRIDGE_NPBACKEND to OFF")
endif()

if(PYTHONINTERP_FOUND)

    # We run `build.bash` in order to generate the wheel package.
    set(VR_PATH ${CMAKE_BINARY_DIR}/bridge/py_api/vr${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    file(GLOB_RECURSE SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/*.py
            ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx
            ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
    add_custom_command(OUTPUT ${VR_PATH}/wheel_npbackend/package_name.txt
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build.bash ${VR_PATH} ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/build.bash)

    # The `npbackend` target is always called, but it will only call the `add_custom_command()` above when it's
    # depdendencies (the files in ${SRC}) changes.
    add_custom_target(npbackend ALL DEPENDS py_api ${VR_PATH}/wheel_npbackend/package_name.txt)

    install(CODE "file(READ ${VR_PATH}/wheel_npbackend/package_name.txt WHEEL_NAME)
                  message(STATUS \"Installing \${WHEEL_NAME} in \${CMAKE_INSTALL_PREFIX}\")
                  set(ENV{PYTHONUSERBASE} ${CMAKE_INSTALL_PREFIX})
                  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install --user -I \${WHEEL_NAME}
                                  RESULT_VARIABLE RES)
                  if(NOT \"\${RES}\" STREQUAL \"0\")
	                message(FATAL_ERROR \"Fatal error when installing \${WHEEL_NAME}: \${RES}.\")
                  endif()")

else()
    message(STATUS "The Python/NumPy bridge cannot be built, PYTHONINTERP_FOUND not found!")
endif()
