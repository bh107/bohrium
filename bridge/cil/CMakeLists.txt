cmake_minimum_required(VERSION 2.8)
set(BRIDGE_CIL true CACHE BOOL "BRIDGE-CIL: Build the CIL-bridge.")
if(NOT BRIDGE_CIL)
    return()
endif()

set(Mono_FIND_QUIETLY 1)
find_package(Mono)

if(BRIDGE_CIL AND MONO_FOUND AND PYTHONINTERP_FOUND)

  # Enumerate over all .tt files and apply T4 to them
  SET(numcil_bohrium_tt_outputs)
  FILE(GLOB files NumCIL.Bohrium/*.tt)

  FOREACH(file ${files})
    SET(src ${file})
    GET_FILENAME_COMPONENT(basename ${file} NAME_WE)
    GET_FILENAME_COMPONENT(targetfolder ${file} PATH)
    SET(dst ${targetfolder}/${basename}.cs)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${dst}
      DEPENDS ${file}
      COMMAND ${MONO_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../NumCIL/buildsupport/TextTransform.exe --out=${dst} ${src}
    )
    SET(numcil_bohrium_tt_outputs ${numcil_bohrium_tt_outputs} ${dst})
  ENDFOREACH(file)

  set(autogen_file ${CMAKE_CURRENT_SOURCE_DIR}/codegen/output/bhc_array_operations.cs)

  add_custom_command(
      OUTPUT ${autogen_file}
      COMMAND ${PYTHON_EXECUTABLE} gen_array_operations.py codegen/output
      DEPENDS gen_array_operations.py
            ../../core/codegen/opcodes.json
            ../cpp/codegen/element_types.json
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  set(build_type "Release")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(build_type "Debug")
  endif()

  file(GLOB build_deps ${CMAKE_CURRENT_SOURCE_DIR}/NumCIL.Bohrium/*.cs)
  set(target_file ${CMAKE_CURRENT_BINARY_DIR}/NumCIL.Bohrium/bin/${build_type}/NumCIL.Bohrium.dll)
  add_custom_command(
      OUTPUT ${target_file}
      COMMAND ${XBUILD_EXECUTABLE} /verbosity:quiet /nologo /property:Configuration=${build_type} /property:OutputPath=${CMAKE_CURRENT_BINARY_DIR}/NumCIL.Bohrium/bin/${build_type} ${CMAKE_CURRENT_SOURCE_DIR}/NumCIL.Bohrium/NumCIL.Bohrium.csproj
      DEPENDS ${numcil_bohrium_tt_outputs} ${autogen_file} ${build_deps} ${CMAKE_CURRENT_SOURCE_DIR}/NumCIL.Bohrium/NumCIL.Bohrium.csproj
  )

  # Add a target, and register a simple name
  add_custom_target(
    numcil_bohrium ALL
    DEPENDS numcil bhc ${target_file}
  )

  install(FILES ${target_file} DESTINATION lib/mono COMPONENT bohrium-numcil)
  install(CODE "execute_process(COMMAND gacutil -i ${CMAKE_INSTALL_PREFIX}/lib/mono/NumCIL.Bohrium.dll)")  

endif()
