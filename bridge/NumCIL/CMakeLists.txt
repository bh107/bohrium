cmake_minimum_required(VERSION 2.8)
set(BRIDGE_NUMCIL true CACHE BOOL "BRIDGE-NUMCIL: Build the NUMCIL-bridge.")
if(NOT BRIDGE_NUMCIL)
    return()
endif()

find_package(Mono)
set_package_properties(Mono PROPERTIES DESCRIPTION "An open source implementation the CLR" URL "www.mono-project.com")
set_package_properties(Mono PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the NumCIL package")

if(BRIDGE_NUMCIL AND MONO_FOUND)

  # Enumerate over all .tt files and apply T4 to them
  SET(numcil_tt_outputs)
  FILE(GLOB files NumCIL/*.tt NumCIL/UFunc/*.tt NumCIL.Unsafe/*.tt)

  FOREACH(file ${files})
    SET(src ${file})
    GET_FILENAME_COMPONENT(basename ${file} NAME_WE)
    GET_FILENAME_COMPONENT(targetfolder ${file} PATH)
    SET(dst ${targetfolder}/${basename}.cs)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${dst}
      DEPENDS ${file}
      COMMAND ${MONO_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/buildsupport/TextTransform.exe --out=${dst} ${src}
    )
    SET(numcil_tt_outputs ${numcil_tt_outputs} ${dst})
  ENDFOREACH(file)

  set(build_type "Release")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(build_type "Debug")
  endif()  

  file(GLOB sln_sources NumCIL/*.cs NumCIL/*.csproj NumCIL.Unsafe/*.cs NumCIL.Unsafe/*.csproj)
  set(numcil_binaries ${CMAKE_CURRENT_BINARY_DIR}/bin/${build_type}/NumCIL.dll ${CMAKE_CURRENT_BINARY_DIR}/bin/${build_type}/NumCIL.Unsafe.dll)
  add_custom_command(
      OUTPUT ${numcil_binaries}
      COMMAND ${XBUILD_EXECUTABLE} /verbosity:quiet /nologo /property:Configuration=${build_type} /property:OutputPath=${CMAKE_CURRENT_BINARY_DIR}/bin/${build_type} ${CMAKE_CURRENT_SOURCE_DIR}/NumCIL.sln
      DEPENDS ${sln_sources} ${numcil_tt_outputs}
  )

  # Re-usable simple name for NumCIL target
  add_custom_target(
    numcil ALL
    DEPENDS ${numcil_binaries}
  )

  install(FILES ${numcil_binaries} DESTINATION lib/mono COMPONENT bohrium-numcil)
  install(CODE "execute_process(COMMAND gacutil -i ${CMAKE_INSTALL_PREFIX}/lib/mono/NumCIL.dll)")  
  install(CODE "execute_process(COMMAND gacutil -i ${CMAKE_INSTALL_PREFIX}/lib/mono/NumCIL.Unsafe.dll)")  

endif()
