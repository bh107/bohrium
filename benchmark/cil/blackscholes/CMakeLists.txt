cmake_minimum_required(VERSION 2.8)

find_package(Mono)
set_package_properties(Mono PROPERTIES DESCRIPTION "An open source implementation the CLR" URL "www.mono-project.com")
set_package_properties(Mono PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the NumCIL package")

if(MONO_FOUND)

  GET_FILENAME_COMPONENT(projname ${CMAKE_CURRENT_BINARY_DIR} NAME)

  # Enumerate over all .tt files and apply T4 to them
  SET(numcil_tt_outputs)
  FILE(GLOB_RECURSE files *.tt)

  FOREACH(file ${files})
    SET(src ${file})
    GET_FILENAME_COMPONENT(basename ${file} NAME_WE)
    GET_FILENAME_COMPONENT(targetfolder ${file} PATH)
    SET(dst ${targetfolder}/${basename}.cs)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${dst}
      DEPENDS ${file}
      COMMAND ${MONO_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../../bridge/NumCIL/buildsupport/TextTransform.exe --out=${dst} ${src}
    )
    SET(numcil_tt_outputs ${numcil_tt_outputs} ${dst})
  ENDFOREACH(file)

  # Find source files
  file(GLOB_RECURSE SRCS *.cs *.sln *.csproj)
  set(output_binary ${CMAKE_CURRENT_BINARY_DIR}/bin/Release/${projname}.exe)
  add_custom_command(
      OUTPUT ${output_binary}
      COMMAND ${XBUILD_EXECUTABLE} /property:Configuration=Release /property:OutputPath=${CMAKE_CURRENT_BINARY_DIR}/bin/Release ${CMAKE_CURRENT_SOURCE_DIR}/${projname}.sln
      DEPENDS ${SRCS} ${numcil_tt_outputs} numcil_bohrium
  )

  # Add a target, and register a simple name
  add_custom_target(
    numcil_test_${projname} ALL
    DEPENDS numcil_bohrium numcil bhc ${output_binary}
  )

  install(FILES ${output_binary} DESTINATION share/bohrium/benchmark/cil/bin COMPONENT bohrium-numcil)
  install(FILES ${SRCS} DESTINATION share/bohrium/benchmark/cil/${projname} COMPONENT bohrium-numcil)

else()
  message("Cannot build NumCIL as Mono or xbuild is missing")
endif()
