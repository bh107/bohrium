cmake_minimum_required(VERSION 2.8)

find_package(Mono)
set_package_properties(Mono PROPERTIES DESCRIPTION "An open source implementation the CLR" URL "www.mono-project.com")
set_package_properties(Mono PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the NumCIL package")

if(MONO_FOUND)

  # Find source files
  file(GLOB_RECURSE SRCS *.cs *.sln *.csproj)
  set(numcil_test_binaries ${CMAKE_CURRENT_BINARY_DIR}/bin/Release/UnitTest.exe)
  add_custom_command(
      OUTPUT ${numcil_test_binaries}
      COMMAND ${XBUILD_EXECUTABLE} /property:Configuration=Release /property:OutputPath=${CMAKE_CURRENT_BINARY_DIR}/bin/Release ${CMAKE_CURRENT_SOURCE_DIR}/Unittest.sln
      DEPENDS ${SRCS}
  )

  # Add a target, and register a simple name
  add_custom_target(
    numcil_test ALL
    DEPENDS numcil_bohrium numcil bhc ${numcil_test_binaries}
  )

  install(FILES ${numcil_test_binaries} DESTINATION share/bohrium/test/cil/bin COMPONENT bohrium-numcil)
  install(FILES ${SRCS} DESTINATION share/bohrium/test/cil/Unittest COMPONENT bohrium-numcil)

else()
  message("Cannot build NumCIL as Mono or xbuild is missing")
endif()
