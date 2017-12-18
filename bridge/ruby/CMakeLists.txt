cmake_minimum_required(VERSION 2.8)
set(BRIDGE_RUBY "OFF" CACHE BOOL "BRIDGE-RUBY: Build the RUBY-bridge.")
if(NOT BRIDGE_RUBY)
    return()
endif()
if(NOT BRIDGE_BHXX)
    message(FATAL_ERROR "BRIDGE_BHXX is required for BRIDGE_RUBY, so please set BRIDGE_BHXX to ON or BRIDGE_RUBY to OFF")
endif()

find_package(Ruby)
set_package_properties(Ruby PROPERTIES DESCRIPTION "Ruby programming language" URL "www.ruby-lang.org")
set_package_properties(Ruby PROPERTIES TYPE RECOMMENDED PURPOSE "Enables the Ruby bridge.")

if(RUBY_FOUND)
    # Make sure `bundle` is up-to-date
    add_custom_target(
        bundle ALL
        COMMAND bundle install --path vendor/bundle
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bohrium
    )

    # The rest must be done in `install` phase
    install(CODE "
        # Run `rake compile` to generate bohrium.bundle
        EXECUTE_PROCESS(
            COMMAND bundle exec rake clean compile -- --with-bohrium-include=${CMAKE_INSTALL_PREFIX}/include/bohrium --with-bhxx-dir=${CMAKE_INSTALL_PREFIX}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bohrium
        )
        # Build gem
        EXECUTE_PROCESS(
            COMMAND gem build bohrium.gemspec
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bohrium
        )
        # Move it to binary directory
        EXECUTE_PROCESS(
            COMMAND mv bohrium-0.1.0.gem ${CMAKE_CURRENT_BINARY_DIR}/.
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bohrium
        )
        # Install the gem
        EXECUTE_PROCESS(
            COMMAND gem install bohrium-0.1.0.gem --user-install -- --with-bohrium-include=${CMAKE_INSTALL_PREFIX}/include/bohrium --with-bhxx-dir=${CMAKE_INSTALL_PREFIX}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    ")
else()
    message(STATUS "The Ruby bridge cannot be built, because Ruby was not found.")
    return()
endif()
