# - Sets LIBDIR to 'lib' or 'lib64'

if (EXISTS "/etc/debian_version") # Debian or Ubuntu?
  set (_libdir_def "lib")
elseif (EXISTS "/etc/fedora-release" OR
        EXISTS "/etc/redhat-release" OR
        EXISTS "/etc/slackware-version" OR
        EXISTS "/etc/gentoo-release")
  set (_libdir_def "lib64")
else ()
  set (_libdir_def "lib")
endif()

set (LIBDIR "${_libdir_def}" CACHE STRING "Name of library directory (not the path)")
