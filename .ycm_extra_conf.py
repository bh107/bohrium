import os

# Canidates for the builddir (relative to the path of this file),
# which we use to find some generated files:
builddir_canidates = [ "build", "b" ]

# Determine actual builddir
builddir = builddir_canidates[0]
for d in builddir_canidates:
  if os.path.exists(d):
    builddir = d
    break

# This file is loosely based upon the file
# cpp/ycm/.ycm_extra_conf.py from the youcompleteme daemon process
# available on github:
# https://github.com/Valloric/ycmd/blob/master/cpp/ycm/.ycm_extra_conf.py

# These are the compilation flags that will be used in case there's no
# compilation database set (by default, one is not set).
flags = [
    # Warnings: For a very detailed discussion about this
    # see the following stackexchange post:
    # https://programmers.stackexchange.com/questions/122608#124574
    '-Wall',
    '-Wextra',
    '-Wnon-virtual-dtor',
    '-Woverloaded-virtual',
    '-Wold-style-cast',
    '-Wcast-align',
    '-Wconversion',
    '-Wsign-conversion',
    '-pedantic',
    '-Werror',
    # Generate unwind information
    '-fexceptions',
    # TODO Why is this needed
    '-Dbhxx_EXPORTS',
    # Compile as c++11
    '-std=c++11',
    #
    # Treat .h header files as c++:
    '-x', 'c++',
    # Include other libraries and show errors and
    # warnings within them
    # To suppress errors shown here, use "-isystem"
    # instead of "-I"
    '-I', 'include',
    '-I', builddir + '/include',
    '-I', './bridge/cxx/include',
    '-I', builddir + '/bridge/cxx/include',
    '-I', './bridge/cpp',  # TODO Why is this needed
    # Explicit clang includes:
    '-isystem', '/usr/lib/ycmd/clang_includes',
]

def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag

    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )

    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break

      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break

    if new_flag:
      new_flags.append( new_flag )
  return new_flags

def FlagsForFile( filename, **kwargs ):
  relative_to = os.path.dirname( os.path.abspath( __file__ ) )
  final_flags = MakeRelativePathsInFlagsAbsolute( flags, relative_to )

  return {
    'flags': final_flags,
    'do_cache': True
  }
