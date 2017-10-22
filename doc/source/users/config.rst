Runtime Configuration
---------------------

Bohrium supports a broad range of front and back-ends.
In order to configure the runtime setup of Bohrium you must provide a configuration file to Bohrium. The installation of Bohrium installs a default configuration file in ``/etc/bohrium/config.ini`` when doing a system-wide installation and ``~/.bohrium/config.ini`` when doing a local installation.

At runtime Bohrium will search through the following prioritized list in order to find the configuration file:

* The environment variable ``BH_CONFIG``
* The home directory config ``~/.bohrium/config.ini`` (Windows: %APPDATA%\bohrium\config.ini)
* The system-wide config ``/etc/bohrium/config.ini`` (Windows: %PROGRAMFILES%\bohrium\config.ini)


The default configuration file looks similar to the config below::

  #
  # Stack configurations, which are a comma separated lists of components.
  # NB: 'stacks' is a reserved section name and 'default'
  #     is used when 'BH_STACK' is unset.
  #     The bridge is never part of the list
  #
  [stacks]
  default    = bcexp, bccon, node, openmp
  openmp     = bcexp, bccon, node, openmp
  opencl     = bcexp, bccon, node, opencl, openmp

  #
  # Managers
  #

  [node]
  impl = /usr/lib/libbh_vem_node.so
  timing = false

  [proxy]
  address = localhost
  port = 4200
  impl = /usr/lib/libbh_vem_proxy.so


  #
  # Filters - Helpers / Tools
  #
  [pprint]
  impl = /usr/lib/libbh_filter_pprint.so

  #
  # Filters - Bytecode transformers
  #
  [bccon]
  impl = /usr/lib/libbh_filter_bccon.so
  collect = true
  stupidmath = true
  muladd = true
  reduction = false
  find_repeats = false
  timing = false
  verbose = false

  [bcexp]
  impl = /usr/lib/libbh_filter_bcexp.so
  powk = true
  sign = false
  repeat = false
  reduce1d = 32000
  timing = false
  verbose = false

  [noneremover]
  impl = /usr/lib/libbh_filter_noneremover.so
  timing = false
  verbose = false

  #
  # Engines
  #
  [openmp]
  impl = /usr/lib/libbh_ve_openmp.so
  tmp_bin_dir = /usr/var/bohrium/object
  tmp_src_dir = /usr/var/bohrium/source
  dump_src = true
  verbose = false
  prof = false #Profiling statistics
  compiler_cmd = "/usr/bin/x86_64-linux-gnu-gcc"
  compiler_inc = "-I/usr/share/bohrium/include"
  compiler_lib = "-lm -L/usr/lib -lbh"
  compiler_flg = "-x c -fPIC -shared  -std=gnu99  -O3 -march=native -Werror -fopenmp"
  compiler_openmp = true
  compiler_openmp_simd = false

  [opencl]
  impl = /usr/lib/libbh_ve_opencl.so
  verbose = false
  prof = false #Profiling statistics
  # Additional options given to the opencl compiler. See documentation for clBuildProgram
  compiler_flg = "-I/usr/share/bohrium/include"
  serial_fusion = false # Topological fusion is default


The configuration file consists of two things: ``components`` and orchestration of components in ``stacks``.

Components marked with square brackets. For example ``[node]``, ``[openmp]``, ``[opencl]`` are all components available for the runtime system.

The ``stacks`` define different default configurations of the runtime environment and one can switch between them using the environment var ``BH_STACK``.

Environment Variables
---------------------

The configuration of a component can be overwritten with environment variables using the naming convention ``BH_[COMPONENT]_[OPTION]``, below are a couple of examples controlling the behavior of the CPU vector engine::

  BH_OPENMP_PROF=true    -- Prints a performance profile at the end of execution.
  BH_OPENMP_VERBOSE=true -- Prints a lot of information including the source of the JIT compiled kernels.

Useful environment variables::

  BH_SYNC_WARN=true       -- Show Python warnings in all instances when copying data to Python.
  BH_MEM_WARN=true        -- Show warnings when memory accesses are problematic.
  BH_<backend>_GRAPH=true -- Dump a dependency graph of the instructions send to the back-ends (.dot file).
  BH_<backend>_VOLATILE=true -- Declare temporary variables using `volatile`, which avoid precision differences because of Intel's use of 80-bit floats internally.
