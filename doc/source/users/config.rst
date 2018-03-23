Runtime Configuration
---------------------

.. highlight:: ruby

Bohrium supports a broad range of front and back-ends.
The default backend is OpenMP. You can change which backend to use by defining the ``BH_STACK`` environment variable:

* The CPU backend that make use of OpenMP: ``BH_STACK=openmp``
* The GPU backend that make use of OpenCL: ``BH_STACK=opencl``
* The GPU backend that make use of CUDA: ``BH_STACK=cude``

For debug information when running Bohrium, use the following environment variables::

  BH_<backend>_PROF=true     -- Prints a performance profile at the end of execution.
  BH_<backend>_VERBOSE=true  -- Prints a lot of information including the source of the JIT compiled kernels. Enables per-kernel profiling when used together with BH_OPENMP_PROF=true.
  BH_SYNC_WARN=true          -- Show Python warnings in all instances when copying data to Python.
  BH_MEM_WARN=true           -- Show warnings when memory accesses are problematic.
  BH_<backend>_GRAPH=true    -- Dump a dependency graph of the instructions send to the back-ends (.dot file).
  BH_<backend>_VOLATILE=true -- Declare temporary variables using `volatile`, which avoid precision differences because of Intel's use of 80-bit floats internally.

Particularly, ``BH_<backend>_PROF=true`` is very useful to explore why Bohrium might not perform as expected::

    BH_OPENMP_PROF=1 python -m bohrium heat_equation.py --size=4000*4000*100
    heat_equation.py - target: bhc, bohrium: True, size: 4000*4000*100, elapsed-time: 6.446084

    [OpenMP] Profiling:
    Fuse cache hits:                 199/203 (98.0296%)
    Codegen cache hits               299/304 (98.3553%)
    Kernel cache hits                300/304 (98.6842%)
    Array contractions:              700/1403 (49.8931%)
    Outer-fusion ratio:              13/23 (56.5217%)

    Max memory usage:                0 MB
    Syncs to NumPy:                  99
    Total Work:                      12800400099 operations
    Throughput:                      1.9235e+09ops
    Work below par-threshold (1000): 0%

    Wall clock:                      6.65473s
    Total Execution:                 6.04354s
      Pre-fusion:                    0.000761211s
      Fusion:                        0.00411354s
      Codegen:                       0.00192224s
      Compile:                       0.285544s
      Exec:                          4.91214s
      Copy2dev:                      0s
      Copy2host:                     0s
      Ext-method:                    0s
      Offload:                       0s
      Other:                         0.839052s

    Unaccounted for (wall - total):  0.611198s

Which tells us, among other things, that the execution of the compiled JIT kernels (``Exec``) takes 4.91 seconds, the JIT compilation (``Compile``) takes 0.29 seconds, and the time spend outside of Bohrium (``Unaccounted for``) takes 0.61.


OpenCL Configuration
~~~~~~~~~~~~~~~~~~~~

In order to choose which OpenCL platform and device to use, set the following environment variables::

  # OpenCL platform. -1 means automatic. Other numbers will index into list of platforms.
  BH_OPENCL_PLATFORM_NO = -1

  # Device type can be one of 'auto', 'gpu', 'cpu', 'accelerator', or 'default'
  BH_OPENCL_DEVICE_TYPE = auto

You can also set the options in the configure file under the ``[opencl]`` section.

Also under the ``[opencl]`` section, you can set the OpenCL work group sizes::

  # OpenCL work group sizes
  work_group_size_1dx = 128
  work_group_size_2dx = 32
  work_group_size_2dy = 4
  work_group_size_3dx = 32
  work_group_size_3dy = 2
  work_group_size_3dz = 2



Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

In order to configure the runtime setup of Bohrium you must provide a configuration file to Bohrium. The installation of Bohrium installs a default configuration file in ``/etc/bohrium/config.ini`` when doing a system-wide installation, ``~/.bohrium/config.ini`` when doing a local installation, and ``<python library>/bohrium/config.ini`` when doing a pip installation.

At runtime Bohrium will search through the following prioritized list in order to find the configuration file:

* The environment variable ``BH_CONFIG``
* The config within the Python package ``bohrium/config.ini`` (in the same directory as ``__init__.py``)
* The home directory config ``~/.bohrium/config.ini``
* The system-wide config ``/usr/local/etc/bohrium/config.ini``
* The system-wide config ``/etc/bohrium/config.ini``

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


The configuration of a component can be overwritten with environment variables using the naming convention ``BH_[COMPONENT]_[OPTION]``, below are a couple of examples controlling the behavior of the CPU vector engine::

  BH_OPENMP_PROF=true    -- Prints a performance profile at the end of execution.
  BH_OPENMP_VERBOSE=true -- Prints a lot of information including the source of the JIT compiled kernels. Enables per-kernel profiling when used together with BH_OPENMP_PROF=true.

Useful environment variables::

  BH_SYNC_WARN=true       -- Show Python warnings in all instances when copying data to Python.
  BH_MEM_WARN=true        -- Show warnings when memory accesses are problematic.
  BH_<backend>_GRAPH=true -- Dump a dependency graph of the instructions send to the back-ends (.dot file).
  BH_<backend>_VOLATILE=true -- Declare temporary variables using `volatile`, which avoid precision differences because of Intel's use of 80-bit floats internally.
