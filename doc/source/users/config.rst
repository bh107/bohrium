Runtime Configuration
---------------------

Bohrium supports a broad range of frontends and backends.
In order to configure the runtime setup of Bohrium you must provide a configuration file to Bohrium. The installation of Bohrium installs a default configuration file in ``/etc/bohrium/config.ini`` when doing a system-wide installation and ``~/.bohrium/config.ini`` when doing a local installation.

At runtime Bohrium will search through the following prioritized list in order to find the configuration file:

* The environment variable ``BH_CONFIG``
* The home directory config ``~/.bohrium/config.ini`` (Windows: %APPDATA%\bohrium\config.ini)
* The system-wide config ``/etc/bohrium/config.ini`` (Windows: %PROGRAMFILES%\bohrium\config.ini)


The default configuration file looks similar to the config below::

  #
  # Stack configuration
  #
  [stack_default]
  type = stack
  stack_default = bcexp_cpu
  bcexp_cpu = bccon
  bccon = topological
  topological = node
  node = cpu

  #
  # Other stack configurations (for reference, experiments, testing, etc.)
  #
  # Use the BH_STACK env-var to choose another stack configuration.
  #
  # Such as BH_STACK="stack_gpu"
  # or
  # modify the default stack configuration above ("stack_default").
  #
  [stack_gpu]
  type = stack
  stack_gpu = bcexp
  bcexp = topological
  topological = node
  node = gpu
  gpu = cpu

  [stack_fuseprinter]
  type = stack
  stack_fuseprinter = bcexp
  bcexp = topological
  topological = pricer
  pricer = fuseprinter
  fuseprinter = node
  node = cpu

  #
  # Component configuration
  #

  #
  # Bridges
  #
  [bridge]
  type = bridge
  children = bcexp

  #
  # Managers
  #

  [proxy]
  type = vem
  port = 4200
  impl = /opt/bohrium/lib/libbh_vem_proxy.so
  children = node

  [cluster]
  type = vem
  children = node
  impl = /opt/bohrium/lib/libbh_vem_cluster.so

  [node]
  type = vem
  children = topological
  impl = /opt/bohrium/lib/libbh_vem_node.so
  timing = false

  #
  # Fusers
  #

  [singleton]
  type = fuser
  impl = /opt/bohrium/lib/libbh_fuser_singleton.so
  children = cpu
  cache_path=/opt/bohrium/var/bohrium/fuse_cache

  [topological]
  type = fuser
  impl = /opt/bohrium/lib/libbh_fuser_topological.so
  children = cpu
  #cache_path=/opt/bohrium/var/bohrium/fuse_cache

  [gentle]
  type = fuser
  impl = /opt/bohrium/lib/libbh_fuser_gentle.so
  children = cpu
  cache_path=/opt/bohrium/var/bohrium/fuse_cache

  [greedy]
  type = fuser
  impl = /opt/bohrium/lib/libbh_fuser_greedy.so
  children = cpu
  cache_path=/opt/bohrium/var/bohrium/fuse_cache

  [optimal]
  type = fuser
  impl = /opt/bohrium/lib/libbh_fuser_optimal.so
  children = cpu
  cache_path=/opt/bohrium/var/bohrium/fuse_cache

  #
  # Filters - Helpers / Tools
  #
  [pprint]
  type = filter
  impl = /opt/bohrium/lib/libbh_filter_pprint.so
  children = cpu

  [fuseprinter]
  type = filter
  impl = /opt/bohrium/lib/libbh_filter_fuseprinter.so
  children = pricer

  [pricer]
  type = filter
  impl = /opt/bohrium/lib/libbh_filter_pricer.so
  children = cpu

  #
  # Filters - Bytecode transformers
  #
  [bccon]
  type = filter
  impl = /opt/bohrium/lib/libbh_filter_bccon.so
  children = node
  reduction = 1

  [bcexp]
  type = filter
  impl = /opt/bohrium/lib/libbh_filter_bcexp.so
  children = node
  gc_threshold = 400
  matmul = 1
  sign = 1

  [bcexp_cpu]
  type = filter
  impl = /opt/bohrium/lib/libbh_filter_bcexp.so
  children = node
  gc_threshold = 400
  matmul = 1
  sign = 0

  #
  # Engines
  #
  [cpu]
  type = ve
  impl = /opt/bohrium/lib/libbh_ve_cpu.so
  libs = /opt/bohrium/lib/libbh_visualizer.so,/opt/bohrium/lib/libbh_fftw.so
  bind = 1
  thread_limit = 0
  vcache_size = 10
  preload = 1
  jit_level = 1
  jit_dumpsrc = 0
  compiler_cmd="/usr/bin/cc"
  compiler_inc="-I/home/safl/.local/include -I/home/safl/.local/include/bohrium -I/home/safl/.local/share/bohrium/include"
  compiler_lib="-lm"
  # Interlagos specifics
  #compiler_flg="-march=bdver1 -mprefer-avx128 -funroll-all-loops -fprefetch-loop-arrays --param prefetch-latency=300 -fno-tree-pre -ftree-vectorize "
  compiler_flg=" -O3 -fstrict-aliasing -march=native --param vect-max-version-for-alias-checks=100 -fopenmp-simd -fopenmp"
  compiler_ext="-fPIC -shared -x c -std=c99"
  object_path=/opt/bohrium/var/bohrium/objects
  template_path=/opt/bohrium/share/bohrium/templates
  kernel_path=/opt/bohrium/var/bohrium/kernels

  [gpu]
  type = ve
  impl = /opt/bohrium/lib/libbh_ve_gpu.so
  libs = /opt/bohrium/lib/libbh_ve_gpu.so
  include = /opt/bohrium/share/bohrium/include
  # Aditional options (string) given to the opencl compiler. See documentation for clBuildProgram
  #compiler_options = "-cl-opt-disable"
  work_goup_size_1dx = 128
  work_goup_size_2dx = 32
  work_goup_size_2dy = 4
  work_goup_size_3dx = 32
  work_goup_size_3dy = 2
  work_goup_size_3dz = 2
  # kernel = {[both],fixed,dynamic}
  kernel = both
  # compile = {[async],sync}
  compile = async
  children = cpu

The configuration file consists of two things: ``components`` and orchestration of components in ``stacks``.

Components marked with square brackets. For example ``[bridge]``, ``[node]`` and ``[cpu]`` are all components available for the runtime system.

Each component has a number of attributes that defines the component:

  ``type = {bridge|vem|ve|filter}`` is the type of the component.

  ``impl = {file path}`` is the path to the shared library that implement the component. Note that the bridge does not have an implementation path.

  ``children = {component}`` specifies the child component. A runtime setup always consists of three components: a bridge, a vem, and a ve.

Additionally, a component may have attributes that are specific for the component. For example the ``compiler_cmd`` attributes is only relevant for the CPU component.

The ``stacks`` define different default sane configurations of the runtime environment and one can switch between them using the environment var ``BH_STACK``.

Environment Variables
---------------------

The configuration of a component can be overwritten with environment vars using the naming convention ``BH_[COMPONENT]_[OPTION]``, below are a couple of examples controlling the behavior of the CPU vector engine::

  BH_CPU_PRELOAD      -- Preload objects into the engine, 0=Disabled, 1=Enabled.
  BH_CPU_JIT_DUMPSRC  -- Dump codegen source, 0=Disabled, 1=Enabled 
  BH_CPU_JIT_LEVEL    -- 1=Single-Instruction JIT, 2=Fusion, 3=Fusion+Contraction
  BH_VCACHE_SIZE      -- Size of victim cache in number of entries.

Experiment with values to obtain optimimal results.
