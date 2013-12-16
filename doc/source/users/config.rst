Runtime Configuration
---------------------

Bohrium supports a broad range of frontends and backends.
In order to configure the runtime setup of Bohrium you must provide a configuration file to Bohrium. The installation of Bohrium installs a default configuration file in ``/etc/bohrium/config.ini`` when doing a system-wide installation and ``~/.bohrium/config.ini`` when doing a local installation.

At runtime Bohrium will search through the following prioritized list in order to find the configuration file:

* The environment variable ``BH_CONFIG``
* The home directory config ``~/.bohrium/config.ini`` (Windows: %APPDATA%\bohrium\config.ini)
* The system-wide config ``/etc/bohrium/config.ini`` (Windows: %PROGRAMFILES%\bohrium\config.ini)


The default configuration file looks like this::

    [bridge]
    type = bridge
    children = node

    [node]
    type = vem
    impl = /opt/bohrium/libbh_vem_node.so
    children = gpu

    [gpu]
    type = ve
    ocldir = /opt/bohrium/lib/ocl_source
    impl = /opt/bohrium/libbh_ve_gpu.so
    children = cpu

    [cpu]
    type = ve
    compiler_cmd="gcc -I/opt/bohrium/cpu/include -lm -O3 -march=native -fopenmp -fPIC -std=c99 -x c -shared - -o "
    object_path=/opt/bohrium/cpu/objects
    kernel_path=/opt/bohrium/cpu/kernels
    template_path=/opt/bohrium/cpu/templates
    impl = /opt/bohrium/libbh_ve_cpu.so
    libs = /opt/bohrium/libbh_matmul.so

The configuration file consists of a number of components marked with square brackets. For example ``[bridge]``, ``[node]`` and ``[cpu]`` are all components available for the runtime system.

Each component has a number of attributes that defines the component:

  ``type = {bridge|vem|ve|filter}`` is the type of the component.

  ``impl = {file path}`` is the path to the shared library that implement the component. Note that the bridge does not have an implementation path.

  ``children = {component}`` specifies the child component. A runtime setup always consists of three components: a bridge, a vem, and a ve.

Additionally, a component may have attributes that are specific for the component. For example the ``compiler_cmd`` attributes is only relevant for the CPU component.

Environment Variables
---------------------

The various engines can be manipulated by environment variables::

  BH_VE_SCORE_BLOCKSIZE - Adjusts size of cache-tiling.
  BH_VE_MCORE_BLOCKSIZE - Adjusts size of work-splits and cache-tiling.
  BH_VE_MCORE_NTHREADS  - Adjusts the number of threads used.

Experiment with values to obtain optimimal results.
