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
    impl = /opt/bohrium/libbh_vem_node.so
    children = simple
    type = vem

    [simple]
    impl = /opt/bohrium/libbh_ve_simple.so
    type = ve

    [score]
    impl = /opt/bohrium/libbh_ve_score.so
    type = ve

    [mcore]
    impl = /opt/bohrium/libbh_ve_mcore.so
    type = ve

    [gpu]
    impl = /opt/bohrium/libbh_ve_gpu.so
    type = ve
    ocldir = /opt/bohrium/lib/ocl_source


The configuration file consists of a number of components marked with square brackets. For example ``[bridge]``, ``[node]`` and ``[simple]`` are all components available for the runtime system. 

Each component has a number of attributes that defines the component:

  ``type = {bridge|vem|ve}`` is the type of the component.

  ``impl = {file path}`` is the path to the shared library that implement the component. Note that the bridge does not have an implementation path.

  ``children = {component}`` specifies the child component. A runtime setup always consists of three components: a bridge, a vem and a ve. The child of the bridge is a vem and the child of the vem is a ve.

Additionally, a component may have attributes that are specific for the component. For example the ``ocldir`` attributes is only relevant for the gpu component. 

Environment Variables
---------------------

The various engines can be manipulated by environment variables::

  BH_VE_SCORE_BLOCKSIZE - Adjusts size of cache-tiling.
  BH_VE_MCORE_BLOCKSIZE - Adjusts size of work-splits and cache-tiling.
  BH_VE_MCORE_NTHREADS - Adjusts the number of threads used.

Experiment with values to obtain optimimal results.
