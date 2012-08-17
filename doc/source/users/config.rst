Runtime Configuration
---------------------

cphVB supports a broad range of frontends and backends. 
In order to configure the runtime setup of cphVB you must provide a configuration file to cphVB. The installation of cphVB installs a default configuration file in ``/etc/cphvb/config.ini`` when doing a system-wide installation and ``~/.cphvb/config.ini`` when doing a local installation.

At runtime cphVB will search through the following prioritized list in order to find the configuration file:

* The environment variable ``CPHVB_CONFIG``
* The home directory config ``~/.cphvb/config.ini`` (Windows: %APPDATA%\cphvb\config.ini)
* The system-wide config ``/etc/cphvb/config.ini`` (Windows: %PROGRAMFILES%\cphvb\config.ini)


The default configuration file looks like this::

    [bridge]
    type = bridge
    children = node

    [node]
    impl = /opt/cphvb/libcphvb_vem_node.so
    children = simple
    type = vem

    [simple]
    impl = /opt/cphvb/libcphvb_ve_simple.so
    type = ve

    [score]
    impl = /opt/cphvb/libcphvb_ve_score.so
    type = ve

    [mcore]
    impl = /opt/cphvb/libcphvb_ve_mcore.so
    type = ve

    [gpu]
    impl = /opt/cphvb/libcphvb_ve_gpu.so
    type = ve
    ocldir = /opt/cphvb/lib/ocl_source
