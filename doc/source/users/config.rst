Runtime Configuration
---------------------

cphVB supports a broad range of frontends and backends. 
In order to configure the runtime setup of cphVB you must provide a configuration file to cphVB. The installation of cphVB installs a default configuration file in ``/etc/cphvb/config.ini`` when doing a system-wide installation and ``~/.cphvb/config.ini`` when doing a local installation.

At runtime cphVB will search through the following prioritized list in order to find the configuration file:

* The environment variable ``CPHVB_CONFIG``
* The home directory config ``~/.cphvb/config.ini``
* The system-wide config ``/etc/cphvb/config.ini``


The default configuration file looks like this::

    [bridge]
    type = bridge
    children = node

    [node]
    impl = /home/madsbk/repos/cphvb-installed/libcphvb_vem_node.so
    children = simple
    type = vem

    [simple]
    impl = /home/madsbk/repos/cphvb-installed/libcphvb_ve_simple.so
    type = ve

    [score]
    impl = /home/madsbk/repos/cphvb-installed/libcphvb_ve_score.so
    type = ve

    [mcore]
    impl = /home/madsbk/repos/cphvb-installed/libcphvb_ve_mcore.so
    type = ve

    [gpu]
    impl = /home/madsbk/repos/cphvb-installed/libcphvb_ve_gpu.so
    type = ve
    ocldir = /home/madsbk/repos/cphvb-installed/lib/ocl_source




