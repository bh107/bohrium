# -*- coding: utf-8 -*-
"""
===============================
Bohrium API: a Python and C API
===============================

"""
import os
from ._bh_api import _C_API  # Exposing the C_API of `_bh_api.c`
from ._info import __version__
from . import stack_info


def _pip_specific_config():
    """Handle pip specific configurations"""

    # Do nothing if we a not a pip package or the user has manually specified a config file
    if not stack_info.installed_through_pypi() or "BH_CONFIG" in os.environ:
        return

    os.environ["BH_CONFIG"] = stack_info.config_file_path()
    # On OSX, we use the `gcc7`, which contains a complete GCC installation
    import platform
    if platform.system() == "Darwin" and "BH_OPENMP_COMPILER_CMD" not in os.environ:
        import gcc7
        # We manually sets the GCC compile command
        cmd = gcc7.path.gcc()
        cmd += ' -x c -fPIC -shared -std=gnu99 -O3 -march=native -arch x86_64 -Werror'
        cmd += ' -Wl,-rpath,%s -fopenmp' % gcc7.path.lib()
        cmd += ' -I{CONF_PATH}/include -lm {IN} -o {OUT}'
        os.environ["BH_OPENMP_COMPILER_CMD"] = cmd
        # Finally, we active the OpenMP code generation but deactivate OpenMP-simd, which doesn't work on mac
        if "BH_OPENMP_COMPILER_OPENMP" not in os.environ:
            os.environ['BH_OPENMP_COMPILER_OPENMP'] = "true"
        if "BH_OPENMP_COMPILER_OPENMP_SIMD" not in os.environ:
            os.environ['BH_OPENMP_COMPILER_OPENMP_SIMD'] = "false"


_pip_specific_config()


def get_include():
    """Return the directory that contains the Bohrium-API *.h header files.

    Extension modules that need to compile against Bohrium-API should use this
    function to locate the appropriate include directory.

    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::

        import bohrium_api
        ...
        Extension('extension_name', ...
                include_dirs=[bohrium_api.get_include()])
        ...
    """
    return stack_info.header_dir()

# print(stack_info.pprint())
