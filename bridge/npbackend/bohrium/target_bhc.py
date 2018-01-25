"""
Bohrium C-backend as target for npbackend.
"""
import ctypes
import numpy
import functools
import operator
import os
import sys

from _util import dtype_name


class BhcAPI:
    """This class encapsulate the Bohrium C API
      NB: when Base.__del__() and View.__del__() is called, all other modules including 'bhc' and '_util' might already
      have been deallocated! Thus, initially we will manually load all 'bhc' functions in order to make this script
      completely self-contained.
    """

    def __init__(self):
        def get_bhc_api():
            bhc_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bhc.py")
            # We need to import '_bhc' manually into bhc_api since SWIG does know about the 'bohrium' package
            from . import _bhc
            sys.modules['_bhc'] = _bhc
            try:
                bhc_api = {"__file__": bhc_py_path, "sys": sys}
                execfile(bhc_py_path, bhc_api)  # execfile updates 'bhc_api'
                return bhc_api
            except NameError:
                import runpy
                # run_path() returns the globals() from the run
                return runpy.run_path(bhc_py_path, init_globals={"sys": sys}, run_name="__main__")

        # Load the bhc API
        for key, val in get_bhc_api().items():
            if key.startswith("bhc_"):
                setattr(self, key[4:], val)  # Save key without the "bhc_"

    def __call__(self, name, *args, **kwargs):
        """Call the API function named `name` with the `*args` and `**kwargs`"""
        func = getattr(self, name)
        return func(*args, **kwargs)

    def call_single_dtype(self, name, dtype_name, *args, **kwargs):
        """Call the API function with only a single type signature it its name."""
        return self("%s_A%s" % (name, dtype_name), *args, **kwargs)


bhc = BhcAPI()


def getDeviceContext():
    """Get the device context, such as OpenCL's cl_context, of the first VE in the runtime stack."""
    return int(bhc.getDeviceContext())


def setDeviceContext(device_context):
    """Set the device context, such as CUDA's cl_context, of the first VE in the runtime stack."""
    bhc.setDeviceContext(device_context)


def message(msg):
    """ Send and receive a message through the component stack """
    return "%s" % (bhc.message(msg))

