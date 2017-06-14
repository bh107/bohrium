"""
Bohrium Contexts
================
"""
import sys
from . import backend_messaging as messaging


class BohriumContext:
    """Enable Bohrium within the context"""
    def __init__(self):
        self.__numpy        = sys.modules['numpy']
        self.__numpy_random = sys.modules['numpy.random']
        self.__numpy_linalg = sys.modules['numpy.linalg']

        # Sub-module matlib has to be imported explicitly once in order to be available through bohrium
        try:
            import numpy.matlib
        except ImportError:
            pass

    def __enter__(self):
        import numpy
        import bohrium
        # Overwrite with Bohrium
        sys.modules['numpy_force']  = numpy
        sys.modules['numpy']        = bohrium
        sys.modules['numpy.random'] = bohrium.random
        sys.modules['numpy.linalg'] = bohrium.linalg

    def __exit__(self, *args):
        # Put NumPy back together
        sys.modules.pop('numpy_force', None)
        sys.modules['numpy']        = self.__numpy
        sys.modules['numpy.random'] = self.__numpy_random
        sys.modules['numpy.linalg'] = self.__numpy_linalg


class Profiling:
    """Profiling the Bohrium backends within the context."""
    def __init__(self):
        pass

    def __enter__(self):
        messaging.statistic_enable_and_reset()

    def __exit__(self, *args):
        print(messaging.statistic())
