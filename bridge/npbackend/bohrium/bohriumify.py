import sys
import re
import inspect
import importlib
from . import bhary


def _target_modules(targets):
    mods = []

    for m in sys.modules:
        for target in targets:
            if m.startswith(target):
                mods.append(m)
    ret = []

    for m in set(mods):
        ret.append(m.replace("numpy.", "numpy_force."))

    return ret


# We will ignore the numpy.core module
IGNORE_MODULES = '[numpy|numpy_force]\.core'

pattern_return_ndarray = re.compile("Return.*ndarray", re.DOTALL)


def modules(targets=["numpy", "numpy_force"], regex_to_exclude=IGNORE_MODULES):
    for m_name in _target_modules(targets):
        if m_name == "numpy":
            # At this point 'numpy' refers to Bohrium and we don't need to bohriumfy Bohrium
            continue

        if re.search(regex_to_exclude, m_name) is not None:
            continue

        try:
            m_obj = importlib.import_module(m_name)
        except:
            continue

        for name, val in inspect.getmembers(m_obj, predicate=inspect.isroutine):
            if not hasattr(val, "_wrapped_fix_biclass"):
                if hasattr(val, "__doc__") and val.__doc__ is not None:
                    if pattern_return_ndarray.search(val.__doc__) is not None:
                        setattr(m_obj, name, bhary.fix_biclass_wrapper(val))
                        # The function should have its original name and not `inner`
                        getattr(m_obj, name).__name__ = name
