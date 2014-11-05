import imp
import sys
import inspect
from importlib import import_module
from . import ndarray
import numpy

def bohriumify(obj):
    """Returns a bohrium compatible version of 'obj'"""
    if inspect.isroutine(obj):
        return ndarray.fix_returned_biclass(obj)
    return obj

def new_module(fullname):
    m = imp.new_module(fullname)
    m.__file__ = "<clone>"
    sys.modules[fullname] = m
    return m

def api(objs):
    """Imports all objects in the list 'objs'.
    Returns a list of import statements, e.g. ["import bohrium.linalg"]
    """
    objs.sort()#sort objects such that 'numpy' is before 'numpy.linalg'
    ret = []
    for name in objs:
        body = name.split(".")
        head_name = body.pop(-1)
        try:
            if len(body) > 0:
                head_obj = getattr(import_module('.'.join(body)), head_name)
            else:
                head_obj = import_module(head_name)
        except ImportError:
            continue

        if len(body) > 0:
            body[0] = "bohrium"
            #Lets create all modules in 'body'
            for i in xrange(len(body)):
                prefix = body[:i]
                fullname = '.'.join(body[:i+1])
                if fullname not in sys.modules:
                    tmp = new_module(fullname)
                    if len(prefix) > 0:
                        m = import_module('.'.join(prefix))
                        setattr(m, body[i], tmp)

        fullname = '.'.join(body + [head_name])
        prefix = '.'.join(body)

        if inspect.ismodule(head_obj):
            if len(body) > 0:
                sys_name = 'bohrium.' + head_obj.__name__.split(".", 1)[1]
            else:
                sys_name = 'bohrium'

            if sys_name not in sys.modules:
                tmp = new_module(fullname)
                if len(prefix) > 0:
                    m = import_module(prefix)
                    setattr(m, head_name, tmp)
            try:
                m = import_module(sys_name)
            except ImportError:
                continue

            for o in dir(head_obj):
                if o not in m.__dict__:
                    setattr(m, o, bohriumify(getattr(head_obj, o)))
            ret.append("import %s"%fullname)
        else:
            m = import_module(prefix)
            if head_name not in m.__dict__:
                setattr(m, head_name, bohriumify(head_obj))
            ret.append("import %s"%prefix)
    return ret

def all_numpy(module=numpy, prefix="numpy", added=set()):
    """Returns a list of import statements that include all of NumPy"""
    added.add(module)
    ret = ["numpy"]
    for k, o in inspect.getmembers(module):
        if inspect.ismodule(o) and o not in added and o.__name__.startswith("numpy") \
                and not k.startswith("_") and not "random" in k:
            t = prefix + "." + k
            ret.append(t)
            ret.extend(all_numpy(o, t, added))
    return ret

