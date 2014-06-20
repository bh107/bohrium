import imp
import sys
import inspect
from importlib import import_module

def api(objs):
    """Imports all objects in the list 'objs'.
    Returns a list of import statements, e.g. ["import linalg"]
    """
    ret = []
    for name in objs:
        body = name.split(".")
        head_name = body.pop(-1)
        head_obj = getattr(import_module('.'.join(body)), head_name)
        body[0] = "bohrium"
        #Lets create all modules in 'body'
        for i in xrange(len(body)):
            prefix = body[:i]
            fullname = '.'.join(body[:i+1])
            if fullname not in sys.modules:
                tmp = imp.new_module(fullname)
                sys.modules[fullname] = tmp
                if len(prefix) > 0:
                    m = import_module('.'.join(prefix))
                    setattr(m, body[i], tmp)

        fullname = '.'.join(body + [head_name])
        prefix = '.'.join(body)

        if inspect.ismodule(head_obj):
            if fullname not in sys.modules:
                tmp = imp.new_module(fullname)
                sys.modules[fullname] = tmp
                if len(prefix) > 0:
                    m = import_module(prefix)
                    setattr(m, head_name, tmp)
            m = import_module(fullname)
            for o in dir(head_obj):
                if o not in m.__dict__:
                    setattr(m, o, getattr(head_obj,o))
            ret.append("import %s"%fullname)
        else:
            m = import_module(prefix)
            setattr(m, head_name, head_obj)
            ret.append("import %s"%prefix)
    return ret

