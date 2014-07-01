import os

b = os.getenv('BHPY_BACKEND', "bhc")
if b == "bhc":
    from backend_bhc import *
elif b == "numpy":
    from backend_numpy import *
elif b == "numexpr":
    from backend_numexpr import *
else:
    raise RuntimeError("Unknown backend '%s'"%b)

