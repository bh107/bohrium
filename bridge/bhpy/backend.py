import os

b = os.getenv('BHPY_BACKEND', "bhc")
if b == "bhc":
    from backend_bhc import *
if b == "numpy":
    from backend_numpy import *
