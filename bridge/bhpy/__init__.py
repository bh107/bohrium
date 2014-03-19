from array_create import *
from ufunc import ufuncs
from ndarray import check

#Expose all ufuncs
for f in ufuncs:
    exec "%s = f"%f.info['np_name']
