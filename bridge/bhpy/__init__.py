from array_create import *
from array_manipulation import *
from ufunc import ufuncs
from ndarray import check
from _info import numpy_types
from _util import flush
import linalg
from linalg import matmul
from numpy import newaxis

#Expose all ufuncs
for f in ufuncs:
    exec "%s = f"%f.info['np_name']

#Expose all data types
for t in numpy_types:
    exec "%s = t"%t.__str__()

#Note that the following modules needs ufuncs and dtypes
import random

#TODO: import all numpy functions
from numpy import meshgrid
