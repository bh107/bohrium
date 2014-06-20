from array_create import *
from array_manipulation import *
from ufunc import ufuncs
from ndarray import check, check_biclass, fix_biclass
from _info import numpy_types
from _util import flush
import linalg
from linalg import matmul, dot
from summations import sum
import import_external

#Expose all ufuncs
for f in ufuncs:
    exec "%s = f"%f.info['name']

#Expose all data types
for t in numpy_types:
    exec "%s = numpy.%s"%(t.__str__(),t.__str__())

#Note that the following modules needs ufuncs and dtypes
import random

#TODO: import all numpy functions
from numpy import meshgrid

#Finally, we import external libraries
numpy_interface = [\
"numpy.lib.stride_tricks.as_strided",
"numpy.newaxis",
"numpy.pi",
"numpy.transpose"]

for i in import_external.api(numpy_interface):
    exec i
