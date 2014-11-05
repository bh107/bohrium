import os

mth = ['Base', 'View', 'get_data_pointer', 'set_bhc_data_from_ary', \
       'ufunc', 'reduce', 'accumulate', 'extmethod', 'range', 'random123']

b = os.getenv('BHPY_BACKEND')
if not b:
    b = os.getenv('BHPY_TARGET', "bhc")

cmd = "from .target_%s import %s"%(b, mth[0])
for m in mth[1:]:
    cmd += ",%s"%m

exec(cmd)

