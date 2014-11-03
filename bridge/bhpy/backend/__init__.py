import os

mth = ['base', 'view', 'get_data_pointer', 'set_bhc_data_from_ary', \
       'ufunc', 'reduce', 'accumulate', 'extmethod', 'range', 'random123']

b = os.getenv('BHPY_BACKEND', "bhc")

cmd = "from .backend_%s import %s"%(b, mth[0])
for m in mth[1:]:
    cmd += ",%s"%m

exec(cmd)

