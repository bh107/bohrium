#!/usr/bin/env python
import subprocess
import glob
import os
import re

regex   = ".*(BH_.*_[\dN]D)_?.*"
root    = [os.sep, 'home', 'safl', '.local', 'cpu']

funcs = {'kernels': [], 'objects': [], 'announced': [], 'available': [],
         'flattened':[]}

for kernel in glob.glob(os.sep.join(root+['kernels','BH_*.c'])):
    m = re.match(regex, kernel)
    if m:
        funcs['kernels'].append(m.group(1))

for obj in glob.glob(os.sep.join(root+['objects','BH_*.so'])):
    m = re.match(regex, obj)
    if m:
        funcs['objects'].append(m.group(1))

for line in open(os.sep.join(root+['objects','bh_libsij_aaaaaa.idx'])).readlines():
    m = re.match(regex, line)
    if m:
        funcs['announced'].append(m.group(1))

for line in open(os.sep.join([os.sep, 'tmp', 'spec.txt'])).readlines():
    funcs['flattened'].append(line.strip()+'_ND')

p = subprocess.Popen(
    ['nm', '-D']+[os.sep.join(root+['objects','bh_libsij_aaaaaa.so'])],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
out, err = p.communicate()

for line in out.split("\n"):
    m = re.match(regex, line)
    if m:
        funcs['available'].append(m.group(1))

for source in funcs:
    print source, len(funcs[source])
    funcs[source].sort()
    with open(os.sep.join([os.sep, 'tmp', source+'.txt']), 'w') as fd:
        for line in funcs[source]:
            fd.write(line+"\n")

