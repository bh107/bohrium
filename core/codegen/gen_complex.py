#!/usr/bin/env python
from collections import OrderedDict
import json

def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses

defs    = json.load(open('tt.json'))
    
COMPLEX = ['CPHVB_IDENTITY','CPHVB_ADD','CPHVB_MULTIPLY','CPHVB_DIVIDE','CPHVB_SUBTRACT','CPHVB_COS','CPHVB_COSH','CPHVB_EXP','CPHVB_LOG','CPHVB_LOG10','CPHVB_NEGATIVE','CPHVB_SIN','CPHVB_SINH','CPHVB_SQRT','CPHVB_SQUARE','CPHVB_TAN','CPHVB_TANH']

ops = []                                                    # Reshape the type-signature
for op in defs:
    new = OrderedDict()                                     # Ensure a nice pretty order of the opcode-keys
    new['opcode']   = op['opcode']                          # When reading the definition it is nice to have
    new['doc']      = op['doc']                             # Opcode on top.
    new['code']     = op['code']
    new['nop']      = op['nop']
    new['system_opcode'] = op['system_opcode']
    new['types']    = op['types']

    if op['opcode'] in COMPLEX:
        op['types'].append([u'CPHVB_COMPLEX64' for i in xrange(op['nop'])])
        op['types'].append([u'CPHVB_COMPLEX128' for i in xrange(op['nop'])])

    if op['opcode'] in ['CPHVB_EQUAL','CPHVB_NOT_EQUAL']:
        op['types'].append([u'CPHVB_BOOL']+[u'CPHVB_COMPLEX64' for i in xrange(op['nop']-1)])
        op['types'].append([u'CPHVB_BOOL']+[u'CPHVB_COMPLEX128' for i in xrange(op['nop']-1)])
 
    if op['opcode'] in "CPHVB_IDENTITY":
        t = []
        for o in op['types']:
            if o[0] not in t:
                t.append(o[0])
        for tt in t:
            if tt != u'CPHVB_COMPLEX64':
                op['types'].append([u'CPHVB_COMPLEX64',tt])
            if tt != u'CPHVB_COMPLEX128':
                op['types'].append([u'CPHVB_COMPLEX128',tt])

    ops.append(new)

ops = sorted(ops, key=lambda k: (-k['nop'], k['opcode']))   # Sort by number of operands + opcode
                                                        
sops, nops = partition(lambda x: x['system_opcode'], ops)   # Partition in sys and non-sys

print json.dumps(nops+sops, indent=4)                       # Dump it out there! Non-sys on top. sys down below.


