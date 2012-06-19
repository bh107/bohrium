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

defs    = json.load(open('opcodes_dict.json'))
reshape = (defs[opcode] for opcode in defs)                 # Remove the dict-index

ops = []                                                    # Reshape the type-signature
for op in reshape:
    typesig = [ [out_t]+ [in_t] * (op['nop']-1) for in_t in op['types'] for out_t in op['types'][in_t] ]
    new = OrderedDict()                                     # Ensure a nice pretty order of the opcode-keys
    new['opcode']   = op['opcode']                          # When reading the definition it is nice to have
    new['doc']      = op['doc']                             # Opcode on top.
    new['code']     = op['code']
    new['nop']      = op['nop']
    new['system_opcode'] = op['system_opcode']
    new['types'] = typesig
    ops.append(new)

ops = sorted(ops, key=lambda k: (-k['nop'], k['opcode']))   # Sort by number of operands + opcode
                                                        
sops, nops = partition(lambda x: x['system_opcode'], ops)   # Partition in sys and non-sys

print json.dumps(nops+sops, indent=4)                       # Dump it out there! Non-sys on top. sys down below.
