#!/usr/bin/env python
import json
import re

def gen( opcodes, ignore ):
    
    filtered    = [f for f in opcodes if not f['system_opcode'] and f['nop'] > 0 and f['elementwise'] and f['opcode'] not in ignore]

    for opcode in filtered:
        opcode['tparams']   = ', '.join(["typename T%d" % n for n in range(1,opcode['nop']+1)])
        opcode['fparams']   = ', '.join(["T%d *op%d" % (n, n) for n in range(1,opcode['nop']+1)])
        opcode['code']      = re.sub('op\d', '*\g<0>', opcode['code'])

    return filtered
    
