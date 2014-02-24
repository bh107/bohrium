#!/usr/bin/env python
from itertools import product
from pprint import pprint
import glob
import json
import sys
import os

from Cheetah.Template import Template

def main(self):

    prefix  = "../../../core/codegen"
    types   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'types')))
    opcodes = json.load(open("%s%s%s.json" % (prefix, os.sep, 'opcodes')))
    opers   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'operators')))

    # Map template names to mapping-functons and fill out the template
    for fn in glob.glob('templates/*.tpl'):
        fn, _ = os.path.basename(fn).split('.tpl')
        if fn in self.__dict__:
            template = Template(
                file = "%s%s%s.tpl" % ("templates", os.sep, fn),
                searchList=globals()[fn](opcodes, types, opers)
            )
            with open('output/%s.c' % fn, 'w') as fd:
                fd.write(str(template))

def bh_opcode_to_cstr(opcodes, types, opers):
    return [{"opcodes": [(o["opcode"], o["opcode"], o["opcode"].replace('BH_','')) for o in opcodes
    ]}]

def bh_opcode_to_cstr_short(opcodes, types, opers):
    return bh_opcode_to_cstr(opcodes, types, opers)

def enum_to_ctypestr(opcodes, types, opers):
    return [{"types": [(t["enum"], t["c"]) for t in types]}]

def enum_to_shorthand(opcodes, types, opers):
    return [{"types": [(t["enum"], t["shorthand"]) for t in types]}]

def enumstr_to_ctypestr(opcodes,types, opers):
    return [{"types": [(t["enum"], t["c"]) for t in types]}]

def enumstr_to_shorthand(opcodes, types, opers):
    return [{"types": [(t["enum"], t["shorthand"]) for t in types]}]

def operators(opcodes, types, opers):
    unary   = []
    binary  = []
    huh     = []
    system  = []
    userdef = ['USERDEFINED']
    generators = ["RANGE", "RANDOM", "FLOOD"]

    for opc in opcodes:
        opcode = opc['opcode'].replace('BH_','')
        if 'REDUCE' in opcode:
            continue
        if 'ACCUMULATE' in opcode:
            continue
        if 'RANDOM' in opcode:
            continue
        if 'RANGE' in opcode:
            continue

        if opc['system_opcode']:
            system.append(opcode)
        else:
            if opc['nop'] == 3:
                binary.append(opcode)
            elif opc['nop'] == 2:
                unary.append(opcode)
            else:
                huh.append(opcode)
    
    if len(huh)>0:
        print "Something is weird here!", huh

    operators = [{
        'unary':        sorted(unary),
        'binary':       sorted(binary),
        'system':       sorted(system),
        'generators':   sorted(generators)
    }]

    return operators

def operator_cexpr(opcodes, types, opers):
    operators = []
    for op in opers:
        operators.append((op, opers[op]['code']))

    return {'operators': operators}

def block(opcodes, types, opers):
    """Construct the data need to create a map from bh_instruction to bh_bytecode_t."""

    ewise_u     = []
    ewise_b     = []    
    scans       = []
    reductions  = []
    generators  = []
    system      = []

    huh = []

    for o in opcodes:
        opcode = o['opcode']

        if o['system_opcode']:
            nin = 1
            if 'BH_FREE' in opcode:
                nin = 0
            system.append([opcode, 'SYSTEM', opcode.replace('BH_',''), nin])

        else:
            if 'REDUCE' in opcode:
                operator = '_'.join(opcode.split('_')[1:-1])
                reductions.append([opcode, 'REDUCE', operator, 2])
            elif 'ACCUMULATE' in opcode:
                operator = '_'.join(opcode.split('_')[1:-1])
                scans.append([opcode, 'SCAN', operator, 2])
            elif 'RANDOM' in opcode:
                generators.append([opcode, 'GENERATE', 'RANDOM', 2])
            elif 'RANGE' in opcode:
                generators.append([opcode, 'GENERATE', 'RANGE', 0])
            else:
                operator = '_'.join(opcode.split('_')[1:])
                if o['nop'] == 3:
                    ewise_b.append([opcode, 'ZIP', operator, 2])
                elif o['nop'] == 2:
                    ewise_u.append([opcode, 'MAP', operator, 1])
                else:
                    huh.append([opcode, '?', operator, 0])
    
    if len(huh)>0:
        print "Something is weird here!", huh

    operations = [{'operations': \
        sorted(ewise_u)         +\
        sorted(ewise_b)         +\
        sorted(reductions)      +\
        sorted(scans)           +\
        sorted(generators)      +\
        sorted(system)
    }]

    return operations

def layoutmask_shorthand(opcodes, types, opers):

    array_l     = ["CONTIGUOUS", "STRIDED", "SPARSE"]
    scalar_l    = ["CONSTANT"]

    mapping = {"CONTIGUOUS": "C", "STRIDED": "S", "SPARSE": "P", "CONSTANT": "K"}

    def shorten(mask):
        right = ''.join([mapping[layout] for layout in mask])
        
        return ('LMASK_'+right, right)

    lmasks = {
        3: [(shorten(x)[0], x) for x in product(array_l, scalar_l+array_l, scalar_l+array_l)],
        2: [(shorten(x)[0], x) for x in product(array_l, scalar_l+array_l)],
        1: [(shorten(x)[0], x) for x in product(array_l)],
    }

    
    lsh =   [shorten(x) for x in product(array_l, scalar_l+array_l, scalar_l+array_l)]   +\
            [shorten(x) for x in product(array_l, scalar_l+array_l)]                     +\
            [shorten(x) for x in product(array_l)]
    
    return {'lmasks': lmasks, 'lshort': lsh}

def typesig_to_shorthand(opcodes, types, opers):

    etu = dict([(t["enum"], t["id"]+1) for t in types])
    ets = dict([(t["enum"], t["shorthand"]) for t in types])

    typesigs = {
        3: [],
        2: [],
        1: [],
        0: []
    }

    for opcode in opcodes:
        for typesig in opcode['types']:
            slen = len(typesig)
            if typesig not in typesigs[slen]:
                typesigs[slen].append(typesig)

    nsigs = []
    tsigs = []
    hsigs = []
    cases = []
    p_slen = -1
    for slen, typesig in ((slen, typesig) for slen in xrange(3,-1,-1) for typesig in typesigs[slen]):

        if slen == 3:
            tsig = "%s + (%s << 4) + (%s << 8)" % tuple(typesig)
            nsig = etu[typesig[0]] + (etu[typesig[1]]<<4) + (etu[typesig[2]]<<8)
            hsig = ets[typesig[0]] + (ets[typesig[1]]) + (ets[typesig[2]])
        elif slen == 2:
            tsig = "%s + (%s << 4)" % tuple(typesig)
            nsig = etu[typesig[0]] + (etu[typesig[1]]<<4)
            hsig = ets[typesig[0]] + (ets[typesig[1]])
        elif slen == 1:
            tsig = "%s" % tuple(typesig)
            nsig = etu[typesig[0]]
            hsig = ets[typesig[0]]
        elif slen == 0:
            tsig = "0"
            nsig = 0
            hsig = "_"
        tsigs.append(tsig)
        nsigs.append(nsig)
        hsigs.append(hsig)

        if slen != p_slen:
            p_slen = slen
        cases.append((nsig, hsig, tsig))

    return [{"cases": cases}]

def bh_typesig_check(opcodes, types, opers):
    return typesig_to_shorthand(opcodes, types, opers)

if __name__ == "__main__":
    main(sys.modules[__name__])
