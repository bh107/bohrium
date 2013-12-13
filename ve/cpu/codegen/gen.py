#!/usr/bin/env python
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

    # Map template names to mapping-functons and fill out the template
    for fn in glob.glob('templates/*.tpl'):
        fn, _ = os.path.basename(fn).split('.tpl')
        if fn in self.__dict__:
            template = Template(
                file = "%s%s%s.tpl" % ("templates", os.sep, fn),
                searchList = self.__dict__[fn](opcodes, types)
            )
            print template

def enum_to_ctypestr(opcodes, types):
    return [{"types": [(t["enum"], t["c"]) for t in types]}]

def enum_to_shorthand(opcodes, types):
    return [{"types": [(t["enum"], t["shorthand"]) for t in types]}]

def enumstr_to_ctypestr(opcodes,types):
    return [{"types": [(t["enum"], t["c"]) for t in types]}]

def enumstr_to_shorthand(opcodes, types):
    return [{"types": [(t["enum"], t["shorthand"]) for t in types]}]

def layoutmask_to_shorthand(opcodes, types):
    A0_CONSTANT = 1 << 0;
    A0_CONTIGUOUS    = 1 << 1;
    A0_STRIDED  = 1 << 2;
    A0_SPARSE   = 1 << 3;

    A1_CONSTANT = 1 << 4;
    A1_CONTIGUOUS    = 1 << 5;
    A1_STRIDED  = 1 << 6;
    A1_SPARSE   = 1 << 7;

    A2_CONSTANT = 1 << 8;
    A2_CONTIGUOUS    = 1 << 9;
    A2_STRIDED  = 1 << 10;
    A2_SPARSE   = 1 << 11;

    hej = []

    # Binary instructions
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_CONTIGUOUS)
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_CONSTANT | A1_CONTIGUOUS | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_CONTIGUOUS | A2_CONTIGUOUS)
    hej.append(A0_CONSTANT | A1_CONTIGUOUS | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_CONTIGUOUS | A2_SPARSE)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_CONTIGUOUS)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_SPARSE)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_CONTIGUOUS)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_SPARSE)
    hej.append(A0_CONTIGUOUS | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_CONTIGUOUS | A1_CONSTANT | A2_CONTIGUOUS)
    hej.append(A0_CONTIGUOUS | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_CONTIGUOUS | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_CONTIGUOUS | A1_CONTIGUOUS | A2_CONSTANT)
    hej.append(A0_CONTIGUOUS | A1_CONTIGUOUS | A2_CONTIGUOUS)
    hej.append(A0_CONTIGUOUS | A1_CONTIGUOUS | A2_STRIDED)
    hej.append(A0_CONTIGUOUS | A1_CONTIGUOUS | A2_SPARSE)
    hej.append(A0_CONTIGUOUS | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_CONTIGUOUS | A1_STRIDED | A2_CONTIGUOUS)
    hej.append(A0_CONTIGUOUS | A1_STRIDED | A2_STRIDED)
    hej.append(A0_CONTIGUOUS | A1_STRIDED | A2_SPARSE)
    hej.append(A0_CONTIGUOUS | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_CONTIGUOUS | A1_SPARSE | A2_CONTIGUOUS)
    hej.append(A0_CONTIGUOUS | A1_SPARSE | A2_STRIDED)
    hej.append(A0_CONTIGUOUS | A1_SPARSE | A2_SPARSE)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_CONTIGUOUS)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_STRIDED | A1_CONTIGUOUS | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_CONTIGUOUS | A2_CONTIGUOUS)
    hej.append(A0_STRIDED | A1_CONTIGUOUS | A2_STRIDED)
    hej.append(A0_STRIDED | A1_CONTIGUOUS | A2_SPARSE)
    hej.append(A0_STRIDED | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_STRIDED | A2_CONTIGUOUS)
    hej.append(A0_STRIDED | A1_STRIDED | A2_STRIDED)
    hej.append(A0_STRIDED | A1_STRIDED | A2_SPARSE)
    hej.append(A0_STRIDED | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_SPARSE | A2_CONTIGUOUS)
    hej.append(A0_STRIDED | A1_SPARSE | A2_STRIDED)
    hej.append(A0_STRIDED | A1_SPARSE | A2_SPARSE)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_CONTIGUOUS)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_SPARSE | A1_CONTIGUOUS | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_CONTIGUOUS | A2_CONTIGUOUS)
    hej.append(A0_SPARSE | A1_CONTIGUOUS | A2_STRIDED)
    hej.append(A0_SPARSE | A1_CONTIGUOUS | A2_SPARSE)
    hej.append(A0_SPARSE | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_STRIDED | A2_CONTIGUOUS)
    hej.append(A0_SPARSE | A1_STRIDED | A2_STRIDED)
    hej.append(A0_SPARSE | A1_STRIDED | A2_SPARSE)
    hej.append(A0_SPARSE | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_SPARSE | A2_CONTIGUOUS)
    hej.append(A0_SPARSE | A1_SPARSE | A2_STRIDED)
    hej.append(A0_SPARSE | A1_SPARSE | A2_SPARSE)

    # Unary
    hej.append(A0_CONSTANT | A1_CONSTANT)
    hej.append(A0_CONSTANT | A1_CONTIGUOUS)
    hej.append(A0_CONSTANT | A1_STRIDED)
    hej.append(A0_CONSTANT | A1_SPARSE)
    hej.append(A0_CONTIGUOUS | A1_CONSTANT)
    hej.append(A0_CONTIGUOUS | A1_CONTIGUOUS)
    hej.append(A0_CONTIGUOUS | A1_STRIDED)
    hej.append(A0_CONTIGUOUS | A1_SPARSE)
    hej.append(A0_STRIDED | A1_CONSTANT)
    hej.append(A0_STRIDED | A1_CONTIGUOUS)
    hej.append(A0_STRIDED | A1_STRIDED)
    hej.append(A0_STRIDED | A1_SPARSE)
    hej.append(A0_SPARSE | A1_CONSTANT)
    hej.append(A0_SPARSE | A1_CONTIGUOUS)
    hej.append(A0_SPARSE | A1_STRIDED)
    hej.append(A0_SPARSE | A1_SPARSE)

    hej.append(A0_CONSTANT)
    hej.append(A0_CONTIGUOUS)
    hej.append(A0_STRIDED)
    hej.append(A0_SPARSE)

    hej.sort()

    masks = []
    for bitmask in hej:
        mask = ""
        if ((bitmask & A0_CONSTANT) != 0):
            mask += "K"
        if ((bitmask & A0_CONTIGUOUS) != 0):
            mask += "C"
        if ((bitmask & A0_STRIDED) != 0):
            mask += "S"
        if ((bitmask & A0_SPARSE) != 0):
            mask += "P"

        if ((bitmask & A1_CONSTANT) != 0):
            mask += "K"
        if ((bitmask & A1_CONTIGUOUS) != 0):
            mask += "C"
        if ((bitmask & A1_STRIDED) != 0):
            mask += "S"
        if ((bitmask & A1_SPARSE) != 0):
            mask += "P"

        if ((bitmask & A2_CONSTANT) != 0):
            mask += "K"
        if ((bitmask & A2_CONTIGUOUS) != 0):
            mask += "C"
        if ((bitmask & A2_STRIDED) != 0):
            mask += "S"
        if ((bitmask & A2_SPARSE) != 0):
            mask += "P"
        masks.append((bitmask, mask))

    return [{'masks': masks}]

def cexpr_todo(opcodes, types):

    def expr(opcode):
        opcode["code"] = opcode["code"].replace("op1", "*a0_current")
        opcode["code"] = opcode["code"].replace("op2", "*a1_current")
        opcode["code"] = opcode["code"].replace("op3", "*a2_current")
        return opcode

    opcodes = [expr(opcode) for opcode in opcodes]
    
    data = {                    # Group the opcodes
        'system': [
            opcode for opcode in opcodes \
            if opcode['system_opcode'] \
            and 'USERFUNC' not in \
            opcode['opcode']
        ],
        'extensions': [
            opcode for opcode in opcodes \
            if 'USERFUNC' in opcode['opcode']
        ],
        'reductions': [
            opcode for opcode in opcodes \
            if 'REDUCE' in opcode['opcode']
        ],
        'binary': [
            opcode for opcode in opcodes \
            if opcode['nop']==3 \
            and opcode['elementwise']
        ],
        'unary': [
            opcode for opcode in opcodes \
            if opcode['nop']==2 \
            and opcode['elementwise']
        ]
    }

    return [data, {'opcodes': opcodes}]


def typesig_to_shorthand(opcodes, types):

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

def bh_typesig_check(opcodes, types):
    return typesig_to_shorthand(opcodes, types)

if __name__ == "__main__":
    main(sys.modules[__name__])
