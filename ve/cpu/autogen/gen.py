#!/usr/bin/env python
from itertools import product
from pprint import pprint
import glob
import json
import sys
import os

from Cheetah.Template import Template

class BohriumTemplate(Template):
    """This class adds a couple of functions which can be used within the template."""

    def addw(text, nw=10):
        w = ' '*nw
        return text+w[len(text):] if nw > len(text) else text
    
    def addsep(element, elements):
        cur_id  = element['id']
        last_id = elements[-1]['id']

        return '' if cur_id == last_id else ','

def forward_everything(opcodes, ops, opers, types, layouts):
    """This is for those functions that do not need some sort of mangling, they just want it all."""

    return {
        'opcodes':  opcodes,
        'ops':      ops,
        'opers':    opers,
        'types':    types,
        'layouts':  layouts
    }

def utils_mapping(opcodes, ops, opers, types, layouts):
    return forward_everything(opcodes, ops, opers, types, layouts)

def tac(opcodes, ops, opers, types, layouts):
    return forward_everything(opcodes, ops, opers, types, layouts)

def acc(opcodes, ops, opers, types, layouts):
    acc_etypes = [
        (typedef['c'], typedef['name'])
        for typedef
        in types 
        if typedef['name'] not in ["KP_COMPLEX128", "KP_COMPLEX64", "KP_PAIRLL"] 
    ]
    return {"ETYPES": acc_etypes}

def instrs_to_tacs(opcodes, ops, opers, types, layouts):
    """Construct the data need to create a map from bh_instruction to tac_t."""

    ewise_u     = []
    ewise_b     = []
    index       = []
    scans       = []
    reductions  = []
    generators  = []
    system      = []

    huh = []
    for o in opcodes:
        opcode = o['opcode']
        if o["composite"]:
            tac_name = opcode.replace("BH_", "KP_")
            if not [tac for tac in opers if tac['name'] == tac_name]:
                print "Non-specialized composite: %s" % opcode
                continue

        if o['system_opcode']:
            system.append([opcode, 'KP_SYSTEM', opcode.replace('BH_','KP_'), 0])

        else:
            operator = opcode.replace("BH_", "KP_")
            if 'REDUCE' in opcode:
                operator = '_'.join(operator.split('_')[:-1])
                reductions.append([opcode, 'KP_REDUCE_PARTIAL', operator, 2])
            elif 'ACCUMULATE' in opcode:
                operator = '_'.join(operator.split('_')[:-1])
                scans.append([opcode, 'KP_SCAN', operator, 2])
            elif 'RANDOM' in opcode:
                generators.append([opcode, 'KP_GENERATE', 'KP_RANDOM', 2])
            elif 'RANGE' in opcode:
                generators.append([opcode, 'KP_GENERATE', 'KP_RANGE', 0])
            elif 'GATHER' in opcode or 'KP_SCATTER' in opcode:
                index.append([opcode, 'KP_INDEX', operator, 3])
            else:
                if o['nop'] == 3:
                    ewise_b.append([opcode, 'KP_ZIP', operator, 2])
                elif o['nop'] == 2:
                    ewise_u.append([opcode, 'KP_MAP', operator, 1])
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
        sorted(index)           +\
        sorted(system)
    }]

    return operations

def main(self):

    root    = "../../../core/codegen"
    prefix  = "./tac/"

    opcodes = json.load(open("%s%s%s.json" % (root,   os.sep, 'opcodes')))
    ops     = json.load(open("%s%s%s.json" % (prefix, os.sep, 'operations')))
    opers   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'operators')))
    types   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'types')))
    layouts = json.load(open("%s%s%s.json" % (prefix, os.sep, 'layouts')))

    # Map template names to mapping-functons and fill out the template
    for fn in glob.glob('templates/*.tpl'):
        fn, _ = os.path.basename(fn).split('.tpl')
        if fn in self.__dict__:
            template = BohriumTemplate(
                file = "%s%s%s.tpl" % ("templates", os.sep, fn),
                searchList=globals()[fn](opcodes, ops, opers, types, layouts)
            )
            with open('output/%s.cpp' % fn, 'w') as fd:
                fd.write(str(template))

if __name__ == "__main__":
    main(sys.modules[__name__])
