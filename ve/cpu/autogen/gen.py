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

def forward_everything(opcodes, ops, opers, types, layouts, cexprs):
    """This is for those functions that do not need some sort of mangling, they just want it all."""

    return {
        'opcodes':  opcodes,
        'ops':      ops,
        'opers':    opers,
        'types':    types,
        'layouts':  layouts,
        'cexprs':    cexprs
    }

def utils_mapping(opcodes, ops, opers, types, layouts, cexprs):
    return forward_everything(opcodes, ops, opers, types, layouts, cexprs)

def tac(opcodes, ops, opers, types, layouts, cexprs):
    return forward_everything(opcodes, ops, opers, types, layouts, cexprs)

def block_compose(opcodes, ops, opers, types, layouts, cexprs):
    """Construct the data need to create a map from bh_instruction to tac_t."""

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
            system.append([opcode, 'SYSTEM', opcode.replace('BH_',''), 0])

        else:
            if 'REDUCE' in opcode:
                operator = '_'.join(opcode.split('_')[1:-1])
                reductions.append([opcode, 'REDUCE', operator, 2])
            elif 'ACCUMULATE' in opcode:
                operator = '_'.join(opcode.split('_')[1:-1])
                scans.append([opcode, 'SCAN', operator, 2])
            elif 'RANDOM' in opcode:
                generators.append([opcode, 'GENERATE', 'RANDOM', 1])
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


def specializer_cexpression(opcodes, ops, opers, types, layouts, cexprs):
    """Apply a naming convention to the pseud-variables, and make it string-formattable."""

    def naming_convention(expression):
        """
        expression  = expression.replace('out', '*a%1$d_current')
        expression  = expression.replace('in1', '*a%2$d_current')
        expression  = expression.replace('in2', '*a%3$d_current')
        """
        expression  = expression.replace('out', '%1$ca%2$d_current')
        expression  = expression.replace('in1', '%3$ca%4$d_current')
        expression  = expression.replace('in2', '%5$ca%6$d_current')
        return expression

    expressions = {}
    for cexpr in cexprs:
        op      = cexpr['op']
        oper    = cexpr['oper']

        # Grab the expressing and apply the naming convention
        oper_expr = [                                   
            (case, naming_convention(expr))             
            for case, expr in cexpr['scalar'].items()  
            if expr
        ]

        if oper in expressions:
            expressions[oper].append((op, oper_expr))
        else:
            expressions[oper] = [(op, oper_expr)]

    expr_list = []
    for oper in expressions:
        for op, oper_expr in expressions[oper]:
            oper_expr.sort()
        expr_list.append((oper, expressions[oper]))

    expr_list.sort()

    return {'expressions': expr_list}

def operator_cexpression(opcodes, ops, opers, types, layouts, cexprs):
    """Apply a naming convention to the pseud-variables, and make it string-formattable."""

    return specializer_cexpression(opcodes, ops, opers, types, layouts, cexprs)

def main(self):

    root    = "../../../core/codegen"
    prefix  = "./tac/"

    opcodes = json.load(open("%s%s%s.json" % (root,   os.sep, 'opcodes')))
    ops     = json.load(open("%s%s%s.json" % (prefix, os.sep, 'operations')))
    opers   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'operators')))
    types   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'types')))
    layouts = json.load(open("%s%s%s.json" % (prefix, os.sep, 'layouts')))
    cexprs  = json.load(open("%s%s%s.json" % (prefix, os.sep, 'cexpressions')))

    # Map template names to mapping-functons and fill out the template
    for fn in glob.glob('templates/*.tpl'):
        fn, _ = os.path.basename(fn).split('.tpl')
        if fn in self.__dict__:
            template = BohriumTemplate(
                file = "%s%s%s.tpl" % ("templates", os.sep, fn),
                searchList=globals()[fn](opcodes, ops, opers, types, layouts, cexprs)
            )
            with open('output/%s.cpp' % fn, 'w') as fd:
                fd.write(str(template))

if __name__ == "__main__":
    main(sys.modules[__name__])
