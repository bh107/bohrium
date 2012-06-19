#!/usr/bin/env python
import re
import json
from Cheetah.Template import Template

def gen_traverse(defs):

    def tfunc(ops):
        t = {
            'ops': ops
        }
        return t

    funcs = [
        [(0, 'a'),(1,'a'),(2, 'a')],
        [(0, 'a'),(1,'a'),(2, 'c')],
        [(0, 'a'),(1,'c'),(2, 'a')],
        [(0, 'a'),(1,'a')],
        [(0, 'a'),(1,'c')],
    ]

    t = []
    for f in funcs:
        template_sig = ', '.join(['typename T%d'%op[0] for op in f])
        func_call = []
        for op in f:
            if op[1] == 'a':
                func_call.append('(off%d+d%d)' % (op[0], op[0]))
            else:
                func_call.append('d%d' % op[0])
        t.append({
            'sig': ''.join([op[1] for op in f]),
            'tsig': template_sig,
            'ops': f,
            'func_call': ', '.join(func_call)
        })

    f_tmpl  = Template(file='traverse.ctpl', searchList=[{'traversers': t}])
    return f_tmpl

def gen_functors():
    pass

def gen_cases(defs, ignore):

    reshape = [defs[opcode] for opcode in defs]
    types   = [(f, out_type, in_type) for f in reshape for out_type in f['types'] for in_type in f['types'][out_type]]
    filter_ignore   = (f for f in reshape if f['opcode'] not in ignore)
    filter_system   = (f for f in filter_ignore if not f['system_opcode'] )
    expand          = (dict(f.items, {'op1': o, 'op2': i, 'op3': i}) for f in filter_system for f['types']

    return list(filter_system)

def main():
    ignore  = json.load(open('ignore.json'))
    defs    = json.loads(re.sub('//.*?\n|/\*.*?\*/', '', open('../../codegen/opcodes.json').read(), re.S, re.DOTALL | re.MULTILINE))
    data = gen_cases( defs, ignore )
    t_tmpl = Template(file='cphvb_compute.ctpl', searchList=[{'cases': data}])
    print t_tmpl

if __name__ == "__main__":
    main()
