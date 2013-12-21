#!/usr/bin/env python
import json
import os
from os.path import join, exists
from pprint import pprint
from Cheetah.Template import Template
import stat

def render( gens, tmpl_dir, output_dir, mtime ):

    prev_output_fn   = gens[0][1]
    prev_output      = ""
    count = (len(gens)-1)
    for c, (tmpl_fn, output_fn, data) in enumerate(gens):   # Concat the rendered template into output_fn
        t_tmpl  = Template(file= "%s%s" % (tmpl_dir, tmpl_fn), searchList=[{
            'data': data,
            'tmpl_dir': tmpl_dir
        }])
        last = count == c

        if (output_fn != prev_output_fn ):
            with open(output_dir + prev_output_fn, 'w') as f:
                f.write( str(prev_output) )
                f.close()
                set_timestamp(f.name, (mtime,mtime))
            prev_output = ""

        prev_output += str(t_tmpl)

        if last:
            with open(output_dir + output_fn, 'w') as f:
                f.write(str(prev_output))
                f.close()
                set_timestamp(f.name, (mtime,mtime))

        prev_output_fn = output_fn

def map_type(typename, types):
    for t in types:
        if typename in t:
            return t[0]

    return "ERR"

def get_timestamp(f):
    st = os.stat(f)
    atime = st[stat.ST_ATIME] #access time
    mtime = st[stat.ST_MTIME] #modification time
    return (atime,mtime)

def set_timestamp(f,timestamp):
    os.utime(f,timestamp)

def main():

    script_dir  = "." + os.sep + "codegen" + os.sep
    output_dir  = script_dir + "output" + os.sep
    tmpl_dir    = script_dir + "templates" + os.sep

    paths = {'types'     : join(script_dir,'element_types.json'),
             'opcodes'  : join(script_dir,'..','..','..','core','codegen','opcodes.json'),
             'operators': join(script_dir,'operators.json'),
             'self'     : join(script_dir,'gen.py')}

    types       = json.loads(open(paths['types']).read())
    opcodes     = json.loads(open(paths['opcodes']).read())
    operators   = json.loads(open(paths['operators']).read())

    #Find the latest modification time
    mtime = 0
    for _,p in paths.iteritems():
        t = get_timestamp(p)
        if t[1] > mtime:
            mtime = t[1]

    op_map  = []
    for name, opcode, t, mapped in (x for x in operators if x[3]):
        code = [x for x in opcodes if x['opcode'] == opcode and not x['system_opcode']]

        typesigs = [x["types"] for x in opcodes if x['opcode'] == opcode and not x['system_opcode']]
        typesigs = typesigs[0] if typesigs else []
        
        new_typesigs = []
        for ttt in typesigs:
            sig = [map_type(typesig,types)  for typesig in ttt]
            new_typesigs.append(sig)

        typesigs = new_typesigs

        opcode_base, nop = opcode.split("_", 1)
        if opcode_base == "CUSTOM":
            opcode  = opcode_base
            nop     = int(nop)
        elif code:
            nop = code[0]["nop"]
        else:
            print "The Bohrium opcodes no longer include [ %s ]." % opcode
            continue

        op_map.append((name, opcode, t, nop, typesigs))

    gens = [
        ('traits.ctpl',     'traits.hpp',    types),
        ('functions.ctpl',  'functions.hpp', op_map),
        ('operators.ctpl',  'operators.hpp', op_map),
    ]

    render( gens, tmpl_dir, output_dir, mtime )

if __name__ == "__main__":
    main()
