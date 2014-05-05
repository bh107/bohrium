#!/usr/bin/env python
import json
import os
from os.path import join, exists
from pprint import pprint
from Cheetah.Template import Template
import stat

def render( gens, tmpl_dir, output_dir, mtime):

    if not exists(output_dir):
        os.mkdir(output_dir)

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
                f.write(str(prev_output))
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
            return t
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

    paths = {'reductions': join(script_dir,'reductions.json'),
             'opcodes'   : join(script_dir,'..','..','..','core','codegen','opcodes.json'),
             'types'     : join(script_dir,'..','..','cpp','codegen','element_types.json'),
             'self'      : join(script_dir,'gen.py')}

    reductions  = json.loads(open(paths['reductions']).read())
    opcodes     = json.loads(open(paths['opcodes']).read())
    types       = json.loads(open(paths['types']).read())

    #Find the latest modification time
    mtime = 0
    for _,p in paths.iteritems():
        t = get_timestamp(p)
        if t[1] > mtime:
            mtime = t[1]

    op_map  = []
    for op in opcodes:
        if op['system_opcode'] or not op['elementwise']:
            continue

        nop = op['nop']
        cpp_name = op['opcode'].lower()
        bh_name = op['opcode']
        c_name = op['opcode'].lower()[3:] #Removing the BH_

        typesigs = []
        for ttt in op['types']:
            sig = [map_type(typesig,types) for typesig in ttt]
            typesigs.append(sig)

        op_map.append((cpp_name, bh_name, c_name, nop, typesigs))

    gens = [
        ('type_header.ctpl',                'bh_c_data_types.h',                    (types, reductions)),
        ('type_definitions.ctpl',           'bh_c_type_definitions.hpp',            types),
        ('implementation_basics.ctpl',      'bh_c_implementation_basics.cpp',       (types, reductions)),
        ('method_header.ctpl',              'bh_c_interface.h',                     op_map),
        ('implementation.ctpl',             'bh_c_implementation.cpp',              op_map),
    ]

    render( gens, tmpl_dir, output_dir, mtime)

    #Merge bh_c_data_types.h and bh_c_interface.h into bhc.h
    with open(join(output_dir,"bh_c.h"), 'w') as outfile:
        for fname in [join(output_dir,"bh_c_data_types.h"), join(output_dir,"bh_c_interface.h")]:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

if __name__ == "__main__":
    main()
