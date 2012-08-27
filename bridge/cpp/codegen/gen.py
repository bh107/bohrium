#!/usr/bin/env python
import json
import os
from Cheetah.Template import Template

def main():

    script_dir  = "." + os.sep + "codegen" + os.sep
    output_dir  = script_dir + "output" + os.sep
    tmpl_dir    = script_dir + "templates" + os.sep

    types       = json.loads(open(script_dir+'..'+ os.sep+'..'+ os.sep +'..'+ os.sep +'core'+ os.sep +'codegen'+ os.sep +'types.json').read())
    opcodes     = json.loads(open(script_dir+'..'+ os.sep+'..'+ os.sep +'..'+ os.sep +'core'+ os.sep +'codegen'+ os.sep +'opcodes.json').read())
    
    operators   = json.loads(open(script_dir +'operators.json').read())

    op_map  = []
    for name, opcode, t in operators:
        code = [x for x in opcodes if x['opcode'] == opcode and not x['system_opcode']]
        if code:
            op_map.append( (name, opcode, t, code[0]["nop"]))

    gens = [
        ('assign_const.ctpl',   'assign_const.cpp', types),
        ('assign_array.ctpl',   'assign_array.cpp', types),
        ('operator.fun.ctpl',   'op_fun.cpp',   op_map),
        ('operator.in.ctpl',    'op_in.cpp',    op_map),
        ('operator.out.ctpl',   'op_out.cpp',   op_map),
    ]

    for tmpl_fn, output_fn, data in gens:
        t_tmpl  = Template(file= "%s%s" % (tmpl_dir, tmpl_fn), searchList=[{'data': data}])
        open( output_dir + output_fn, 'w').write( str(t_tmpl) )

if __name__ == "__main__":
    main()
