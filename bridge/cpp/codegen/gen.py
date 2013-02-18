#!/usr/bin/env python
import json
import os
from Cheetah.Template import Template

def main():

    script_dir  = "." + os.sep + "codegen" + os.sep
    output_dir  = script_dir + "output" + os.sep
    tmpl_dir    = script_dir + "templates" + os.sep

    types       = json.loads(open(script_dir+'element_types.json').read())
    opcodes     = json.loads(open(script_dir+'..'+ os.sep+'..'+ os.sep +'..'+ os.sep +'core'+ os.sep +'codegen'+ os.sep +'opcodes.json').read())
    
    operators   = json.loads(open(script_dir +'operators.json').read())

    op_map  = []
    for name, opcode, t in operators:
        code = [x for x in opcodes if x['opcode'] == opcode and not x['system_opcode']]
        if code:
            op_map.append( (name, opcode, t, code[0]["nop"]))

    gens = [
        ('head.ctpl',                   'bh_cppb_traits.hpp', types),
        ('bh_cppb_traits.ctpl',         'bh_cppb_traits.hpp', types),
        ('bh_cppb_traits.array.ctpl',   'bh_cppb_traits.hpp', types),
        ('bh_cppb_traits.const.ctpl',   'bh_cppb_traits.hpp', types),
        ('end.ctpl',                    'bh_cppb_traits.hpp', types),

        
        ('head.ctpl',                   'bh_cppb_functions.hpp', types),
        ('bh_cppb_functions.ctpl',      'bh_cppb_functions.hpp',   op_map),

        ('bh_cppb_operators.ctpl',      'bh_cppb_operators.hpp',   op_map),
        ('bh_cppb_operators.in.ctpl',   'bh_cppb_operators.hpp',   op_map),
        ('bh_cppb_operators.out.ctpl',  'bh_cppb_operators.hpp',   op_map),
        ('end.ctpl',                    'bh_cppb_operators.hpp',   op_map),
    ]

    prev_output_fn   = gens[0][1]
    prev_output      = ""
    count = (len(gens)-1)
    for c, (tmpl_fn, output_fn, data) in enumerate(gens):   # Concat the rendered template into output_fn
        t_tmpl  = Template(file= "%s%s" % (tmpl_dir, tmpl_fn), searchList=[{'data': data}])
        last = count == c

        if (output_fn != prev_output_fn ):
            open( output_dir + prev_output_fn, 'w').write( str(prev_output) )
            prev_output = ""

        prev_output += str(t_tmpl)

        if last:
            open( output_dir + output_fn, 'w').write( str(prev_output) )

        prev_output_fn   = output_fn

if __name__ == "__main__":
    main()
