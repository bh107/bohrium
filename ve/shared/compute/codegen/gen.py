#!/usr/bin/env python
import pprint
import json
import os
from Cheetah.Template import Template

def main():

    script_dir  = "."+os.sep
    #output_dir  = script_dir + "./output/" + os.sep
    output_dir  = script_dir + "../" + os.sep
    tmpl_dir    = script_dir + "templates" + os.sep
    
    gens = [
        ('traverser',   'traverser.ctpl',       'traverser.hpp'),
        ('functors',    'functors.ctpl',        'functors.hpp'),
        ('compute',     'bh_compute.ctpl',      'bh_compute.cpp'),
        ('reduce',      'bh_reduce.ctpl',       'bh_compute_reduce.cpp'),
        ('typeutil',    'bh_typeutil.ctpl',     'bh_typeutil.cpp'),
    ]

    ignore  = json.load(open(script_dir+'ignore.json'))
    opcodes = json.loads(open(script_dir+'../../../core/codegen/opcodes.json').read())

    for mod_name, tmpl_fn, output_fn in gens:

        module  = __import__("gen.%s" % mod_name, globals(), locals(), [], -1 ).__dict__[mod_name]
        data    = module.gen( opcodes, ignore )
        t_tmpl  = Template(file= "%s%s" % (tmpl_dir, tmpl_fn), searchList=[{'data': data}])

        open( output_dir + output_fn, 'w').write( str(t_tmpl) )

if __name__ == "__main__":
    main()
