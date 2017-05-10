#!/usr/bin/env python
import json
import os
from os.path import join
import stat
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(join(script_dir, "..", "..", "..", "thirdparty", "pyratemp"))
from pyratemp import Template


def render(gens, tmpl_dir, output_dir, mtime):
    license = open("%s%s" % (tmpl_dir, 'license.txt')).read()
    warn    = open("%s%s" % (tmpl_dir, 'warn.txt')).read()

    prev_output_fn = gens[0][1]
    prev_output    = ""

    count = (len(gens)-1)
    for c, (tmpl_fn, output_fn, data) in enumerate(gens):   # Concat the rendered template into output_fn
        const_file = Template(open("%s%s" % (tmpl_dir, "traits.const.tpl")).read(), escape=None)
        const_file = const_file(data=data)

        array_file = Template(open("%s%s" % (tmpl_dir, "traits.array.tpl")).read(), escape=None)
        array_file = array_file(data=data)

        t_tmpl = Template(open("%s%s" % (tmpl_dir, tmpl_fn)).read(), escape=None)
        t_tmpl = t_tmpl(
            tmpl_dir=tmpl_dir,
            license=license,
            warn=warn,
            const_file=const_file,
            array_file=array_file,
            data=data
        )

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
    atime = st[stat.ST_ATIME] # access time
    mtime = st[stat.ST_MTIME] # modification time
    return (atime,mtime)

def set_timestamp(f, timestamp):
    os.utime(f, timestamp)

def main():
    script_dir = "." + os.sep + "codegen" + os.sep
    output_dir = script_dir + "output" + os.sep
    tmpl_dir   = script_dir + "templates" + os.sep

    paths = {'types'    : join(script_dir,'element_types.json'),
             'opcodes'  : join(script_dir,'..','..','..','core','codegen','opcodes.json'),
             'operators': join(script_dir,'operators.json'),
             'self'     : join(script_dir,'gen.py')}

    types       = json.loads(open(paths['types']).read())
    opcodes     = json.loads(open(paths['opcodes']).read())
    operators   = json.loads(open(paths['operators']).read())

    # Find the latest modification time
    mtime = 0
    for _,p in paths.items():
        t = get_timestamp(p)
        if t[1] > mtime:
            mtime = t[1]

    op_map = []

    datasets = {}
    for name, opcode, mapper, mapped in (x for x in operators if x[3]):
        bytecode = [x for x in opcodes if x['opcode'] == opcode]
        if not bytecode:
            print ("skipping %s (\"%s\")" % (opcode, name))
            continue

        bytecode = bytecode[0]

        typesigs = bytecode["types"]

        layouts = bytecode["layout"]
        broadcast = bytecode["elementwise"]

        new_typesigs = []
        for ttt in typesigs:
            sig = [map_type(typesig, types) for typesig in ttt]
            new_typesigs.append(tuple(sig))

        typesigs = new_typesigs

        opcode_base, nop = opcode.split("_", 1)
        if opcode_base == "CUSTOM":
            opcode  = opcode_base
            nop     = int(nop)
        elif bytecode:
            nop = bytecode["nop"]
        else:
            print ("The Bohrium opcodes no longer include [ %s ]." % opcode)
            continue

        if mapper not in datasets:
            datasets[mapper] = []

        foo = (name, opcode, mapper, nop, typesigs, layouts, broadcast)
        datasets[mapper].append(foo)
        op_map.append(foo)
    op_map.sort()

    # Generate data for generation of type-checker.
    enums = set()
    checker = []

    for op in op_map:
        fun, enum, template, nop, typesigs, layouts, broadcast = op

        if enum == "BH_RANDOM":
            nop = 3
            typesigs = [(u'uint64_t', u'uint64_t', u'uint64_t')]
            op = (fun, enum, template, nop, typesigs, layouts, broadcast)
        elif enum == "BH_REPEAT":
            nop = 2
            typesigs = [(u'uint64_t', u'uint64_t')]
            op = (fun, enum, template, nop, typesigs, layouts, broadcast)

        if enum not in enums:
            checker.append(op)
            enums.add(enum)

    gens = [
        ('traits.tpl',              'traits.hpp',              types),

        ('sugar.header.tpl',        'operators.hpp',           datasets['sugar.nops2']),
        ('sugar.nops2.tpl',         'operators.hpp',           datasets['sugar.nops2']),
        ('sugar.nops2.bool.tpl',    'operators.hpp',           datasets['sugar.nops2.bool']),
        ('sugar.nops3.tpl',         'operators.hpp',           datasets['sugar.nops3']),
        ('sugar.nops3.bool.tpl',    'operators.hpp',           datasets['sugar.nops3.bool']),
        ('sugar.nops3.intern.tpl',  'operators.hpp',           datasets['sugar.nops3.intern']),
        ('sugar.footer.tpl',        'operators.hpp',           datasets['sugar.nops2']),

        ('runtime.typechecker.tpl', 'runtime.typechecker.hpp', checker),

        ('runtime.header.tpl',      'runtime.operations.hpp',  datasets['runtime.nops3']),
        ('runtime.nops0.tpl',       'runtime.operations.hpp',  datasets['runtime.nops0']),
        ('runtime.nops4.tpl',       'runtime.operations.hpp',  datasets['runtime.nops4']),
        ('runtime.nops3.tpl',       'runtime.operations.hpp',  datasets['runtime.nops3']),
        ('runtime.nops2.tpl',       'runtime.operations.hpp',  datasets['runtime.nops2']),
        ('runtime.nops1.tpl',       'runtime.operations.hpp',  datasets['runtime.nops1']),
        ('runtime.random.tpl',      'runtime.operations.hpp',  datasets['runtime.random']),
        ('runtime.accumulate.tpl',  'runtime.operations.hpp',  datasets['runtime.accumulate']),
        ('runtime.reduce.tpl',      'runtime.operations.hpp',  datasets['runtime.reduce']),
        ('runtime.footer.tpl',      'runtime.operations.hpp',  datasets['runtime.reduce']),
    ]

    render(gens, tmpl_dir, output_dir, mtime)

if __name__ == "__main__":
    main()
