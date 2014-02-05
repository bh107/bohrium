#!/usr/bin/env python
from pprint import pprint
import glob
import json
import sys
import os

from Cheetah.Template import Template

#
#   Functions defining mappings between opcodes, types and the template-files.
#
def bh_type_sig_check(opcodes, types):

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

#
#   Read bytecode definition(opcodes.json and types.json) and call
#   data mapping functions to fill out function templates.
#
def main(script_dir):

    templ_path   = os.sep.join([script_dir, 'templates'])
    types_path   = os.sep.join([script_dir, 'types.json'])
    opcodes_path = os.sep.join([script_dir, 'opcodes.json'])

    types   = json.load(open(types_path))
    opcodes = json.load(open(opcodes_path))

    code = {}

    # Map template names to mapping-functons and fill out the template
    for fn in glob.glob('templates/*.tpl'):
        fn, _ = os.path.basename(fn).split('.tpl')

        # Find the functions within scope
        if fn in globals().keys():
            code[fn] = str(Template(
                file = "%s%s%s.tpl" % ("templates", os.sep, fn),
                searchList = globals()[fn](opcodes, types)
            ))

    # Now decide what to do with the generated code
    for k in code:
        print code[k]

if __name__ == "__main__":
    try:
        script_dir = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        print "The build script cannot run interactively."
        sys.exit(-1)
    main(script_dir)
