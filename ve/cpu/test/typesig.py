#!/usr/bin/env python
from pprint import pprint
import json

etu = {
    "BH_BOOL": 1,
    "BH_INT8": 2,
    "BH_INT16": 3,
    "BH_INT32": 4,
    "BH_INT64": 5,
    "BH_UINT8": 6,
    "BH_UINT16":7,
    "BH_UINT32":8,
    "BH_UINT64":9,
    "BH_FLOAT16":10,
    "BH_FLOAT32":11,
    "BH_FLOAT64":12,
    "BH_COMPLEX64":13,
    "BH_COMPLEX128":14,
    "BH_UNKNOWN":15
}

ets = {
    "BH_BOOL": "z",
    "BH_INT8": "b",
    "BH_INT16": "s",
    "BH_INT32": "i",
    "BH_INT64": "l",
    "BH_UINT8": "B",
    "BH_UINT16": "S",
    "BH_UINT32": "I",
    "BH_UINT64": "L",
    "BH_FLOAT16": "h",
    "BH_FLOAT32": "f",
    "BH_FLOAT64": "d",
    "BH_COMPLEX64": "c",
    "BH_COMPLEX128": "C",
    "BH_UNKNOWN": "U"
}

opcodes = json.load(open('../../../core/codegen/opcodes.json'))

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
        cases.append('      // Sig-Length %d' % slen)
    cases.append('      case %d: return "%s"; // %s' %(nsig, hsig, tsig))

# Sig-length 1 is missing.. will come around later on..

pprint(tsigs)
pprint(nsigs)
pprint(zip(tsigs, nsigs))
print len([typesig for slen in typesigs for typesig in typesigs[slen]])
print len(set(tsigs))
print len(set(nsigs))
print("\n".join(cases))
