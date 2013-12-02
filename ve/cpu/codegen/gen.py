#!/usr/bin/env python
from pprint import pprint
import glob
import json
import sys
import os

from Cheetah.Template import Template

def main(self):

    prefix  = "../../../core/codegen"
    types   = json.load(open("%s%s%s.json" % (prefix, os.sep, 'types')))
    opcodes = json.load(open("%s%s%s.json" % (prefix, os.sep, 'opcodes')))

    # Map template names to mapping-functons and fill out the template
    for fn in glob.glob('templates/*.tpl'):
        fn, _ = os.path.basename(fn).split('.tpl')
        if fn in self.__dict__:
            func    = self.__dict__[fn]
            mapping = func(opcodes, types)
            template = Template(
                file = "%s%s%s.tpl" % ("templates", os.sep, fn),
                searchList = [mapping]
            )
            print template

def enum_to_ctypestr(opcodes, types):
    return {"types": [(t["enum"], t["cpp"]) for t in types]}

def enum_to_shorthand(opcodes, types):
    return {"types": [(t["enum"], t["shorthand"]) for t in types]}

def enumstr_to_ctypestr(opcodes,types):
    return {"types": [(t["enum"], t["c"]) for t in types]}

def enumstr_to_shorthand(opcodes, types):
    return {"types": [(t["enum"], t["shorthand"]) for t in types]}

def layoutmask_to_shorthand(opcodes, types):
    A0_CONSTANT = 1 << 0;
    A0_DENSE    = 1 << 1;
    A0_STRIDED  = 1 << 2;
    A0_SPARSE   = 1 << 3;

    A1_CONSTANT = 1 << 4;
    A1_DENSE    = 1 << 5;
    A1_STRIDED  = 1 << 6;
    A1_SPARSE   = 1 << 7;

    A2_CONSTANT = 1 << 8;
    A2_DENSE    = 1 << 9;
    A2_STRIDED  = 1 << 10;
    A2_SPARSE   = 1 << 11;

    hej = []

    # Binary instructions
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_DENSE)
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_CONSTANT | A1_DENSE | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_DENSE | A2_DENSE)
    hej.append(A0_CONSTANT | A1_DENSE | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_DENSE | A2_SPARSE)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_DENSE)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_STRIDED | A2_SPARSE)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_DENSE)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_STRIDED)
    hej.append(A0_CONSTANT | A1_SPARSE | A2_SPARSE)
    hej.append(A0_DENSE | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_DENSE | A1_CONSTANT | A2_DENSE)
    hej.append(A0_DENSE | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_DENSE | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_DENSE | A1_DENSE | A2_CONSTANT)
    hej.append(A0_DENSE | A1_DENSE | A2_DENSE)
    hej.append(A0_DENSE | A1_DENSE | A2_STRIDED)
    hej.append(A0_DENSE | A1_DENSE | A2_SPARSE)
    hej.append(A0_DENSE | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_DENSE | A1_STRIDED | A2_DENSE)
    hej.append(A0_DENSE | A1_STRIDED | A2_STRIDED)
    hej.append(A0_DENSE | A1_STRIDED | A2_SPARSE)
    hej.append(A0_DENSE | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_DENSE | A1_SPARSE | A2_DENSE)
    hej.append(A0_DENSE | A1_SPARSE | A2_STRIDED)
    hej.append(A0_DENSE | A1_SPARSE | A2_SPARSE)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_DENSE)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_STRIDED | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_STRIDED | A1_DENSE | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_DENSE | A2_DENSE)
    hej.append(A0_STRIDED | A1_DENSE | A2_STRIDED)
    hej.append(A0_STRIDED | A1_DENSE | A2_SPARSE)
    hej.append(A0_STRIDED | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_STRIDED | A2_DENSE)
    hej.append(A0_STRIDED | A1_STRIDED | A2_STRIDED)
    hej.append(A0_STRIDED | A1_STRIDED | A2_SPARSE)
    hej.append(A0_STRIDED | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_STRIDED | A1_SPARSE | A2_DENSE)
    hej.append(A0_STRIDED | A1_SPARSE | A2_STRIDED)
    hej.append(A0_STRIDED | A1_SPARSE | A2_SPARSE)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_DENSE)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_STRIDED)
    hej.append(A0_SPARSE | A1_CONSTANT | A2_SPARSE)
    hej.append(A0_SPARSE | A1_DENSE | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_DENSE | A2_DENSE)
    hej.append(A0_SPARSE | A1_DENSE | A2_STRIDED)
    hej.append(A0_SPARSE | A1_DENSE | A2_SPARSE)
    hej.append(A0_SPARSE | A1_STRIDED | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_STRIDED | A2_DENSE)
    hej.append(A0_SPARSE | A1_STRIDED | A2_STRIDED)
    hej.append(A0_SPARSE | A1_STRIDED | A2_SPARSE)
    hej.append(A0_SPARSE | A1_SPARSE | A2_CONSTANT)
    hej.append(A0_SPARSE | A1_SPARSE | A2_DENSE)
    hej.append(A0_SPARSE | A1_SPARSE | A2_STRIDED)
    hej.append(A0_SPARSE | A1_SPARSE | A2_SPARSE)

    # Unary
    hej.append(A0_CONSTANT | A1_CONSTANT)
    hej.append(A0_CONSTANT | A1_DENSE)
    hej.append(A0_CONSTANT | A1_STRIDED)
    hej.append(A0_CONSTANT | A1_SPARSE)
    hej.append(A0_DENSE | A1_CONSTANT)
    hej.append(A0_DENSE | A1_DENSE)
    hej.append(A0_DENSE | A1_STRIDED)
    hej.append(A0_DENSE | A1_SPARSE)
    hej.append(A0_STRIDED | A1_CONSTANT)
    hej.append(A0_STRIDED | A1_DENSE)
    hej.append(A0_STRIDED | A1_STRIDED)
    hej.append(A0_STRIDED | A1_SPARSE)
    hej.append(A0_SPARSE | A1_CONSTANT)
    hej.append(A0_SPARSE | A1_DENSE)
    hej.append(A0_SPARSE | A1_STRIDED)
    hej.append(A0_SPARSE | A1_SPARSE)

    hej.append(A0_CONSTANT)
    hej.append(A0_DENSE)
    hej.append(A0_STRIDED)
    hej.append(A0_SPARSE)

    hej.sort()

    masks = []
    for bitmask in hej:
        mask = ""
        if ((bitmask & A0_CONSTANT) != 0):
            mask += "C"
        if ((bitmask & A0_DENSE) != 0):
            mask += "D"
        if ((bitmask & A0_STRIDED) != 0):
            mask += "S"
        if ((bitmask & A0_SPARSE) != 0):
            mask += "P"

        if ((bitmask & A1_CONSTANT) != 0):
            mask += "C"
        if ((bitmask & A1_DENSE) != 0):
            mask += "D"
        if ((bitmask & A1_STRIDED) != 0):
            mask += "S"
        if ((bitmask & A1_SPARSE) != 0):
            mask += "P"

        if ((bitmask & A2_CONSTANT) != 0):
            mask += "C"
        if ((bitmask & A2_DENSE) != 0):
            mask += "D"
        if ((bitmask & A2_STRIDED) != 0):
            mask += "S"
        if ((bitmask & A2_SPARSE) != 0):
            mask += "P"
        masks.append((bitmask, mask))

    return {'masks': masks}

if __name__ == "__main__":
    main(sys.modules[__name__])
