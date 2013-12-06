#!/usr/bin/env python
from pprint import pprint

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

mask_tmpl = """
const char* bh_layoutmask_to_shorthand(const int mask)
{
    switch(mask) {
__BITMASKS__
        default:
            return "___";
    }
}"""

masks = ""
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
    masks += '        case %d: return "%s";\n' % (bitmask, mask)

print mask_tmpl.replace("__BITMASKS__", masks)
