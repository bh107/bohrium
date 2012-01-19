#!/usr/bin/env python
from string import Template
import json

nonmapped = {        # The following types does not have an equivalent in CPHVB

    'g': 'longdouble',

    'e': 'float16',
    'F': 'cfloat',
    'D': 'cdouble',
    'G': 'clongdouble',

    'M': 'datetime',
    'm': 'timedelta',
    'O': 'OBJECT',
    'P': 'OBJECT',
}

mapped = {          # Datatype mapping from numpy to CPHVB
    '?': 'CPHVB_BOOL',

    'b': 'CPHVB_INT8',
    'B': 'CPHVB_UINT8',

    'h': 'CPHVB_INT16',
    'H': 'CPHVB_UINT16',

    'i': 'CPHVB_INT32',
    'I': 'CPHVB_UINT32',

    'l': 'CPHVB_INT32',
    'L': 'CPHVB_UINT32',

    'q': 'CPHVB_INT64',
    'Q': 'CPHVB_UINT64',

    'f': 'CPHVB_FLOAT32',
    'd': 'CPHVB_FLOAT64',
}

def indent( text, w=4, c=1):
    return '\n'.join([(' '*w)*c+line for line in text.split('\n')])

def main():

    functions   = json.load(open('functions.json'))         # Read the function definitions.
    ignore      = json.load(open('functions.ignore.json'))  # Read list of functions to ignore.
                                
    dispatch    = open('dispatch.tmpl.cpp').read()          # Read the templates.
    case2       = open('dispatch.case.2.tmpl.cpp').read()
    case3       = open('dispatch.case.3.tmpl.cpp').read()
    functor     = open('functor.tmpl.cpp').read()
    functors    = open('functors.tmpl.cpp').read()
    cases = ''
                                                            # Generate the cpp code.
    for (opcode, nin, nout, signatures) in [f for f in functions if f[0] not in ignore]:

        opcount = nin+nout

        sigs        = []                                    # Filter signatures
        unsupported = []

        for sin, sout in signatures:
            type_sig    = [mapped[t] for t in sin+sout if t in mapped]
            u_sig       = [t for t in sin+sout if t not in mapped]
    
            if u_sig:
                unsupported.append( u_sig )

            if type_sig not in sigs and len(type_sig) > 1:
                sigs.append( type_sig )

        for signature in sigs:                              # Generate case

            types_l = len(signature)

            if types_l == 2 and types_l == opcount:

                cases   += Template(case2).substitute(
                    opcode  = opcode,
                    op1     = signature[0].upper(),
                    op2     = signature[1].upper(),
                    opcount = opcount,
                    ftypes  = ','.join(signature).lower(),
                    fname   = opcode.replace('CPHVB_','').lower()
                )

            elif types_l == 3 and types_l == opcount:

                cases   += Template(case3).substitute(
                    opcode  = opcode,
                    op1     = signature[0].upper(),
                    op2     = signature[1].upper(),
                    op3     = signature[2].upper(),
                    opcount = opcount,
                    ftypes  = ','.join(signature).lower(),
                    fname   = opcode.replace('CPHVB_','').lower()
                )
            
        if unsupported:
            print "%s\t\tUnsupported signatures\t%s" %(opcode, ' | '.join([','.join(s) for s in unsupported]))

    with open('functors.gen.hpp','w') as fd:            # Store the cpp code.
        fd.write( functors )

    with open('dispatch.gen.cpp','w') as fd:
        fd.write( Template(dispatch).substitute( cases=indent("\n"+cases,4,4) ))

if __name__ == "__main__":
    main()
