#!/usr/bin/env python
import json
from Cheetah.Template import Template

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

def main():

    functions   = json.load(open('functions.json'))                 # Read the function definitions.
    ignore      = json.load(open('functions.ignore.json'))          # Read list of functions to ignore.

    cases       = []
    functors    = []
    log         = []
                                                                    # Generate the cpp code.
    for (opcode, nin, nout, signatures) in [f for f in functions if f[0] not in ignore]:

        fname   = opcode.replace('CPHVB_','').lower()
        opcount = nin+nout

        sigs        = []                                            # Filter signatures
        unsupported = []

        for sin, sout in signatures:
            type_sig    = [mapped[t] for t in sin+sout if t in mapped][::-1]
            u_sig       = [t for t in sin+sout if t not in mapped]
    
            if u_sig:
                unsupported.append( u_sig )

            if type_sig not in sigs and len(type_sig) > 1:
                sigs.append( type_sig )

        for signature in [s for s in sigs if len(s) == opcount]:    # case for each signature

            case = {
                'opcode':     opcode,
                'op1':        signature[0].upper(),
                'op2':        signature[1].upper(),
                'opcount':    opcount,
                'ftypes':     ','.join(signature).lower(),
                'fname':      fname
            }

            if opcount == 3:
                case['op3'] = signature[2].upper()
            cases.append( case )

        functor = {                                                 # Abstract functor for "function"
            'fname':        fname,
            'type_params':  ', '.join(['typename T%d'%t for t in xrange(1, opcount+1)]),
            'fparams':      ', '.join(['T%d *op%d'% (t, t) for t in xrange(1, opcount+1)])
        }

        functors.append( functor )

        if unsupported:
            log.append("%s\t\tUnsupported signatures\t%s" %(opcode, ' | '.join([','.join(s) for s in unsupported])))

                                                                    # Generate the cpp code.
    f_tmpl  = Template(file='functors.ctpl', searchList=[{'functors': functors}])
    d_tmpl  = Template(file='dispatch.ctpl', searchList=[{'cases': cases}])
    t_tmpl  = Template(file='get_traverse.ctpl', searchList=[{'cases': cases}])
    
                                                                    # Write them to file
    open('functors.gen','w').write(str(f_tmpl))
    open('dispatch.gen','w').write(str(d_tmpl))
    open('get_traverse.gen','w').write(str(t_tmpl))
    open('gen.log','w+').write('\n'.join(log))

if __name__ == "__main__":
    main()
