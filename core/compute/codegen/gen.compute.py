#!/usr/bin/env python
import re
import json
from Cheetah.Template import Template

def main():

    defs = json.loads(re.sub('//.*?\n|/\*.*?\*/', '', open('../../codegen/opcodes.json').read(), re.S, re.DOTALL | re.MULTILINE))

    functions = ((opcode, 1, defs[opcode]['nop']-1, defs[opcode]['types']) for opcode in defs)

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

        for signature in [s for s in sigs if len(s) == opcount]:            # case for each signature

            case = {
                'opcode':       opcode,
                'op1':          signature[0].upper(),
                'op2':          signature[1].upper(),
                'opcount':      opcount,
                'ftypes':       ','.join(signature).lower(),
                'fname':        fname
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
    t_tmpl  = Template(file='cphvb_compute.ctpl', searchList=[{'cases': cases}])

    open('cphvb_compute.gen','w').write(str(t_tmpl))                # Write them to file
    open('gen.log','w+').write('\n'.join(log))

if __name__ == "__main__":
    main()
