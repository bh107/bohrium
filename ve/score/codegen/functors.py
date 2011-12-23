#!/usr/bin/env python
from enums import *

dispatch_tmpl="""
#include <cphvb.h>
#include "dispatch.h"
#include "traverse.hpp"
#include "functors.hpp"

inline cphvb_error dispatch( cphvb_instruction *instr ) {

    cphvb_error res = CPHVB_SUCCESS;

    switch(instr->opcode) {

        case CPHVB_NONE:        // Nothing to do since we only use main memory.
        case CPHVB_DISCARD:
        case CPHVB_RELEASE:
        case CPHVB_SYNC:
            break;

        case CPHVB_USERFUNC:
            if(instr->userfunc->id == reduce_impl_id)
            {
                reduce_impl(instr->userfunc);
                break;
            }
            else if(instr->userfunc->id == random_impl_id)
            {
                random_impl(instr->userfunc);
                break;
            }
            else
            {
                // Unsupported instruction
                instr->status = CPHVB_TYPE_NOT_SUPPORTED;
                return CPHVB_PARTIAL_SUCCESS;
            }

        default:                // Element-wise functions + Memory Functions

            const long int poly = instr->opcode*100 + instr->operand[0]->type;

            switch(poly) {

                __CASES__

                default:                // Unsupported instruction
                    instr->status = CPHVB_TYPE_NOT_SUPPORTED;
                    return CPHVB_PARTIAL_SUCCESS;

            }

    }

    return res;

}

"""

case_tmpl="""
case __OPCODE__*100+__ETYPE__:
    __TYPE_CHECK__
    traverse___OPCOUNT__<__TYPES__, __FUNC___functor<__TYPES__> >( instr );
    break;"""

functor_prefix_tmpl="""
#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG_CIR 360.0
#define DEG_RAD (M_PI / (DEG_CIR / 2.0))
#define RAD_DEG ((DEG_CIR / 2.0) / M_PI)
"""

functor_tmpl="""
template <__TYPE_PARAMS__>
struct __FUNC___functor {
    void operator()(__PARAMS__) {
    }
};
"""

def indent( text, c=1):
    return '\n'.join(['    '*c+line for line in text.split('\n')])

def main():

    func = ''
    case = ''

    for (count, opcodes) in opcode_map:
        for x in (x for x in opcodes.split('\n') if x):
            for t in (x for x in types.split('\n') if x and x not in ignore):
                if not (t in types_float and x in opcode_no_float_support):
                    typelist_u = [t.upper() for i in xrange(1,count+1)]
                    typelist_l = [t.lower() for i in xrange(1,count+1)]

                    if x in opcode_bool_out:
                        typelist_u[0] = "CPHVB_BOOL"
                        typelist_l[0] = typelist_u[0].lower()

                    type_check = 'if(instr->operand[0]->type != %s'%typelist_u[0]
                    for i in xrange(1,count):
                        type_check += ' || instr->operand[%d]->type != %s'%(i,typelist_u[i])
                    type_check += '){instr->status = CPHVB_TYPE_NOT_SUPPORTED; return CPHVB_PARTIAL_SUCCESS;}'

                    case  += case_tmpl\
                            .replace('__OPCOUNT__', str(count))\
                            .replace('__OPCODE__', x)\
                            .replace('__ETYPE__', t)\
                            .replace('__FUNC__', x.lower().replace('cphvb_', ''))\
                            .replace('__TYPE__', t.lower())\
                            .replace('__TYPES__', ','.join(typelist_l))\
                            .replace('__TYPE_CHECK__', type_check)
            func += functor_tmpl\
                    .replace('__FUNC__', x.lower().replace('cphvb_', ''))\
                    .replace('__PARAMS__',      ', '.join(["T%d *op%d" % (i, i)        for i in xrange(1,count+1)]))\
                    .replace('__TYPE_PARAMS__', ', '.join(["typename T%d" % i    for i in xrange(1,count+1)]))

    dispatch = dispatch_tmpl.replace('__CASES__', indent(case, 4))

    with open('functors.gen.hpp','w') as fd:
        fd.write( functor_prefix_tmpl+func )

    with open('dispatch.gen.cpp','w') as fd:
        fd.write( dispatch )

if __name__ == "__main__":
    main()
