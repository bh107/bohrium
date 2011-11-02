#!/usr/bin/env python

from enums import *

switch_tmpl="""

cphvb_instruction   *instr  = &instruction_list[i];
const long int      poly    = instr->opcode*100 + instr->operand[0]->type;

switch(poly) {

    case CPHVB_NONE:        // Nothing to do since we only use main memory.
    case CPHVB_DISCARD:
    case CPHVB_RELEASE:
    case CPHVB_SYNC:
        break;

    __CASES__

    case CPHVB_RANDOM:      // Memory functions
    case CPHVB_ARANGE:
        iter_1<cphvb_float32>( instr, &opcode<CPHVB_RANDOM, cphvb_float32>::invoke  );
        break;


    default:                // Unsupported instruction
        fprintf(
            stderr, 
            "cphvb_ve_score_execute() encountered an unknown opcode: %s.",
            cphvb_opcode_text( instr->opcode )
        );
        exit(CPHVB_INST_NOT_SUPPORTED);

}
"""

case_tmpl="""
case __OPCODE__*100+__ETYPE__:
    iter___OPCOUNT__<__TYPE__>( instr, &opcode<__OPCODE__, __TYPE__>::invoke );
    break;"""

def indent( text, c=1):
    return '\n'.join(['    '*c+line for line in text.split('\n')])

def main():
    
    case = ''

    for x in (x for x in (blank).split('\n') if x):
        for t in (x for x in types.split('\n') if x and x not in ignore):
            case += "case __OPCODE__*100+__ETYPE__:\n"\
                    .replace('__OPCODE__', x)\
                    .replace('__ETYPE__', t)
    case += "    break;\n"

    for (count, opcodes) in opcode_map:
        for x in (x for x in opcodes.split('\n') if x):
            for t in (x for x in types.split('\n') if x and x not in ignore):
                case  += case_tmpl\
                        .replace('__OPCOUNT__', str(count))\
                        .replace('__OPCODE__', x)\
                        .replace('__ETYPE__', t)\
                        .replace('__TYPE__', t.lower())
 
    print switch_tmpl.replace('__CASES__', indent(case))

if __name__ == "__main__":
    main()
