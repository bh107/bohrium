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

            if(instr->userfunc->id == reduce_impl_id) {
                reduce_impl(instr->userfunc);
                break;
            } else if(instr->userfunc->id == random_impl_id) {
                random_impl(instr->userfunc);
                break;
            } else {
                                // Unsupported instruction
                instr->status = CPHVB_TYPE_NOT_SUPPORTED;
                return CPHVB_PARTIAL_SUCCESS;
            }

        default:                // Element-wise functions + Memory Functions

            // Poly contains a unique value, pairing an opcode with its function signature.
            // All in one nicely switchable value.
            const long int poly = (cphvb_operands(instr->opcode) == 3) ? 
                                    instr->opcode +
                                    (instr->operand[0]->type    << 8)
                                    +(instr->operand[1]->type   << 12)
                                    +(instr->operand[2]->type   << 16):

                                    instr->opcode +
                                    (instr->operand[0]->type    << 8)
                                    +(instr->operand[1]->type   << 12);

            switch(poly) {

                ${cases}

                default:        // Unsupported instruction / type
                    instr->status = CPHVB_TYPE_NOT_SUPPORTED;
                    return CPHVB_PARTIAL_SUCCESS;

            }

    }

    return res;

}

