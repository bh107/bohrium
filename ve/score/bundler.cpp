/*
 * Copyright 2011 Simon A. F. Lund <safl@safl.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include "bundler.hpp"
#include <iostream>
#include <set>

typedef cphvb_array* cphvb_array_ptr;

/* Calculates the bundleable instructions.
 *
 * @inst The instruction list
 * @size Size of the instruction list
 * @return Number of consecutive bundable instructions.
 */
cphvb_intp bundle(cphvb_instruction *insts[], cphvb_intp size)
{

    bool do_fuse = true;                                            // Loop invariant
    cphvb_intp bundle_len = 0;                                      // Number of cons. bundl. instr.
                                                                    // incremented on each iteration

    int opcount = 0;                                                // Per-instruction variables
    cphvb_array_ptr op, base;                                       // re-assigned on each iteration.

    for(cphvb_intp i=0; ((do_fuse) && (i<size)); i++) {             // Go through the instructions...

        opcount = cphvb_operands(insts[i]->opcode);

        for(int j=0; ((do_fuse) && (j<opcount)); j++) {             // Go through each operand.

            op      = insts[i]->operand[j];
            base    = op->base == NULL ? op : op->base;
            
            // Check alignment
            // i == 0:  - output operand => check against kernel input and output
            // i > 0:   - input operand  => check against kernel-output

            // perhaps this should be aided by two multisets, one containing output operands
            // another containing input operands.

        }

        if (do_fuse) {                                                  // Instruction is allowed
            bundle_len++;
        }

    }

    #ifdef DEBUG_BNDL
    if (bundle_len > 1) {

        std::cout << "BUNDLING " << size << " {" << std::endl;
        for(cphvb_intp i=0; ((do_fuse) && (i<size)); i++) {             // Go through the instructions...
            cphvb_instr_pprint( insts[i] );
        }
        std::cout << "} ops {" << std::endl << "  ";
        for(it = ops.begin(); it != ops.end(); it++)
        {
            std::cout << *it << ",";
        }
        std::cout << std::endl;
        std::cout << "} bundle len = [" << bundle_len << "]" << std::endl;
    }
    #endif
    if(bundle_len<1) {
        bundle_len = 1;
    }

    bundle_len = 1; // This is just here until bundling is done...
    return bundle_len;

}
