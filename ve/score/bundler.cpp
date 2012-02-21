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

    std::set<cphvb_array_ptr> ops, out;                             // out = output operands in kernel
    std::set<cphvb_array_ptr>::iterator it;                         // ops = all operands in kernel (out+in)
    std::pair<std::set<cphvb_array_ptr>::iterator, bool> ins_res;

    bool do_fuse = true;                                            // Loop invariant
    cphvb_intp bundle_len = 0;                                      // Number of cons. bundl. instr.
                                                                    // incremented on each iteration

    int opcount = 0;                                                // Per-instruction variables
    cphvb_array_ptr op;                                             // re-assigned on each iteration.

    for(cphvb_intp i=0; ((do_fuse) && (i<size)); i++) {             // Go through the instructions...

        opcount = cphvb_operands(insts[i]->opcode);

        for(int j=0; j<opcount; j++) {                              // Go through each operand.
            ops.insert( insts[i]->operand[j] );
        }
        for(int j=0; ((do_fuse) && (j<opcount)); j++) {             // Go through each operand.

            op = insts[i]->operand[j];
                                                                    // Determine splicability
            for(it = ops.begin(); ((do_fuse) && (it != ops.end())); it++) {

                if (op->base != NULL) {
                    do_fuse = false;
                
                } else {

                    if ( (op == (*it)->base) || (op->base == *it) || (op->base == (*it)->base )) {                    // Same base
                        if ((op->ndim == (*it)->ndim) &&                // Same dim and start
                            (op->start == (*it)->start)) {
                            
                            for(cphvb_intp k =0; k<op->ndim; k++) {
                                if ((op->stride[k] != (*it)->stride[k]) ||
                                    (op->shape[k] != (*it)->shape[k])) {
                                    do_fuse = false;                    // Incompatible shape or stride
                                    break;
                                }
                            }

                        } else {                                        // Incompatible dim or start

                            do_fuse = false;
                            break;

                        }

                    } // Different base => all is good.

                }

            }

        }

        if (do_fuse) {
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
    return bundle_len;

}
