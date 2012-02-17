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
#include "pp.h"

typedef cphvb_array* cphvb_array_ptr;

/* Calculates the bundleable instructions.
 *
 * @inst The instruction list
 * @size Size of the instruction list
 * @return Number of consecutive bundable instructions.
 */
cphvb_intp bundle(cphvb_instruction *insts[], cphvb_intp size)
{

    cphvb_intp bundle_len = 0;                                      // Number of cons. bundl. instr.

    std::set<cphvb_array_ptr> out;                                  // Sets for classifying operands.
    std::set<cphvb_array_ptr>::iterator it;
    std::pair<std::set<cphvb_array_ptr>::iterator, bool> ins_res;

    cphvb_array_ptr op;

    bool do_fuse = true;

    std::cout << "BUNDLING " << size << " {" << std::endl;
    for(cphvb_intp i=0; ((do_fuse) && (i<size)); i++) {

        pp_instr( insts[i] );

        op = insts[i]->operand[0];

        if ( out.count(op) > 0 ) {                                  // Exactly the same array
                                                                    // All good just continue
            bundle_len++;

        } else if (op->base == NULL) {                              // Base - first sighting

            out.insert( op );
            bundle_len++;

        } else if (out.empty()) {                                   // View - no clashes possible

            out.insert( op );
            bundle_len++;

        } else {                                                    // View - clashes possible 
                                                                    
            for(it = out.begin(); it != out.end(); it++) {          // Determine splicability

                if ( op->base == (*it)->base ) {                    // Same base

                    if ((op->ndim == (*it)->ndim) &&                // Same dim and start
                        (op->start == (*it)->start)) {
                        
                        for(cphvb_intp j =0; i<op->ndim; i++) {
                            if ((op->stride[j] != (*it)->stride[j]) ||
                                (op->shape[j] != (*it)->shape[j])) {
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

            if (do_fuse) {
                out.insert( op );
                bundle_len++;
            }

        }

    }

    std::cout << "} out {" << std::endl << "  ";
    for(it = out.begin(); it != out.end(); it++)
    {
        std::cout << *it << ",";
    }
    std::cout << std::endl;
    std::cout << "} " << bundle_len  << std::endl;
    
    return bundle_len;

}
