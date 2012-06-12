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
#include "cphvb.h"
#include "cphvb_bundler.h"
#include <iostream>
#include <map>

typedef cphvb_array* cphvb_array_ptr;

/**
 * Determines whether two operands are aligned.
 * Specifically whether the following meta-data is equal to one another:
 *
 * ndim, start, shape, and stride.
 *
 * NOTE: base is not checked for equality.
 *
 * @param op_l Some operand
 * @param op_r Some other operand to compare with
 *
 * @return True when aligned, false when they are not.
 *
 */
inline bool ops_aligned( cphvb_array_ptr op_l, cphvb_array_ptr op_r) {

    if ((op_l->ndim != op_r->ndim) || (op_l->start != op_r->start)) // Check dim and start
    {
        return false;                                               // Incompatible dim or start

    } else {
                                                                    
        for(cphvb_intp i=0; i < op_l->ndim; i++)                    // Check shape and stride
        {
            if ((op_l->stride[i] != op_r->stride[i]) || (op_l->shape[i] != op_r->shape[i]))
            {                                                       
                return false;                                       // Incompatible shape or stride
            }
        }

        // Reaching this point means that the operands are aligned aka they have equal:
        // ndim, start, shape and stride.
        return true;
    }

}

/**
 * Calculates the bundleable instructions.
 *
 * @param inst A list of instructions.
 * @param start Start from and with instruction with index 'start'.
 * @param end Stop at and with instruction with index 'end'.
 * @return Number of consecutive bundleable instructions.
 *
 */
cphvb_intp cphvb_inst_bundle(cphvb_instruction *insts, cphvb_intp start, cphvb_intp end)
{

    std::multimap<cphvb_array_ptr, cphvb_array_ptr> ops;            // Operands in kernel
    std::multimap<cphvb_array_ptr, cphvb_array_ptr> ops_out;        // Output-operands in kernel
    std::multimap<cphvb_array_ptr, cphvb_array_ptr>::iterator it;   // it / ret = Iterators
    std::pair< 
        std::multimap<cphvb_array_ptr, cphvb_array_ptr>::iterator, 
        std::multimap<cphvb_array_ptr, cphvb_array_ptr>::iterator
    > ret;

    bool do_fuse = true;                                            // Loop invariant
    cphvb_intp bundle_len = 0;                                      // Number of cons. bundl. instr.
                                                                    // incremented on each iteration

    int opcount = 0;                                                // Per-instruction variables
    cphvb_array_ptr op, base;                                       // re-assigned on each iteration.

    for(cphvb_intp i=start; ((do_fuse) && (i<=end)); i++)               // Go through the instructions...
    {

        opcount = cphvb_operands(insts[i].opcode);
                                                                    // Check for collisions
        op      = insts[i].operand[0];                             // Look at the output-operand
        base    = cphvb_base_array( op );

        ret = ops.equal_range( base );                              // Compare to all kernel operands.
        for(it = ret.first; it != ret.second; ++it)
        {
            if (!ops_aligned( op, (*it).second ))
            {
                do_fuse = false;
                break;
            }
        }
                                                                    
        for(int j=1; ((do_fuse) && (j<opcount)); j++)               // Look at the input-operands
        {
            op      = insts[i].operand[j];
            base    = cphvb_base_array( op );

            if (!cphvb_is_constant( op )) {                         // Ignore constants
                break;
            }

            ret = ops_out.equal_range( base );                      // Compare to kernel-output-operands
            for(it = ret.first; it != ret.second; ++it)
            {
                if (!ops_aligned( op, (*it).second ))
                {
                    do_fuse = false;
                    break;
                }
            }

        }

        if (do_fuse)                                                // Instruction is allowed
        {
            bundle_len++;                                           // Increment bundle
                                                                    //
            op      = insts[i].operand[0];                         // Add operand(s) to "kernel"
            base    = cphvb_base_array( op );                       //
                                                                    // - output operand
            ops.insert(     std::pair<cphvb_array_ptr, cphvb_array_ptr>( base, op ) );
            ops_out.insert( std::pair<cphvb_array_ptr, cphvb_array_ptr>( base, op ) );

            for(int j=1; j < opcount; j++)                          // - input operand(s)
            {

                op      = insts[i].operand[j];
                if (cphvb_is_constant(op)) {                        // Ignore constants
                    break;
                }
                base    = cphvb_base_array( op );
                ops.insert( std::pair<cphvb_array_ptr, cphvb_array_ptr>( base, op ) );
            }
        }

    }

    #ifdef DEBUG_BNDL
    if (bundle_len > 1)
    {
        std::cout << "BUNDLING " << end-start << " {" << std::endl;
        for(cphvb_intp i=start; ((do_fuse) && (i<=end)); i++)
        {
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

    return bundle_len;

}
