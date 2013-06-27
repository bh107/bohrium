/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#include "bh.h"
#include "bh_bundler.h"
#include <iostream>
#include <map>
#include <set>

typedef bh_view* bh_view_ptr;
typedef bh_base* bh_base_ptr;

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
inline bool ops_aligned( bh_view_ptr op_l, bh_view_ptr op_r) {

    if ((op_l->ndim != op_r->ndim) || (op_l->start != op_r->start)) // Check dim and start
    {
        return false;                                               // Incompatible dim or start

    } else {

        for(bh_intp i=0; i < op_l->ndim; i++)                    // Check shape and stride
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
 * WARN: This function ignores sys-ops by simply incrementing the bundle-size when sys-ops are encountered.
 * It is the responsibility of the caller to handle the sys-ops.
 *
 * @param inst A list of instructions.
 * @param start Start from and with instruction with index 'start'.
 * @param end Stop at and with instruction with index 'end'.
 * @return Number of consecutive bundleable instructions.
 *
 */
bh_intp bh_inst_bundle(bh_instruction *insts, bh_intp start, bh_intp end, bh_intp base_max)
{

    std::multimap<bh_base_ptr, bh_view_ptr> ops;            // Operands in kernel
    std::multimap<bh_base_ptr, bh_view_ptr> ops_out;        // Output-operands in kernel
    std::multimap<bh_base_ptr, bh_view_ptr>::iterator it;   // it / ret = Iterators
    std::pair<
        std::multimap<bh_base_ptr, bh_view_ptr>::iterator,
        std::multimap<bh_base_ptr, bh_view_ptr>::iterator
    > ret;

    std::pair<std::set<bh_base_ptr>::iterator, bool> base_ret;
    std::set<bh_base_ptr> bases; // List of distinct bases seen so far.
    int base_count = 0;          // How many distinct bases seen to far
    //int base_max = 5;          // Max amount of bases in bundle
                                 // This will be made parameterizable

    bool do_fuse = true;                                         // Loop invariant
    bh_intp bundle_len = 0;                                      // Number of cons. bundl. instr.
                                                                 // incremented on each iteration

    int opcount = 0;                                             // Per-instruction variables
    bh_view_ptr op;                                              // re-assigned on each iteration.
    bh_base_ptr base;
    bh_index nelements = 0;                                      // Get the number of elements
    for(bh_intp i=start; i<= end; i++) {
         switch(insts[i].opcode) {
            case BH_DISCARD:
            case BH_FREE:
            case BH_SYNC:
            case BH_NONE:
                continue;
        }
        nelements = bh_nelements( insts[i].operand[0].ndim, insts[i].operand[0].shape );
    }

    for(bh_intp i=start; ((do_fuse) && (i<=end)); i++)           // Go through the instructions...
    {
        switch(insts[i].opcode) {                                   // Ignore sys-ops
            case BH_DISCARD:
            case BH_FREE:
            case BH_SYNC:
            case BH_NONE:
                bundle_len++;
                continue;
        }
        opcount = bh_operands(insts[i].opcode);
                                                                    // Check for collisions
        op      = &insts[i].operand[0];                             // Look at the output-operand
        base    = bh_base_array( op );

        if (bh_nelements(op->ndim, op->shape) != nelements) {
            do_fuse = false;
            break;
        }

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
            op = &insts[i].operand[j];
            if (bh_is_constant( op )) {                             // Ignore constants
                break;
            }
            base = bh_base_array( op );

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
            op      = &insts[i].operand[0];                          // Add operand(s) to "kernel"
            base    = bh_base_array( op );                       //
                                                                    // - output operand
            ops.insert(     std::pair<bh_base_ptr, bh_view_ptr>( base, op ) );
            ops_out.insert( std::pair<bh_base_ptr, bh_view_ptr>( base, op ) );

            base_ret = bases.insert( base );                        // Update base count
            if (base_ret.second) {
                base_count++;
            }

            for(int j=1; j < opcount; j++)                          // - input operand(s)
            {

                op      = &insts[i].operand[j];
                if (bh_is_constant(op)) {                        // Ignore constants
                    break;
                }
                base    = bh_base_array( op );
                ops.insert( std::pair<bh_base_ptr, bh_view_ptr>( base, op ) );

                base_ret = bases.insert( base );                    // Update base count
                if (base_ret.second) {
                    base_count++;
                }

            }

            do_fuse = base_count <= base_max;                       // Check whether we break base-threshold

        }

    }

    #ifdef DEBUG_BNDL
    if (bundle_len > 1)
    {
        /*
        std::cout << "BUNDLING " << end-start << " {" << std::endl;
        for(bh_intp i=start; ((do_fuse) && (i<=end)); i++)
        {
            bh_pprint_instr( &insts[i] );
        }
        std::cout << "} ops {" << std::endl << "  ";
        for(it = ops.begin(); it != ops.end(); it++)
        {
            std::cout << it->first << "," << it->second << std::endl;
        }
        std::cout << "} bundle len = [" << bundle_len << "]" << std::endl;
        */

        std::cout << "{";
        for(it = ops.begin(); it != ops.end(); it++)
        {
            std::cout << it->first << "," << it->second << std::endl;
        }
        std::cout << std::endl;

        std::cout << "} bundle len = [" << bundle_len << "]" << std::endl;
    }
    #endif

    return bundle_len;

}
