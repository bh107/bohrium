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
#include <set>

/* Calculates the bundleable instructions.
 *
 * @inst The instruction list
 * @size Size of the instruction list
 * @return Number of consecutive bundeable instruction
 */
cphvb_intp bundle(cphvb_instruction *insts[], cphvb_intp size)
{

    cphvb_intp res = 1;                                             // Number of cons. bundl. instr.
    
    std::set<cphvb_data_ptr> in, out;                               // Sets for classifying operands.
    std::set<cphvb_data_ptr>::iterator in_it;                       // NOTE: Use boost::set? It is O( k )!
    std::set<cphvb_data_ptr>::iterator out_it;
    std::pair<std::set<cphvb_data_ptr>::iterator, bool> ins_res;

    for(cphvb_intp i=0; i<size; i++)
    {

        // determine bundlability...
        // res++

    }

    return res;
}
