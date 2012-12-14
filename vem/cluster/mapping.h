/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __CPHVB_VEM_CLUSTER_MAPPING_H
#define __CPHVB_VEM_CLUSTER_MAPPING_H

#include <cphvb.h>
#include "array.h"
#include <vector>

/* Creates a list of local array chunks that enables local
 * execution of the instruction
 *
 * @nop         Number of global array operands
 * @operand     List of global array operands
 * @chunks      The output chunks
 * @chunks_ext  The output chunks extention
 * @return      Error codes (CPHVB_SUCCESS, CPHVB_ERROR)
 */
cphvb_error mapping_chunks(cphvb_intp nop,
                           cphvb_array *operand[],
                           std::vector<cphvb_array>& chunks,  
                           std::vector<array_ext>& chunks_ext);

#endif
