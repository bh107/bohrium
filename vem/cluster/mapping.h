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

#ifndef __BH_VEM_CLUSTER_MAPPING_H
#define __BH_VEM_CLUSTER_MAPPING_H

#include <bh.h>
#include "array.h"
#include <vector>

/* Creates a list of local array chunks that enables local
 * execution of the instruction
 *
 * @nop         Number of global array operands
 * @operands    List of global array operands
 * @chunks      The output chunks
 */
void mapping_chunks(bh_intp nop,
                    bh_view *operands,
                    std::vector<ary_chunk>& chunks);

#endif
