/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://bohrium.bitbucket.org

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
#ifndef __BOHRIUM_BRIDGE_CPP_BROADCAST
#define __BOHRIUM_BRIDGE_CPP_BROADCAST
#include "bh.h"

namespace bh {

void bh_pprint_shape(bh_index shape[], bh_intp len)
{
    std::cout << "Shape: ";
    for(bh_intp k=0; k<len; k++) {
        std::cout << shape[k];
        if (k<len-1) {
            std::cout << ", ";
        }
    }
    std::cout << "." << std::endl;
}

template <typename T>
inline
bool compatible_shape(multi_array<T> & left, multi_array<T> & right)
{
    bh_array *left_a     = &storage[left.getKey()];
    bh_array *right_a    = &storage[right.getKey()];
    bool compatible = left_a->ndim == right_a->ndim;

    for(bh_index dim=right_a->ndim-1; compatible && (dim < right_a->ndim-1); dim++) {
        compatible = (left_a->shape[dim] == right_a->shape[dim]);
    }

    return compatible;
}

/**
 * Broadcast shape of operands.
 *
 * @return Whether or not the shapes are compatible.
 *
 */
template <typename T>
inline
bool broadcast_shape(multi_array<T> & stretch, multi_array<T> operand, multi_array<T> output)
{
    bool compatible = true;
    
    bh_array *stretch_a = &storage[stretch.getKey()];   // The operand which will be "stretched"
    bh_array *operand_a = &storage[operand.getKey()];   // The possibly larger shape
    bh_array *output_a  = &storage[output.getKey()];    // The new "broadcasted" shape

    bh_intp stretch_dim = stretch_a->ndim-1;            
    bh_intp operand_dim = operand_a->ndim-1;

    while((stretch_dim>=0) && compatible) {             // Shape compatibility-check + copy
        compatible =   ((stretch_a->shape[stretch_dim] == operand_a->shape[operand_dim]) || \
                        (stretch_a->shape[stretch_dim] == 1) || \
                        (operand_a->shape[operand_dim] == 1)
                        );

        output_a->shape[operand_dim] = operand_a->shape[operand_dim] >= stretch_a->shape[stretch_dim] ? \
                            operand_a->shape[operand_dim] : \
                            stretch_a->shape[stretch_dim];
        stretch_dim--;
        operand_dim--;
    }
                                                        // Copy the remaining shapes
    memcpy(output_a->shape, operand_a->shape, (operand_dim+1) * sizeof(bh_index));

    output_a->ndim = operand_a->ndim;                   // Set ndim
                                                        // TODO: Set remaining meta-data.
                                                        //       This should probably be
                                                        //       managed somewhere else...

    bh_pprint_array( stretch_a );
    bh_pprint_array( operand_a );
    bh_pprint_array( output_a );

    return compatible;
}

}
#endif
