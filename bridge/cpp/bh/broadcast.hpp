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

/**
 * Determine whether or not the shapes of the provides operands are the same.
 */
template <typename T>
inline
bool same_shape(multi_array<T> & left, multi_array<T> & right)
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
 * Broadcast operands.
 *
 * @param lower,higher 'lower' much have a rank <= to the rank of 'higher'.
 * @param view Is a view on 'lower'; It will contain the resulting broadcast shape/stride/ndim.
 * 
 * @return Whether or not the operand is broadcastable.
 *
 */
template <typename T>
inline
bool broadcast(multi_array<T>& lower, multi_array<T>& higher, multi_array<T>& view)
{
    bh_array *lower_a   = &storage[lower.getKey()];     // The operand which will be "stretched"
    bh_array *higher_a  = &storage[higher.getKey()];    // The possibly "larger" shape
    bh_array *view_a    = &storage[view.getKey()];      // The new "broadcasted" shape
    bool broadcastable  = true;
    
    bh_intp stretch_dim = lower_a->ndim-1;              // Checks: shape compatibility
    bh_intp operand_dim = higher_a->ndim-1;             // Create: shape and stride.

    while((stretch_dim>=0) && broadcastable) {             
        broadcastable =   ((lower_a->shape[stretch_dim] == higher_a->shape[operand_dim]) || \
                        (lower_a->shape[stretch_dim] == 1) || \
                        (higher_a->shape[operand_dim] == 1)
                        );

        view_a->shape[operand_dim] = higher_a->shape[operand_dim] >= lower_a->shape[stretch_dim] ? \
                                        higher_a->shape[operand_dim] : \
                                        lower_a->shape[stretch_dim];

        view_a->stride[operand_dim] = higher_a->shape[operand_dim] > lower_a->shape[stretch_dim] ? \
                                        0 : \
                                        lower_a->stride[stretch_dim];

        stretch_dim--;
        operand_dim--;
    }
                                                        // Copy the remaining shapes.
    memcpy(view_a->shape, higher_a->shape, (operand_dim+1) * sizeof(bh_index));
                                                        // And set the remaining strides.
    memset(view_a->stride, 0, (operand_dim+1) * sizeof(bh_index));

    view_a->ndim = higher_a->ndim;                   // Set ndim

    return broadcastable;
}

}
#endif
