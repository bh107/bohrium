/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
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

inline void bh_pprint_shape(int64_t shape[], int64_t len)
{
    std::cout << "Shape: ";
    for(int64_t k=0; k<len; k++) {
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
    bool compatible = left.meta.ndim == right.meta.ndim;

    for(int64_t dim=right.meta.ndim-1; compatible && (dim < right.meta.ndim-1); dim++) {
        compatible = (left.meta.shape[dim] == right.meta.shape[dim]);
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
    bool broadcastable  = true;
    int64_t stretch_dim = lower.meta.ndim-1;              // Checks: shape compatibility
    int64_t operand_dim = higher.meta.ndim-1;             // Create: shape and stride.

    while((stretch_dim>=0) && broadcastable) {
        broadcastable = ((lower.meta.shape[stretch_dim] == higher.meta.shape[operand_dim]) || \
                         (lower.meta.shape[stretch_dim] == 1) || \
                         (higher.meta.shape[operand_dim] == 1)
                        );

        view.meta.shape[operand_dim] = higher.meta.shape[operand_dim] >= lower.meta.shape[stretch_dim] ? \
                                        higher.meta.shape[operand_dim] : \
                                        lower.meta.shape[stretch_dim];

        view.meta.stride[operand_dim] = higher.meta.shape[operand_dim] > lower.meta.shape[stretch_dim] ? \
                                        0 : \
                                        lower.meta.stride[stretch_dim];

        stretch_dim--;
        operand_dim--;
    }
                                                        // Copy the remaining shapes.
    memcpy(view.meta.shape, higher.meta.shape, (operand_dim+1) * sizeof(int64_t));
                                                        // And set the remaining strides.
    memset(view.meta.stride, 0, (operand_dim+1) * sizeof(int64_t));

    view.meta.ndim = higher.meta.ndim;                  // Set ndim
    view.meta.start = lower.meta.ndim;                  // Copy start offset

    return broadcastable;
}

}
#endif
