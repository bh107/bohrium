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
#include <bh.h>

namespace bxx {

/**
 * Determine whether or not the shapes of the provides operands are the same.
 */
template <typename LT, typename RT>
inline
bool same_shape(const multi_array<LT> & left, const multi_array<RT> & right)
{
    bool compatible = left.meta.ndim == right.meta.ndim;

    for(int64_t dim=0; compatible && (dim < right.meta.ndim-1); dim++) {
        compatible = (left.meta.shape[dim] == right.meta.shape[dim]);
    }

    return compatible;
}

template <typename LT, typename RT>
inline
bool broadcast_right(const multi_array<LT>& left, multi_array<RT>& right)
{
    int64_t highest_dim  = left.meta.ndim - 1;
    int64_t stretch_dims = left.meta.ndim - right.meta.ndim;
    bool broadcastable   = true;

    if (stretch_dims>0) {       // Stretch the dimensions
        for(int64_t i=0; i<stretch_dims; i++) {
            right.meta.shape[highest_dim-i]  = left.meta.shape[highest_dim-i];
            right.meta.stride[highest_dim-i] = 0;
        }
        right.meta.ndim = left.meta.ndim;                 // Set ndim
    }

    //
    // Fix the shapes
    for(int64_t dim_idx=0; (dim_idx <= highest_dim) && (broadcastable); dim_idx++) {
        broadcastable = ((right.meta.shape[dim_idx] == left.meta.shape[dim_idx]) || \
                         (right.meta.shape[dim_idx] == 1)
                        );

        if (right.meta.shape[dim_idx] < left.meta.shape[dim_idx]) {
            right.meta.shape[dim_idx]   = left.meta.shape[dim_idx];
            right.meta.stride[dim_idx]  = 0;
        }
    }

    return broadcastable;
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
template <typename LT, typename HT>
inline
bool broadcast(multi_array<LT>& lower, multi_array<HT>& higher)
{
    int64_t highest_dim  = higher.meta.ndim - 1;
    int64_t stretch_dims = higher.meta.ndim - lower.meta.ndim;
    bool broadcastable   = true;

    if (stretch_dims>0) {       // Stretch the dimensions
        for(int64_t i=0; i<stretch_dims; i++) {
            lower.meta.shape[highest_dim-i]  = higher.meta.shape[highest_dim-i];
            lower.meta.stride[highest_dim-i] = 0;
        }
        lower.meta.ndim = higher.meta.ndim;                 // Set ndim
    }

    //
    // Fix the shapes
    for(int64_t dim_idx=0; (dim_idx <= highest_dim) && (broadcastable); dim_idx++) {
        broadcastable = ((lower.meta.shape[dim_idx] == higher.meta.shape[dim_idx]) || \
                         (lower.meta.shape[dim_idx] == 1) || \
                         (higher.meta.shape[dim_idx] == 1)
                        );

        if (lower.meta.shape[dim_idx] < higher.meta.shape[dim_idx]) {
            lower.meta.shape[dim_idx]   = higher.meta.shape[dim_idx];
            lower.meta.stride[dim_idx]  = 0;
        } else if (higher.meta.shape[dim_idx] < lower.meta.shape[dim_idx]) {
            higher.meta.shape[dim_idx]  = lower.meta.shape[dim_idx];
            higher.meta.stride[dim_idx] = 0;
        }
    }

    return broadcastable;
}

/**
 * Broadcast operands.
 *
 * @param lower,higher 'lower' much have a rank <= to the rank of 'higher'.
 *
 * @return Whether or not the operand is broadcastable.
 *
 */
template <typename OT, typename LT, typename HT>
inline
bool broadcast(const multi_array<OT>& output, multi_array<LT>& lower, multi_array<HT>& higher)
{
    bool broadcastable = true;

    broadcastable = broadcastable && broadcast_right(output, lower);
    broadcastable = broadcastable && broadcast(lower, higher);
    broadcastable = broadcastable && same_shape(output, lower);

    return broadcastable;
}

}
#endif
